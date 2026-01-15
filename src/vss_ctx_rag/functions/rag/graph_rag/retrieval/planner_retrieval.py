# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import traceback
from typing import Optional, ClassVar, Dict, List

from langchain_core.runnables import RunnableConfig
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field
from langgraph.errors import GraphRecursionError
import yaml

from vss_ctx_rag.functions.rag.graph_rag.retrieval.graph_retrieval_base import (
    GraphRetrievalBaseFunc,
)
from vss_ctx_rag.functions.rag.graph_rag.retrieval.planner import get_agent
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_CHAT_HISTORY,
    DEFAULT_NUM_FRAMES_PER_CHUNK,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from vss_ctx_rag.tools.image.image_fetcher import ImageFetcher


@register_function_config("adv_graph_retrieval")
class PlannerRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "vlm": ["llm"],
        "neo4j": ["db"],
        "image_fetcher": ["image_fetcher"],
    }

    class PlannerRetrievalParams(RetrieverConfig.RetrieverParams):
        tools: Optional[List[str]] = Field(default=None)
        multi_channel: Optional[bool] = Field(default=False)
        multi_choice: Optional[bool] = Field(default=False)
        max_iterations: Optional[int] = Field(default=20)
        num_frames_per_chunk: Optional[int] = Field(
            default=DEFAULT_NUM_FRAMES_PER_CHUNK
        )
        include_adjacent_chunks: Optional[bool] = Field(default=False)
        pass_video_to_vlm: Optional[bool] = Field(default=False)
        num_prev_chunks: Optional[int] = Field(default=1)
        num_next_chunks: Optional[int] = Field(default=1)

    params: PlannerRetrievalParams


@register_function(config=PlannerRetrievalConfig)
class Planner(GraphRetrievalBaseFunc):
    """
    Planner Function that uses iterative thinking and execution for video analysis.
    """

    config: dict
    output_parser = StrOutputParser()
    graph_db: GraphStorageTool
    image_fetcher: ImageFetcher
    metrics = GraphMetrics()

    def setup(self) -> None:
        """
        Setup the Planner class.

        Args:
            None

        Instance variables:
            self.graph_db: GraphStorageTool
            self.llm: LLMTool
            self.vlm: VLM Tool
            self.top_k: int
            self.uuid: str
            self.agent: Compiled graph agent

        Returns:
            None
        """
        super().setup()

        # Get VLM tool for image analysis
        self.vlm = self.get_tool("vlm")
        self.llm = self.get_tool("llm")

        self.graph_db = self.get_tool("db")
        self.image_fetcher = self.get_tool("image_fetcher")
        self.multi_channel = self.get_param("multi_channel", default=False)
        self.multi_choice = self.get_param("multi_choice", default=False)
        self.num_prev_chunks = self.get_param("num_prev_chunks", default=1)
        self.num_next_chunks = self.get_param("num_next_chunks", default=1)
        # Get max iterations parameter
        self.max_iterations = self.get_param("max_iterations", default=20)
        self.tools = self.get_param("tools", default=None)
        self.num_frames_per_chunk = self.get_param(
            "num_frames_per_chunk", default=DEFAULT_NUM_FRAMES_PER_CHUNK
        )
        self.include_adjacent_chunks = self.get_param(
            "include_adjacent_chunks", default=False
        )
        self.pass_video_to_vlm = self.get_param("pass_video_to_vlm", default=False)
        self.prompt_config_path = self.get_param("prompt_config_path", default=None)
        self.prompt_config = None
        if self.prompt_config_path:
            try:
                with open(self.prompt_config_path, "r") as f:
                    self.prompt_config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading prompt config: {str(e)}")
                self.prompt_config = None
        try:
            # Create the planner agent using the adv_graph_retrieval logic
            self.agent = get_agent(
                llm=self.llm,
                vlm=self.vlm,
                graph_db=self.graph_db,
                uuids=self.uuid,
                multi_channel=self.multi_channel,
                top_k=self.top_k,
                multi_choice=self.multi_choice,
                tools=self.tools,
                image_fetcher=self.image_fetcher,
                num_frames_per_chunk=self.num_frames_per_chunk,
                include_adjacent_chunks=self.include_adjacent_chunks,
                pass_video_to_vlm=self.pass_video_to_vlm,
                num_prev_chunks=self.num_prev_chunks,
                num_next_chunks=self.num_next_chunks,
                prompt_config=self.prompt_config,
            )
            logger.info(f"Initialized planner agent with top_k={self.top_k}")
        except Exception as e:
            logger.error(f"Error initializing planner agent: {e}")
            raise

        self.chat_history = self.get_param("chat_history", default=DEFAULT_CHAT_HISTORY)

    async def acall(self, state: dict) -> dict:
        """
        Call the Planner agent for iterative thinking and execution.

        Args:
            state: State of the function containing the question.

        Returns:
            State of the function with the response.
        """
        with Metrics(
            "Planner/call", "blue", span_kind=Metrics.SPAN_KIND["AGENT"]
        ) as tm:
            tm.input({"state": state})
            try:
                question = state.get("question", "").strip()
                is_live = state.get("is_live", False)
                is_subtitle = state.get(
                    "is_subtitle", False
                )  # Additional tools from state

                if not question:
                    logger.debug("No question provided in state")
                    state["response"] = "Please provide a question"
                    return state

                if question.lower() == "/clear":
                    logger.debug("Clearing chat history...")
                    # Reset the agent state
                    state["response"] = "Cleared chat history"
                    return state

                logger.info(f"Processing question with planner: {question}")

                # Check if we need to recreate agent with additional tools
                if is_subtitle:
                    subtitles_tools = ["subtitle_search", "subtitle_filter"]
                    logger.info(f"Adding subtitles tools: {subtitles_tools}")
                    # Combine base tools with runtime tools
                    extended_tools = (
                        subtitles_tools + self.tools if self.tools else subtitles_tools
                    )
                    # Recreate agent with extended tools
                    agent = get_agent(
                        llm=self.llm,
                        vlm=self.vlm,
                        graph_db=self.graph_db,
                        uuids=self.uuid,
                        multi_channel=self.multi_channel,
                        top_k=self.top_k,
                        multi_choice=self.multi_choice,
                        tools=extended_tools,
                        image_fetcher=self.image_fetcher,
                        num_frames_per_chunk=self.num_frames_per_chunk,
                        include_adjacent_chunks=self.include_adjacent_chunks,
                        pass_video_to_vlm=self.pass_video_to_vlm,
                        num_prev_chunks=self.num_prev_chunks,
                        num_next_chunks=self.num_next_chunks,
                        prompt_config=self.prompt_config,
                    )
                else:
                    # Use the existing agent
                    agent = self.agent

                num_cameras = self.graph_db.get_num_cameras()
                logger.debug(f"Runtime num_cameras: {num_cameras}")

                if not is_live:
                    video_length = self.graph_db.get_video_length()
                else:
                    video_length = {}
                logger.debug(f"Runtime video_length: {video_length}")

                chunk_size = self.graph_db.get_chunk_size()
                logger.debug(f"Runtime chunk_size: {chunk_size}")

                # Prepare the initial state for the agent
                agent_state = {
                    "messages": [{"role": "human", "content": question}],
                    "original_query": question,
                    "current_plan": "",
                    "execution_results": "",
                    "iteration_count": 0,
                    "max_iterations": self.max_iterations,
                    "thinking_complete": False,
                    "num_cameras": num_cameras,  # Pass runtime num_cameras
                    "video_length": video_length,  # Pass runtime video_length
                    "chunk_size": chunk_size,  # Pass runtime chunk_size
                    "is_subtitle": is_subtitle,  # Pass runtime is_subtitle
                }

                # Run the planner agent
                logger.info("Starting planner agent execution...")
                config = RunnableConfig(recursion_limit=self.max_iterations)
                result = await agent.ainvoke(agent_state, config=config)

                # Extract the final response from the agent result
                final_response = ""
                if "messages" in result and result["messages"]:
                    # Get the last message which should be the final response
                    for msg in reversed(result["messages"]):
                        if hasattr(msg, "content") and msg.content:
                            final_response = msg.content
                            break

                if not final_response:
                    final_response = "No response generated from planner agent"

                logger.info(f"Planner agent response: {final_response}")

                # Update state with response
                state["response"] = final_response

                # Store execution results if available
                if "execution_results" in result:
                    state["execution_results"] = result["execution_results"]

                # Store iteration count for debugging
                if "iteration_count" in result:
                    state["iteration_count"] = result["iteration_count"]
            except GraphRecursionError as e:
                state["response"] = (
                    "Sorry, I'm having trouble answering that question. Please try again."
                )
                state["error"] = str(e)
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error("Error in Planner %s", str(e))
                state["error"] = str(e)
                state["response"] = "Error in Planner"

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict) -> str:
        """
        Process a document.

        Args:
            doc: Document to process.
            doc_i: Index of the document.
            doc_meta: Metadata of the document.

        Returns:
            Success.
        """
        pass

    async def areset(self, state: dict) -> None:
        """
        Reset the Planner class.

        Args:
            state: State of the function.

        Returns:
            None
        """
        # Reset any internal state if needed
        await asyncio.sleep(0.01)
