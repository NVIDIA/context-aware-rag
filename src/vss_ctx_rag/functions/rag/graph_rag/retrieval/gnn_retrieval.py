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

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from typing import ClassVar, Dict, List

from vss_ctx_rag.functions.rag.graph_rag.retrieval.base import GraphRetrieval
from vss_ctx_rag.functions.rag.graph_rag.retrieval.graph_retrieval_base import (
    GraphRetrievalBaseFunc,
)
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.tools.gnn.gnn_tool import GnnTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_CHAT_HISTORY,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig


@register_function_config("gnn_retrieval")
class GnnRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "neo4j": ["db"],
        "arango": ["db"],
        "gnn": ["gnn"],
    }

    params: RetrieverConfig.RetrieverParams


@register_function(config=GnnRetrievalConfig)
class GnnRetrievalFunc(GraphRetrievalBaseFunc):
    """
    GNN-based Retrieval Function for the ArangoDB backend.
    """

    config: dict
    output_parser = StrOutputParser()
    graph_db: GraphStorageTool
    gnn_tool: GnnTool
    metrics = GraphMetrics()

    def setup(self) -> None:
        """
        Setup the GraphRetrieval class.

        Args:
            None

        Instance variables:
            self.graph_db: GraphStorageTool
            self.chat_llm: LLMTool
            self.gnn_tool: GnnTool
            self.top_k: int
            self.uuid: str
            self.graph_retrieval: GraphRetrieval

        Returns:
            None
        """
        super().setup()
        # Retrieve the GNN tool added to this function (keyword should be 'gnn')
        self.gnn_tool = self.get_tool("gnn")
        if self.gnn_tool is None:
            raise ValueError(
                "GNN tool with keyword 'gnn' is not configured for GnnRetrievalFunc"
            )
        try:
            self.graph_retrieval = GraphRetrieval(
                llm=self.chat_llm,
            )
        except Exception as e:
            logger.error(f"Error initializing GraphRetrieval: {e}")
            raise

        self.chat_history = self.get_param("chat_history", default=DEFAULT_CHAT_HISTORY)
        self.headers = {"Content-Type": "application/json"}

    async def acall(self, state: dict) -> dict:
        """
        Call the GnnRetrieval class.

        Args:
            state: State of the function.

        Returns:
            State of the function.
        """
        try:
            question = state.get("question", "").strip()
            if not question:
                raise ValueError("No input provided in state.")

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.graph_retrieval.clear_chat_history()
                state["response"] = "Cleared chat history"
                return state
            if (
                state.get("response_method") is not None
                or state.get("response_schema") is not None
            ):
                logger.warning(
                    "Advanced Graph RAG does not support structured mode, ignoring response_method and response_schema"
                )

            with Metrics("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=question)
                self.graph_retrieval.add_message(user_message)

                transformed_question = (
                    self.graph_retrieval.question_transform_chain.invoke(
                        {"messages": self.graph_retrieval.chat_history.messages}
                    )
                )
                logger.debug(f"Transformed question: {transformed_question}")

                (payload_data, raw_docs) = self.graph_db.retrieve_documents_for_gnn(
                    transformed_question, self.uuid, self.multi_channel, self.top_k
                )
                if (
                    payload_data["nodes"]
                    and payload_data["edges"]
                    and payload_data["edge_indices"]
                    and len(payload_data["edge_indices"]) == 2
                ):
                    logger.debug(
                        f"Source node indices: {payload_data['edge_indices'][0]}"
                    )
                    logger.debug(
                        f"Target node indices: {payload_data['edge_indices'][1]}"
                    )
                    payload_data["question"] = question
                    logger.info("Enabled GNN for QnA")
                    answer = await self.gnn_tool.acall_inference(
                        payload_data, self.headers
                    )
                    if self.chat_history:
                        with Metrics("GraphRetrieval/AIMsg", "red"):
                            ai_message = AIMessage(content=answer)
                            self.graph_retrieval.add_message(ai_message)

                        self.graph_retrieval.summarize_chat_history()

                        logger.debug("Summarizing chat history thread started.")
                    else:
                        self.graph_retrieval.clear_chat_history()
                else:
                    answer = "Sorry, I don't see that in the video."
                    self.graph_retrieval.chat_history.messages.pop()

                state["response"] = self.graph_retrieval.regex_object.sub(
                    r"\g<1>", answer
                )
                state["source_docs"] = raw_docs
                state["formatted_docs"] = [i["page_content"] for i in raw_docs]

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in GnnRetrievalFunc %s", str(e))
            state["error"] = str(e)

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
        Reset the GraphRetrievalFuncArango class.

        Args:
            state: State of the function.

        Returns:
            None
        """
        self.graph_retrieval.clear_chat_history()
        await asyncio.sleep(0.01)
