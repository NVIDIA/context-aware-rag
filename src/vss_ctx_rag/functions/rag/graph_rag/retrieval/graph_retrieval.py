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
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_CHAT_HISTORY,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from vss_ctx_rag.models.state_models import RetrieverFunctionState


@register_function_config("graph_retrieval")
class GraphRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "neo4j": ["db"],
        "arango": ["db"],
    }

    params: RetrieverConfig.RetrieverParams


@register_function(config=GraphRetrievalConfig)
class GraphRetrievalFunc(GraphRetrievalBaseFunc):
    """
    GraphRetrieval Function for the ArangoDB backend.
    """

    config: dict
    output_parser = StrOutputParser()
    graph_db: GraphStorageTool
    metrics = GraphMetrics()

    def setup(self) -> None:
        """
        Setup the GraphRetrieval class.

        Args:
            None

        Instance variables:
            self.graph_db: GraphStorageTool
            self.chat_llm: LLMTool
            self.top_k: int
            self.uuid: str
            self.graph_retrieval: GraphRetrieval

        Returns:
            None
        """
        super().setup()
        try:
            self.graph_retrieval = GraphRetrieval(
                llm=self.chat_llm,
            )
        except Exception as e:
            logger.error(f"Error initializing GraphRetrieval: {e}")
            raise

        self.chat_history = self.get_param("chat_history", default=DEFAULT_CHAT_HISTORY)
        self.image = self.get_param("image", default=False)

    async def acall(self, state: RetrieverFunctionState) -> RetrieverFunctionState:
        """
        Call the GraphRetrieval class.

        Args:
            state (RetrieverFunctionState): State of the function containing:
                Required keys:
                    - question (str): The input question/query to process
                Optional keys:
                    - response_method (str): Method for response generation
                    - response_schema (dict): Schema for structured response
                    - retriever_type (str): Type of retriever to use ("chunk", "entity", or "subtitle", default: "chunk")

        Returns:
            RetrieverFunctionState: Updated state containing:
                - response (str): Generated response to the question
                - source_docs (list): Retrieved source documents used for response
                - error (str, optional): Error message if processing failed
        """

        try:
            question = state.get("question", "").strip()
            retriever_type = state.get("retriever_type", "chunk")

            logger.info(f"Retriever type for graph retrieval: {retriever_type}")

            if not question:
                raise ValueError("No input provided in state.")

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.graph_retrieval.clear_chat_history()
                state["response"] = "Cleared chat history"
                return state

            with Metrics("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=question)
                self.graph_retrieval.add_message(user_message)

            transformed_question = (
                await self.graph_retrieval.question_transform_chain.ainvoke(
                    {"messages": self.graph_retrieval.chat_history.messages}
                )
            )

            documents, raw_docs = self.graph_db.retrieve_documents(
                question=transformed_question,
                uuid=self.uuid,
                multi_channel=self.multi_channel,
                top_k=self.top_k,
                retriever=retriever_type,
            )

            if not documents:
                if self.graph_retrieval.chat_history.messages:
                    self.graph_retrieval.chat_history.messages.pop()

            image_list_base64 = []
            if self.image:
                image_list_base64 = await self.extract_images(raw_docs)

            response = await self.graph_retrieval.get_response(
                question,
                documents,
                image_list_base64,
                response_method=state.get("response_method"),
                response_schema=state.get("response_schema"),
            )

            logger.info(f"AI response: {response}")

            state["response"] = response
            state["source_docs"] = raw_docs
            state["formatted_docs"] = [i["page_content"] for i in raw_docs]

            if self.chat_history:
                with Metrics("GraphRetrieval/AIMsg", "red"):
                    ai_message = AIMessage(content=response)
                    self.graph_retrieval.add_message(ai_message)

                self.graph_retrieval.summarize_chat_history()

                logger.debug("Summarizing chat history thread started.")
            else:
                self.graph_retrieval.clear_chat_history()

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in GraphRetrievalFunc %s", str(e))
            state["response"] = "Sorry, something went wrong. Please try again."
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
