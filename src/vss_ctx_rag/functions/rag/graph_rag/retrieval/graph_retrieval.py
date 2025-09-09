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
import os
import aiohttp
import json
from typing import Optional, Dict, Any

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from typing import ClassVar, Dict, List
from nvidia_rag.rag_server.main import NvidiaRAG

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
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger, TimeMeasure
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
        "vector_db": ["db"],
        "reranker": ["reranker"],
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

        # Enrichment prompt from config. The default is set in config.yaml
        self.enrichment_prompt: str = self.get_param("enrichment_prompt")

        # External RAG configuration
        self.external_rag_enabled = self.get_param("external_rag_enabled", default=False)
        if self.external_rag_enabled:
            self.vector_db = self.get_tool("vector_db")
            self.reranker_tool = self.get_tool("reranker")
            self.nvidia_rag = NvidiaRAG()
            collection_str = self.get_param("external_rag_collection", default="")
            self.external_rag_collection = [item.strip() for item in collection_str.split(',') if item.strip()]

    def _parse_search_results(self, search_results) -> str:
        """Parse search results from NvidiaRAG into a single string."""
        doc_list: List[str] = []
        for result in search_results.results:
            content = getattr(result, "content", "")
            doc_list.append(content)
        return "\n".join(doc_list)

    async def _get_external_rag_context(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get context from external RAG service using the NvidiaRAG tool."""
        
        if not self.external_rag_enabled:
            logger.info("External RAG is disabled, returning empty string")
            return ""

        if not self.external_rag_collection:
            logger.error("External RAG collections are required but not provided. Check the `external_rag_collection` parameter in the config.")
            return ""
            
        with TimeMeasure("external_rag/get_context", "blue"):
            try:
                if self.vector_db.embedding.base_url == "https://integrate.api.nvidia.com/v1":
                    embedding_endpoint = self.vector_db.embedding.base_url + "/embeddings"
                else:
                    embedding_endpoint = self.vector_db.embedding.base_url

                if self.reranker_tool:
                    search_results = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self.nvidia_rag.search(
                            query=query,
                            messages=[],
                            reranker_top_k=self.top_k,
                            vdb_top_k=self.top_k + 1,
                            collection_names=self.external_rag_collection,
                            vdb_endpoint=self.vector_db.connection["uri"],
                            enable_query_rewriting=True,
                            enable_reranker=True,
                            embedding_model=self.vector_db.embedding.model,
                            embedding_endpoint=embedding_endpoint,
                            reranker_model=self.reranker_tool.reranker.model,
                            reranker_endpoint=self.reranker_tool.reranker.base_url,
                        ),
                    )
                else:
                    search_results = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self.nvidia_rag.search(
                            query=query,
                            messages=[],
                            reranker_top_k=self.top_k,
                            vdb_top_k=self.top_k + 1,
                            collection_names=self.external_rag_collection,
                            vdb_endpoint=self.vector_db.connection["uri"],
                            enable_query_rewriting=True,
                            enable_reranker=False,
                            embedding_model=self.vector_db.embedding.model,
                            embedding_endpoint=embedding_endpoint,
                        ),
                    )
                
                context = self._parse_search_results(search_results)
                logger.info(f"External RAG context: {context[:100]}...")
                return context
            except Exception as e:
                logger.error(f"Error fetching from external RAG service: {e}")
                logger.error(traceback.format_exc())
                return ""

    def _extract_external_rag_query(self, text: str) -> tuple[str, str]:
        """Extract external RAG query from text marked with <e> tags.
        Returns (text_without_tags, external_rag_query)"""
        import re
        pattern = r'<e>(.*?)<e>'
        # Remove the tagged content and get clean text
        clean_text = re.sub(pattern, '', text).strip()
        # Extract the content between tags
        matches = re.findall(pattern, text)
        external_rag_query = matches[0] if matches else ''
        
        return clean_text, external_rag_query

    async def acall(self, state: RetrieverFunctionState) -> RetrieverFunctionState:
        """
        Call the GraphRetrieval class.

        Args:
            state: State of the function. keys: question, response_method, response_schema, response, error, source_docs

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

            # Extract external RAG query and clean question
            clean_question, external_rag_query = self._extract_external_rag_query(question)

            with Metrics("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=clean_question)
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
            )

            if not documents:
                if self.graph_retrieval.chat_history.messages:
                    self.graph_retrieval.chat_history.messages.pop()

            image_list_base64 = []
            if self.image:
                image_list_base64 = await self.extract_images(raw_docs)

            response = await self.graph_retrieval.get_response(
                clean_question,
                documents,
                image_list_base64,
                response_method=state.get("response_method"),
                response_schema=state.get("response_schema"),
            )

            # External RAG enrichment (only if enabled and user provided <e>...<e>)
            if external_rag_query and self.external_rag_enabled:
                external_context = await self._get_external_rag_context(external_rag_query)
                if external_context:
                    enrichment_prompt = ChatPromptTemplate.from_template(self.enrichment_prompt)
                    final_chain = enrichment_prompt | self.chat_llm | self.output_parser
                    unified_answer = await final_chain.ainvoke(
                        {
                            "original_response": response,
                            "external_context": external_context,
                        }
                    )
                    response = unified_answer

            # Log the final (possibly enriched) response
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
