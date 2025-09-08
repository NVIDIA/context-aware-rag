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


DEFAULT_GRAPH_ENRICHMENT_PROMPT = """You are providing a response from multiple sources.

GRAPH RAG CONTENT:
{original_response}

EXTERNAL RAG CONTENT:
{external_context}

Combine them into a single, coherent answer, preserving important details from both.
Do not include any introductory phrases, notes, explanations, or comments about how the inputs were combined. Do not reference the video summary or external context. Only provide the enriched summary itself."""


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

        # External RAG configuration
        self.external_rag_enabled = os.environ.get("EXTERNAL_RAG_ENABLED", "false").lower() == "true"
        self.external_rag_timeout = int(os.environ.get("EXTERNAL_RAG_TIMEOUT", "30"))
        
        logger.info(f"GraphRetrieval External RAG enabled: {self.external_rag_enabled}")
        logger.info(f"GraphRetrieval External RAG timeout: {self.external_rag_timeout}")

    async def _get_external_rag_context(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get context from external RAG service."""
        
        self.external_rag_collection = [item.strip() for item in os.environ.get("EXTERNAL_RAG_COLLECTION", "").split(',') if item.strip()]
        self.reranker_endpoint = os.environ.get("RERANKER_ENDPOINT")
        self.llm_endpoint = os.environ.get("LLM_ENDPOINT")
        self.embedding_endpoint = os.environ.get("EMBEDDING_ENDPOINT")
        
        if not self.external_rag_enabled:
            logger.info("External RAG is disabled, returning empty string")
            return ""
        if not self.external_rag_collection:
            logger.error("External RAG collections are required but not provided. Set EXTERNAL_RAG_COLLECTION env var.")
            return ""
            
        with TimeMeasure("external_rag/get_context", "blue"):
            try:
                headers = {"Content-Type": "application/json"}
                
                # Check if we have a custom external RAG server
                custom_server = os.getenv("EXTERNAL_RAG_CUSTOM_SERVER")
                
                if custom_server:
                    # Use the custom format
                    endpoint = f"{custom_server.rstrip('/')}/v1/generate"
                    payload = {
                        "messages": [{"role": "user", "content": query}],
                        "use_knowledge_base": True,
                        "temperature": 0.2,
                        "top_p": 0.7,
                        "max_tokens": 1024,
                        "reranker_top_k": 2,
                        "vdb_top_k": 10,
                        "vdb_endpoint": "http://milvus:19530",
                        "collection_names": self.external_rag_collection,
                        "enable_query_rewriting": True,
                        "enable_reranker": True,
                        "enable_citations": True,
                        "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
                        "reranker_model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
                        "embedding_model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
                        "stop": [],
                        "filter_expr": ''
                    }

                    if self.llm_endpoint:
                        payload["llm_endpoint"] = self.llm_endpoint
                    if self.embedding_endpoint:
                        payload["embedding_endpoint"] = self.embedding_endpoint
                    if self.reranker_endpoint:
                        payload["reranker_endpoint"] = self.reranker_endpoint
                else:
                    # If custom_server is not set, log an error and return empty string
                    logger.error("EXTERNAL_RAG_ENABLED is true, but EXTERNAL_RAG_CUSTOM_SERVER is not set.")
                    return ""
                
                logger.info(f"Sending request to external RAG service: {endpoint}")
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        headers=headers,
                        json=payload,
                        timeout=self.external_rag_timeout
                    ) as response:
                        if response.status != 200:
                            logger.warning(
                                f"External RAG service returned status {response.status}: "
                                f"{await response.text()}"
                            )
                            return ""
                        
                        # Check content type
                        content_type = response.headers.get('Content-Type', '')

                        if 'text/event-stream' in content_type:
                            # Process streaming response
                            full_text = ""
                            citations_list = [] 
                            
                            async for line in response.content:
                                line_decoded = line.decode('utf-8').strip()
                                if line_decoded.startswith('data: '):
                                    data = line_decoded[6:]
                                    if data == '[DONE]':
                                        break
                                    try:
                                        event_data = json.loads(data)
                                        if 'choices' in event_data and len(event_data['choices']) > 0:
                                            delta = event_data['choices'][0].get('delta', {})
                                            if 'content' in delta and delta['content'] is not None:
                                                full_text += delta['content']
                                        
                                        if 'citations' in event_data:
                                            results = event_data['citations'].get('results', [])
                                            for citation_item in results:
                                                if 'content' in citation_item:
                                                    citations_list.append(citation_item['content'])
                                    except json.JSONDecodeError:
                                        logger.warning(f"Failed to decode JSON from stream: {data}")
                                        pass
                            context = full_text
                            if citations_list:
                                unique_citations = list(set(citations_list))
                                context += "\n\nCitations:\n" + "\n".join([f"- {cite}" for cite in unique_citations])
                        else:
                            # Standard JSON response
                            json_response = await response.json()
                            context = json_response.get("answer", "") 
                            citations_data = json_response.get("citations", {})
                            if isinstance(citations_data, dict):
                                citation_results = citations_data.get('results', [])
                                if citation_results:
                                    context += "\n\nCitations:\n" + "\n".join([f"- {cite.get('content', '')}" for cite in citation_results if cite.get('content')])
                            elif isinstance(citations_data, list):
                                context += "\n\nCitations:\n" + "\n".join([f"- {cite}" for cite in citations_data])
                            
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
                    enrichment_prompt = ChatPromptTemplate.from_template(DEFAULT_GRAPH_ENRICHMENT_PROMPT)
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
