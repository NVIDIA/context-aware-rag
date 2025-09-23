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

"""vector_retrieval_func.py: File contains Function class"""

import asyncio
import os
import traceback
from pathlib import Path
from re import compile
from typing import Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.documents import Document
from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.models.state_models import RetrieverFunctionState
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from typing import ClassVar, Dict, List
from vss_ctx_rag.utils.utils import format_docs
from vss_ctx_rag.utils.prompts import BASIC_QA_PROMPT
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


@register_function_config("vector_retrieval")
class VectorRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "milvus": ["db"],
        "elasticsearch": ["db"],
        "reranker": ["reranker"],
    }

    class VectorRetrievalParams(RetrieverConfig.RetrieverParams):
        custom_metadata: Optional[dict] = None
        is_user_specified_collection: Optional[bool] = False

    params: VectorRetrievalParams


@register_function(config=VectorRetrievalConfig)
class VectorRetrievalFunc(Function):
    """VectorRAG Function"""

    config: dict
    output_parser = StrOutputParser()
    db: StorageTool
    metrics = GraphMetrics()

    def setup(self):
        # Log the entire configuration passed to this function
        logger.info("VectorRetrievalFunc setup called")

        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.db = self.get_tool("db")
        self.reranker_tool = self.get_tool("reranker")
        self.top_k = self.get_param("top_k", default=DEFAULT_RAG_TOP_K)
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        # Simplified citation configuration
        logger.info("Attempting to load citations configuration...")
        self.citations_config = self.get_param("citations", default={})
        logger.info(f"Raw citations_config retrieved: {self.citations_config}")

        self.citations_enabled = self.citations_config.get("enabled", False)
        self.include_metadata = self.citations_config.get("include_metadata", True)
        self.citation_fields = self.citations_config.get(
            "citation_fields", ["file", "chunkIdx"]
        )
        self.citation_template = self.citations_config.get(
            "citation_template", "[{file}] {chunkIdx}\n"
        )
        self.show_snippets = self.citations_config.get("show_snippets", True)
        self.snippet_length = self.citations_config.get("snippet_length", 200)

        # Log citation configuration
        logger.info(f"Citations configuration loaded: {self.citations_config}")
        logger.info(f"Citations enabled: {self.citations_enabled}")
        logger.info(f"Citation template: {self.citation_template}")
        logger.info(f"Show snippets: {self.show_snippets}")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        embeddings_dimension = int(os.environ.get("CA_RAG_EMBEDDINGS_DIMENSION", 2048))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=embeddings_dimension,
            chunk_overlap=0,
            separators=["\n\n", "\n", "\n-"],
        )
        if self.reranker_tool:
            pipeline_compressor = DocumentCompressorPipeline(
                transformers=[splitter, self.reranker_tool.reranker]
            )
        else:
            pipeline_compressor = DocumentCompressorPipeline(transformers=[splitter])
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=self.db.as_retriever(
                search_kwargs={
                    "expr": "content_metadata['doc_type'] == 'caption'",
                    "k": self.top_k,
                }
            ),
        )
        self.retrieval_chain = {
            "source_docs": self.compression_retriever,
            "input": RunnablePassthrough(),
        } | RunnablePassthrough.assign(context=lambda x: format_docs(x["source_docs"]))

    def format_docs_for_return(self, docs: list[Document]):
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                {"metadata": doc.metadata, "page_content": doc.page_content}
            )
        return formatted_docs

    async def get_semantic_sim_response(
        self,
        question: str,
        response_method: str | None = None,
        response_schema: dict | None = None,
        **kwargs,
    ) -> tuple[str | dict, list[Document]]:
        # TODO: with pydantic state validation
        if response_method is not None and response_method not in [
            "json_mode",
            "text",
            "function_calling",
        ]:
            raise ValueError(
                f"Invalid response_method: {response_method}, has to be one of json_mode, text, or function_calling"
            )

        if response_method is not None and response_method != "text":
            if response_method == "json_mode" and "json" not in question.lower():
                raise ValueError("JSON mode requires 'json' in the question")

            logger.debug(
                f"Changing response format to:{response_method}, schema:{response_schema}"
            )

            llm_chain = self.chat_llm.with_structured_output(
                method=response_method, schema=response_schema
            )

        else:
            llm_chain = (
                self.chat_llm
                | self.output_parser
                | RunnableLambda(lambda x: self.regex_object.sub(r"\g<1>", x))
            )

        final_chain = self.retrieval_chain | RunnablePassthrough.assign(
            response=PromptTemplate.from_template(BASIC_QA_PROMPT) | llm_chain
        )
        result = await asyncio.get_running_loop().run_in_executor(
            None,
            final_chain.invoke,
            question,
        )

        response = result["response"]
        source_docs = result["source_docs"]
        return response, source_docs

    def format_citation(self, doc_metadata: Dict, citation_id: int) -> str:
        """Format a single citation based on the configuration template."""
        logger.debug(f"Formatting citation {citation_id} with metadata: {doc_metadata}")
        try:
            # Add citation ID to metadata for template formatting
            metadata_with_id = {**doc_metadata, "citation_id": citation_id}

            # Format using the template, with fallback values for missing fields
            formatted_citation = self.citation_template
            for field in self.citation_fields:
                placeholder = f"{{{field}}}"
                value = metadata_with_id.get(field, f"Unknown_{field}")
                formatted_citation = formatted_citation.replace(placeholder, str(value))

            logger.debug(f"Formatted citation: {formatted_citation}")
            return formatted_citation
        except Exception as e:
            logger.warning(f"Error formatting citation: {e}")
            return f"[{citation_id}] Document"

    def format_citations_display(self, citations: List[Dict], retrieved_docs) -> str:
        """Format citations for inline display."""
        logger.info(
            f"Formatting citations display. Citations count: {len(citations) if citations else 0}"
        )
        logger.info(f"Citations enabled: {self.citations_enabled}")

        if not citations or not self.citations_enabled:
            logger.info("No citations to display or citations disabled")
            return ""

        citation_text = "\n**Sources:**\n"
        logger.info(f"Processing {len(citations)} citations")

        for i, citation in enumerate(citations):
            formatted_citation = self.format_citation(citation, i + 1)

            if self.show_snippets and i < len(retrieved_docs):
                snippet = retrieved_docs[i].page_content[: self.snippet_length]
                if len(retrieved_docs[i].page_content) > self.snippet_length:
                    snippet += "..."
                citation_text += f'- **{formatted_citation}** - *"{snippet}"*\n'
                logger.debug(f"Added snippet for citation {i + 1}: {snippet[:50]}...")
            else:
                citation_text += f"- **{formatted_citation}**\n"
        citation_text += "\n - - - - - \n"

        logger.info(f"Final citation text length: {len(citation_text)}")
        logger.debug(f"Citation text preview: {citation_text[:200]}...")
        return citation_text

    def extract_citations_from_docs(self, retrieved_docs) -> List[Dict]:
        """Extract citation information from retrieved documents."""
        logger.info(
            f"Extracting citations from {len(retrieved_docs)} retrieved documents"
        )
        citations = []

        for i, doc in enumerate(retrieved_docs):
            logger.debug(
                f"Processing document {i}: has metadata: {hasattr(doc, 'metadata')}"
            )
            if hasattr(doc, "metadata"):
                logger.info(f"Document {i} metadata keys: {list(doc.metadata.keys())}")
                logger.info(f"Document {i} full metadata: {doc.metadata}")

            if hasattr(doc, "metadata") and self.include_metadata:
                citation_info = {}
                for field in self.citation_fields:
                    citation_info[field] = doc.metadata.get("content_metadata", {}).get(
                        field, f"Unknown_{field}"
                    )
                citations.append(citation_info)
                logger.debug(f"Added citation {i}: {citation_info}")
            else:
                # Basic citation info if metadata is not available
                basic_citation = {
                    "doc_id": len(citations) + 1,
                    "filename": "Document",
                    "timestamp": "Unknown",
                }
                citations.append(basic_citation)
                logger.debug(f"Added basic citation {i}: {basic_citation}")

        logger.info(f"Extracted {len(citations)} citations total")
        return citations

    async def acall(self, state: RetrieverFunctionState) -> RetrieverFunctionState:
        """QnA function call
        state keys: question, response_method, response_schema, response, error, source_docs
        """
        if self.log_dir:
            with Metrics("VectorRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("vector_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        with Metrics(
            "VectorRAG/retrieval", "red", span_kind=Metrics.SPAN_KIND["AGENT"]
        ) as tm:
            try:
                logger.debug("Running qna with question: %s", state["question"])
                tm.input(state)

                response, source_docs = await self.get_semantic_sim_response(
                    state["question"],
                    state.get("response_method"),
                    state.get("response_schema"),
                )
                logger.info(f"Source docs: {source_docs}")

                # Add citations if enabled
                logger.info(
                    f"Checking for citations. Enabled: {self.citations_enabled}"
                )
                if self.citations_enabled and source_docs:
                    retrieved_docs = source_docs
                    logger.info(f"Found {len(retrieved_docs)} source documents")

                    citations = self.extract_citations_from_docs(retrieved_docs)
                    citation_display = self.format_citations_display(
                        citations, retrieved_docs
                    )

                    if citation_display:
                        logger.info("Adding citations to response")
                        response = citation_display + response
                    else:
                        logger.warning("Citation display is empty")

                    # Store citations in state for potential UI use
                    state["citations"] = citations
                    state["retrieved_docs"] = len(retrieved_docs)
                    logger.info(f"Stored {len(citations)} citations in state")
                else:
                    if not self.citations_enabled:
                        logger.info("Citations are disabled")
                    if not source_docs:
                        logger.warning(
                            "No source_documents in semantic search response"
                        )

                logger.info(f"Final response length: {len(response)}")

                state["response"] = response
                state["source_docs"] = self.format_docs_for_return(source_docs)
                state["formatted_docs"] = [i.page_content for i in source_docs]

                logger.debug("Semantic search response: %s", response)
            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error("Error in VectorRetrievalFunc %s", str(e))
                state["error"] = str(e)
                tm.error(e)
            finally:
                tm.output({"state": state})
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        pass

    async def areset(self, state: dict):
        self.metrics.reset()
        await asyncio.sleep(0.01)
