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

"""foundation_retrieval_func.py: File contains Function class"""

import asyncio
from nvidia_rag.rag_server.main import NvidiaRAG

import os
import traceback
from pathlib import Path
from re import compile
from typing import Optional
from langchain_core.output_parsers import StrOutputParser
from typing import List
from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.state_models import RetrieverFunctionState
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.utils.prompts import (
    CHAT_SYSTEM_TEMPLATE_PREFIX,
    CHAT_SYSTEM_TEMPLATE_SUFFIX,
)
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from typing import ClassVar, Dict
from langchain_core.documents import Document


@register_function_config("foundation_retrieval")
class FoundationRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "milvus": ["db"],
        "reranker": ["reranker"],
    }

    params: RetrieverConfig.RetrieverParams


@register_function(config=FoundationRetrievalConfig)
class FoundationRetrievalFunc(Function):
    """FoundationRAG Function"""

    config: dict
    output_parser = StrOutputParser()
    db: StorageTool
    metrics = GraphMetrics()

    def setup(self):
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.db = self.get_tool("db")
        self.reranker_tool = self.get_tool("reranker")
        self.top_k = self.get_param("top_k", default=DEFAULT_RAG_TOP_K)
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.nvidia_rag = NvidiaRAG()

    def format_docs_for_return(self, docs: list[Document]):
        formatted_docs = []

        for doc in docs:
            formatted_docs.append(
                {
                    "metadata": {
                        "asset_dirs": doc.metadata.content_metadata.get(
                            "asset_dirs", None
                        ),
                        "length": doc.metadata.content_metadata.get("length", None),
                    },
                    "page_content": doc.content,
                }
            )
        return formatted_docs

    def parse_search_results(self, search_results) -> List[str]:
        doc_list: List[str] = []
        for result in search_results.results:
            content = getattr(result, "content", "")
            doc_list.append(content)
        return doc_list

    async def get_response(
        self,
        question: str,
        doc_list: List[str],
        response_method: str = None,
        response_schema: dict = None,
        **kwargs,
    ) -> str | dict:
        messages = [
            {
                "role": "system",
                "content": CHAT_SYSTEM_TEMPLATE_PREFIX + CHAT_SYSTEM_TEMPLATE_SUFFIX,
            },
            {
                "role": "user",
                "content": f"Question: {question}\nVideo Summary: {doc_list}",
            },
        ]
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
            llm = self.chat_llm.with_structured_output(
                method=response_method,
                schema=response_schema,
            )
            response = await llm.ainvoke(messages)
            return response

        response = await self.chat_llm.ainvoke(messages)
        return response.content

    async def acall(self, state: RetrieverFunctionState) -> RetrieverFunctionState:
        """QnA function call
        state keys: question, response, error, response_method, response_schema
        """
        if self.log_dir:
            with Metrics("FoundationRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("foundation_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        try:
            logger.debug("Running qna with question: %s", state["question"])
            logger.debug(
                f"Vector DB COLLECTION NAME: {self.db.current_collection_name}"
            )

            with Metrics(
                "FoundationRAG/retrieval", "red", span_kind=Metrics.SPAN_KIND["AGENT"]
            ):
                if self.db.embedding.base_url == "https://integrate.api.nvidia.com/v1":
                    embedding_endpoint = self.db.embedding.base_url + "/embeddings"
                else:
                    embedding_endpoint = self.db.embedding.base_url
                if self.reranker_tool:
                    search_results = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self.nvidia_rag.search(
                            query=state["question"],
                            messages=[],
                            reranker_top_k=self.top_k,
                            vdb_top_k=self.top_k + 1,
                            collection_names=[self.db.current_collection_name],
                            vdb_endpoint=self.db.connection["uri"],
                            enable_query_rewriting=True,
                            enable_reranker=True,
                            embedding_model=self.db.embedding.model,
                            embedding_endpoint=embedding_endpoint,
                            reranker_model=self.reranker_tool.reranker.model,
                            reranker_endpoint=self.reranker_tool.reranker.base_url,
                            filter_expr='content_metadata["doc_type"] == "caption"',
                        ),
                    )
                else:
                    search_results = await asyncio.get_running_loop().run_in_executor(
                        None,
                        lambda: self.nvidia_rag.search(
                            query=state["question"],
                            messages=[],
                            reranker_top_k=self.top_k,
                            vdb_top_k=self.top_k + 1,
                            collection_names=[self.db.current_collection_name],
                            vdb_endpoint=self.db.connection["uri"],
                            enable_query_rewriting=True,
                            enable_reranker=False,
                            embedding_model=self.db.embedding.model,
                            embedding_endpoint=embedding_endpoint,
                            filter_expr='content_metadata["doc_type"] == "caption"',
                        ),
                    )

                logger.debug(f"Search results: {search_results}")

                doc_list = self.parse_search_results(search_results)

                response = await self.get_response(
                    question=state["question"],
                    doc_list=doc_list,
                    response_method=state.get("response_method"),
                    response_schema=state.get("response_schema"),
                )

                logger.debug(f"FRAG Retrieval Response: {response}")
                state["response"] = response
                state["source_docs"] = self.format_docs_for_return(
                    search_results.results
                )
                state["formatted_docs"] = doc_list

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in FoundationRetrievalFunc %s", str(e))
            state["response"] = "Sorry, something went wrong. Please try again."
            state["error"] = str(e)

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        pass

    async def areset(self, state: dict):
        self.metrics.reset()
        await asyncio.sleep(0.01)
