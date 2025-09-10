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

"""external_rag_client.py: File contains ExternalRAGClient class"""

import asyncio
import traceback
from typing import Any, List

from nvidia_rag.rag_server.main import NvidiaRAG

from vss_ctx_rag.utils.ctx_rag_logger import logger, Metrics


class ExternalRAGClient:
    """Client for interacting with an external RAG service."""

    def __init__(
        self,
        nvidia_rag: NvidiaRAG,
        vector_db: Any,
        reranker_tool: Any,
        external_rag_collection: List[str],
    ):
        """Initialize the ExternalRAGClient."""
        self.nvidia_rag = nvidia_rag
        self.vector_db = vector_db
        self.reranker_tool = reranker_tool
        self.external_rag_collection = external_rag_collection

    @staticmethod
    def _parse_search_results(search_results) -> str:
        """Parse search results from NvidiaRAG into a single string."""
        doc_list: List[str] = []
        if search_results and getattr(search_results, "results", None):
            for result in search_results.results:
                content = getattr(result, "content", "")
                doc_list.append(content)
        return "\n".join(doc_list)

    async def get_context(
        self, query: str, reranker_top_k: int, vdb_top_k: int
    ) -> str:
        """Get context from external RAG service using the NvidiaRAG tool."""
        if not self.external_rag_collection:
            logger.error("External RAG collections are required but not provided.")
            return ""

        with Metrics(
            "external_rag/get_context", "blue", span_kind=Metrics.SPAN_KIND["CHAIN"]
        ) as tm:
            tm.input(
                {
                    "query": query,
                    "reranker_top_k": reranker_top_k,
                    "vdb_top_k": vdb_top_k,
                }
            )
            try:
                if (
                    self.vector_db.embedding.base_url
                    == "https://integrate.api.nvidia.com/v1"
                ):
                    embedding_endpoint = (
                        self.vector_db.embedding.base_url + "/embeddings"
                    )
                else:
                    embedding_endpoint = self.vector_db.embedding.base_url

                search_kwargs = {
                    "query": query,
                    "messages": [],
                    "reranker_top_k": reranker_top_k,
                    "vdb_top_k": vdb_top_k,
                    "collection_names": self.external_rag_collection,
                    "vdb_endpoint": self.vector_db.connection["uri"],
                    "enable_query_rewriting": True,
                    "embedding_model": self.vector_db.embedding.model,
                    "embedding_endpoint": embedding_endpoint,
                }

                if self.reranker_tool:
                    search_kwargs.update(
                        {
                            "enable_reranker": True,
                            "reranker_model": self.reranker_tool.reranker.model,
                            "reranker_endpoint": self.reranker_tool.reranker.base_url,
                        }
                    )
                else:
                    search_kwargs.update({"enable_reranker": False})

                search_results = await asyncio.get_running_loop().run_in_executor(
                    None,
                    lambda: self.nvidia_rag.search(**search_kwargs),
                )

                context = self._parse_search_results(search_results)
                logger.info(f"External RAG context: {context[:100]}...")
                tm.output({"context": context})
                return context
            except Exception as e:
                logger.error(f"Error fetching from external RAG service: {e}")
                logger.error(traceback.format_exc())
                tm.error(e)
                return ""
