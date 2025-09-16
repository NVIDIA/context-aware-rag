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

"""external_rag_tool.py: File contains ExternalRAGTool class"""

import asyncio
import traceback
from typing import Any, ClassVar, Dict, List, Optional

from nvidia_rag.rag_server.main import NvidiaRAG
from pydantic import Field

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    ToolBaseModel,
    register_tool,
    register_tool_config,
)
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger


@register_tool_config("external_rag")
class ExternalRAGToolConfig(ToolBaseModel):
    """External RAG Tool configuration."""

    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "vector_db": ["db"],
        "reranker": ["reranker"],
    }
    collection: str = Field(default="")


@register_tool(config=ExternalRAGToolConfig)
class ExternalRAGTool(Tool):
    """Tool for interacting with an external RAG service."""

    def __init__(self, name="external_rag", config=None, tools=None) -> None:
        """Initialize the ExternalRAGTool."""
        super().__init__(name, config, tools)
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        """Update the tool with new configuration."""
        self.config = config
        self.vector_db = tools.get("vector_db")  # This is now the external_vector_db
        self.reranker_tool = tools.get("reranker")
        self.nvidia_rag = NvidiaRAG()
        self.collection = self.config.collection

    @staticmethod
    def _parse_search_results(search_results) -> str:
        """Parse search results from NvidiaRAG into a single string."""
        doc_list: List[str] = []
        if search_results and getattr(search_results, "results", None):
            for result in search_results.results:
                content = getattr(result, "content", "")
                doc_list.append(content)
        return "\n".join(doc_list)

    async def query(self, query: str, reranker_top_k: int, vdb_top_k: int) -> str:
        """Query the external RAG service."""
        if not self.collection:
            logger.error("External RAG collection is required but not provided.")
            return ""

        with Metrics(
            "external_rag/query", "blue", span_kind=Metrics.SPAN_KIND["CHAIN"]
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
                    "collection_names": [
                        item.strip() for item in self.collection.split(",") if item.strip()
                    ],
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
