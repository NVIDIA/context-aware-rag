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

from typing import List

from langchain_nvidia_ai_endpoints import NVIDIARerank
from langchain.schema import Document
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    register_tool_config,
    register_tool,
    ToolBaseModel,
)


@register_tool_config("reranker")
class RerankerConfig(ToolBaseModel):
    model: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    base_url: str = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
    api_key: str = "NOAPIKEYSET"
    top_n: int = 5


@register_tool(config=RerankerConfig)
class NVIDIARerankerTool(Tool):
    """NVIDIA Reranker Tool that wraps NVIDIARerank for use as a proper Tool."""

    def __init__(
        self,
        name: str,
        tools=None,
        config=None,
    ):
        super().__init__(name, tools)

        self.config = config
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        self.reranker = NVIDIARerank(
            model=self.config.params.model,
            api_key=self.config.params.api_key,
            base_url=self.config.params.base_url,
            top_n=self.config.params.top_n,
        )

    def compress_documents(
        self, documents: List[Document], query: str
    ) -> List[Document]:
        """Rerank documents based on relevance to query.

        Args:
            documents: List of documents to rerank
            query: Query to rank documents against

        Returns:
            Reranked list of documents
        """
        return self.reranker.compress_documents(documents, query)

    async def acompress_documents(
        self, documents: List[Document], query: str
    ) -> List[Document]:
        """Async rerank documents based on relevance to query.

        Args:
            documents: List of documents to rerank
            query: Query to rank documents against

        Returns:
            Reranked list of documents
        """
        return await self.reranker.acompress_documents(documents, query)
