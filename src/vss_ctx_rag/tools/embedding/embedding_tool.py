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

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    register_tool_config,
    register_tool,
    ToolBaseModel,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger


@register_tool_config("embedding")
class EmbeddingConfig(ToolBaseModel):
    model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = "NOAPIKEYSET"
    truncate: str = "END"


@register_tool(config=EmbeddingConfig)
class NVIDIAEmbeddingTool(Tool):
    """NVIDIA Embedding Tool that wraps NVIDIAEmbeddings for use as a proper Tool."""

    def __init__(
        self,
        name: str,
        tools=None,
        config=None,
    ):
        super().__init__(name, config, tools)
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        self.embedding = NVIDIAEmbeddings(
            model=self.config.params.model,
            truncate=self.config.params.truncate,
            api_key=self.config.params.api_key,
            base_url=self.config.params.base_url,
        )

        logger.info(
            f"Initialized NVIDIAEmbeddingTool with model: {self.config.params.model}"
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return self.embedding.embed_query(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings
        """
        return await self.embedding.aembed_documents(texts)

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query.

        Args:
            text: Text to embed

        Returns:
            Embedding vector
        """
        return await self.embedding.aembed_query(text)
