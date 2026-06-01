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

from typing import List, Optional
import random

from langchain_core.embeddings import Embeddings
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    register_tool_config,
    register_tool,
    ToolBaseModel,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger


class NullEmbedding(Embeddings):
    """Null Embedding that generates dummy embeddings for testing/development.

    This implements the same interface as NVIDIAEmbeddings but generates dummy
    embeddings without making any API calls. Useful for testing, development, or
    when actual embeddings are not needed.
    """

    def __init__(
        self,
        dimensions: int = 1024,
        seed: int = 42,
        use_random: bool = False,
        model: str = "null-embedding",
        **kwargs,
    ):
        """Initialize NullEmbedding.

        Args:
            dimensions: Size of the embedding vector
            seed: Random seed for reproducibility
            use_random: If True, generate random embeddings each time.
                       If False, generate deterministic embeddings based on text hash.
            model: Model name (for compatibility)
            **kwargs: Additional arguments for compatibility
        """
        self.dimensions = dimensions
        self.seed = seed
        self.use_random = use_random
        self.model = model

        # Create local Random instance for thread safety
        self._random = random.Random(seed if not use_random else None)

        logger.info(
            f"Initialized NullEmbedding with dimensions: {self.dimensions}, "
            f"use_random: {self.use_random}, model: {self.model}"
        )

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate a dummy embedding vector.

        Args:
            text: Input text (used for seeding if use_random is False)

        Returns:
            Dummy embedding vector
        """
        if self.use_random:
            # Generate random embeddings
            return [self._random.random() for _ in range(self.dimensions)]
        else:
            # Generate deterministic embeddings based on text hash
            text_hash = hash(text)
            local_rng = random.Random(self.seed + text_hash)
            return [local_rng.random() for _ in range(self.dimensions)]

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query with dummy embeddings.

        Args:
            text: Text to embed

        Returns:
            Dummy embedding vector
        """
        logger.debug(f"Generating null embedding for query: {text[:50]}...")
        return self._generate_embedding(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents with dummy embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of dummy embeddings
        """
        logger.debug(f"Generating null embeddings for {len(texts)} documents")
        return [self._generate_embedding(text) for text in texts]

    async def aembed_query(self, text: str) -> List[float]:
        """Async embed a single query with dummy embeddings.

        Args:
            text: Text to embed

        Returns:
            Dummy embedding vector
        """
        logger.debug(f"Generating null embedding for query (async): {text[:50]}...")
        return self._generate_embedding(text)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async embed a list of documents with dummy embeddings.

        Args:
            texts: List of texts to embed

        Returns:
            List of dummy embeddings
        """
        logger.debug(f"Generating null embeddings for {len(texts)} documents (async)")
        return [self._generate_embedding(text) for text in texts]


@register_tool_config("embedding")
class EmbeddingConfig(ToolBaseModel):
    model: str = "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: str = "https://integrate.api.nvidia.com/v1"
    api_key: str = "NOAPIKEYSET"
    truncate: str = "END"
    enable: bool = True
    dimensions: Optional[int] = None


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

        # Use NullEmbedding if enable parameter is False, otherwise use NVIDIAEmbeddings
        if not self.config.params.enable:
            if self.config.params.dimensions is None:
                self.config.params.dimensions = 1024
            self.embedding = NullEmbedding(dimensions=self.config.params.dimensions)
            logger.info("Initialized NVIDIAEmbeddingTool with NullEmbedding")
        else:
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
