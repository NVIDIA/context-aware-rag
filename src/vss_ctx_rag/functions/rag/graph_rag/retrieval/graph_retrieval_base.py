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

from langchain_core.output_parsers import StrOutputParser

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_CHUNKS,
    DEFAULT_FRAMES_PER_CHUNK,
    MAX_FRAMES,
)


class GraphRetrievalBaseFunc(Function):
    """
    GraphRetrieval Function Base class.
    """

    config: dict
    output_parser = StrOutputParser()
    graph_db: GraphStorageTool
    metrics = GraphMetrics()

    def setup(self) -> None:
        """
        Setup the GraphRetrievalFunc class.

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
        self.graph_db = self.get_tool("db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.image_fetcher = self.get_tool("image_fetcher")
        self.top_k = self.get_param("top_k", default=DEFAULT_RAG_TOP_K)

        self.multi_channel = self.get_param(
            "multi_channel", default=DEFAULT_MULTI_CHANNEL
        )

        self.uuid = self.get_param("uuid", default="default")

        self.num_chunks = self.get_param("num_chunks", default=DEFAULT_CHUNKS)
        self.num_frames_per_chunk = self.get_param(
            "num_frames_per_chunk", default=DEFAULT_FRAMES_PER_CHUNK
        )

        self.max_total_images = self.get_param("max_total_images", default=MAX_FRAMES)

        self.multi_modal_models = ["gpt-4o", "gpt-4.1", "o3", "o4-mini", "o1"]
        logger.debug(f"Supported multi-model LLMs: {self.multi_modal_models}")

    async def extract_images(self, docs):
        """Extract images based on num_chunks and num_frames_per_chunk."""
        logger.info("Extracting images from documents")

        if self.chat_llm.model_name not in self.multi_modal_models:
            logger.info("Model is not supported for image extraction")
            return []

        image_list_base64 = []  # Initialize as empty list

        if not docs:
            logger.warning("No documents found.")
            return []

        ## asset_dirs are returned as a list of all relevant asset_dirs for the context
        ## We get the first document and use the asset_dirs from that since all documents will have the same asset_dirs
        context = docs[0]

        if type(context) is dict:
            asset_dirs = context.get("metadata", {}).get("asset_dirs", [])
        else:
            ## If type _DocumentWithState
            asset_dirs = context.metadata.get("asset_dirs", [])

        num_chunks_to_process = min(len(asset_dirs), self.num_chunks)
        asset_dirs_to_process = asset_dirs[:num_chunks_to_process]

        logger.debug(f"Got {len(docs)} documents to extract images from")

        # Log first document to
        logger.debug(
            f"Processing up to {num_chunks_to_process} chunks, selecting up to {self.num_frames_per_chunk} frames per chunk."
        )

        logger.debug(
            f"Processing up to {num_chunks_to_process} chunks, selecting up to {self.num_frames_per_chunk} frames per chunk."
        )

        for asset_dir in asset_dirs_to_process:
            if len(image_list_base64) >= self.max_total_images:
                logger.info(
                    f"Reached maximum total image limit ({self.max_total_images})."
                )
                break
            chunk_images_base64 = self.image_fetcher.get_image_base64(
                asset_dir, self.num_frames_per_chunk
            )

            if not chunk_images_base64:
                logger.debug(
                    f"No images found by ImageHelper for directory {asset_dir}"
                )
                continue

            image_list_base64.extend(chunk_images_base64)

        logger.info(f"Extracted a total of {len(image_list_base64)} images.")
        return image_list_base64
