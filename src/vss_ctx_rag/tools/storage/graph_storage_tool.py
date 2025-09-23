# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Optional

from langchain_community.graphs.graph_document import GraphDocument

from vss_ctx_rag.tools.storage.storage_tool import StorageTool


class GraphStorageTool(StorageTool):
    def __init__(self, name="storage_tool", config=None, tools=None) -> None:
        super().__init__(name, config, tools)

    @abstractmethod
    def add_graph_documents_to_db(self, graph_documents: List[GraphDocument]):
        """Persists the initial nodes and relationships from LLMGraphTransformer."""
        pass

    @abstractmethod
    def persist_chunk_data(
        self, batch_data: List[Dict], relationships: List[Dict], document_uuid: str
    ):
        """Persists chunk nodes and their structural relationships (PART_OF, FIRST_CHUNK, NEXT_CHUNK)."""
        pass

    @abstractmethod
    def persist_summary_chunk_relationships(self, document_uuid: str):
        """Persists IN_SUMMARY relationships between chunks and summaries.
        Called after _persist_chunk_data as chunks need to exist first.
        Implementation might query summaries and link based on chunkIdx.
        """
        pass

    @abstractmethod
    def persist_chunk_embeddings(self, data_for_embedding: List[Dict]):
        """Updates chunk nodes with their calculated embeddings."""
        pass

    @abstractmethod
    def persist_chunk_entity_relationships(
        self, batch_data: List[Dict], document_uuid: str
    ):
        """Persists HAS_ENTITY relationships between existing chunks and entities."""
        pass

    @abstractmethod
    def update_knn(self):
        """Updates the K-Nearest Neighbors graph based on chunk embeddings."""
        pass

    @abstractmethod
    def fetch_subtitle_for_embedding(self) -> List[Dict[str, Any]]:
        """Fetches subtitles that do not have an embedding property."""
        pass

    @abstractmethod
    def fetch_entities_needing_embedding(self) -> List[Dict[str, Any]]:
        """Fetches entities that do not have an embedding property."""
        # Should return list of dicts: [{'elementId': id, 'text': text_for_embedding}, ...]
        pass

    @abstractmethod
    def persist_entity_embeddings(self, rows_with_embeddings: List[Dict]):
        """Updates entity nodes with their calculated embeddings."""
        pass

    @abstractmethod
    def fetch_summaries_needing_embedding(self) -> List[Dict[str, Any]]:
        """Fetches summary nodes that do not have an embedding property."""
        # Should return list of dicts: [{'id': id, 'content': content_for_embedding}, ...]
        pass

    @abstractmethod
    def persist_summary_embeddings(self, summaries_with_embeddings: List[Dict]):
        """Updates summary nodes with their calculated embeddings."""
        pass

    @abstractmethod
    def create_document_node(self, document_uuid: str, camera_id: str):
        """Ensures the main Document node exists in the graph."""
        pass

    @abstractmethod
    async def finalize_graph_creation(self):
        """Perform any final database-specific actions (e.g., creating indexes)."""
        pass

    @abstractmethod
    def reset(self, state: dict = {}):
        """Reset the graph database."""
        pass

    @abstractmethod
    def retrieve_documents(
        self,
        question: str,
        uuid: str = "default",
        multi_channel: bool = False,
        top_k: int = None,
    ) -> Tuple[List[str], List[str]]:
        """Retrieves documents from the graph."""
        pass

    @abstractmethod
    def create_document_retriever(self) -> List[str]:
        """Creates a document retriever from the graph."""
        pass

    @abstractmethod
    def persist_subtitle_frames(self, subtitle_frames):
        """Persists subtitle frames in the graph."""
        pass

    @abstractmethod
    def persist_subtitle_embeddings(self, rows_with_embeddings: List[Dict]):
        """Updates subtitle nodes with their calculated embeddings."""
        pass

    @abstractmethod
    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_neighbors(self, node_id: int) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_next_chunks(self, chunk_id, number_of_hops: int = 1) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_chunk_asset_dir(self, chunk_id) -> Optional[str]:
        pass

    @abstractmethod
    def get_chunk_time_range(self, chunk_id) -> Optional[Tuple[float, float]]:
        pass

    @abstractmethod
    def get_asset_dirs_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[str]:
        pass

    @abstractmethod
    def filter_subtitles_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def merge_duplicate_nodes(self, duplicate_score_value: float):
        pass

    @abstractmethod
    def get_num_cameras(self) -> int:
        pass

    @abstractmethod
    def get_video_length(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_chunk_size(self) -> Dict[str, float]:
        pass

    @abstractmethod
    def get_chunk_camera_id(self, chunk_id):
        pass
