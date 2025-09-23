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

"""storage_handler.py:"""

from abc import ABC, abstractmethod
from typing import Optional, ClassVar, Dict, List, Any

from pydantic import Field
from langchain_core.retrievers import RetrieverLike

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    ToolBaseModel,
)
import uuid


class DBConfig(ToolBaseModel):
    """Base configuration for database tools with embedded tool references."""

    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "embedding": [],
        "reranker": [],
    }

    host: Optional[str] = None
    port: Optional[str] = None
    username: Optional[str] = ""
    password: Optional[str] = ""
    collection_name: Optional[str] = Field(
        default_factory=lambda: "default_" + str(uuid.uuid4()).replace("-", "_")
    )


class StorageTool(Tool, ABC):
    """
    Abstract base class for all storage tools.
    """

    @abstractmethod
    def add_summary(self, summary, metadata):
        """Add a summary to the storage system.

        Args:
            summary: The summary text/content to store
            metadata: Dictionary containing metadata like batch_i, uuid, etc.
        """
        pass

    @abstractmethod
    def reset(self, state: dict = {}):
        """Reset the storage system, optionally for a specific UUID.

        Args:
            state: Dictionary containing the state of the storage system.
        """
        pass

    @abstractmethod
    async def aget_text_data(self, start_batch_index, end_batch_index, uuid):
        """Async method to retrieve text data for a range of batch indices.

        Args:
            start_batch_index: Starting batch index (inclusive)
            end_batch_index: Ending batch index (inclusive)
            uuid: UUID to filter results

        Returns:
            List of dictionaries containing text and batch_i for each batch
        """
        pass

    @abstractmethod
    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_start_time: Optional[float] = None,
        min_end_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        chunk_id: Optional[int] = None,
        uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def query(self, query, params: dict = {}):
        """Execute a query against the storage system.

        Args:
            query: Query string/command specific to the storage implementation
            params: Dictionary of parameters for the query

        Returns:
            Query results in a format specific to the storage implementation
        """
        pass

    @abstractmethod
    async def aget_max_batch_index(self, uuid: str) -> int:
        pass

    @abstractmethod
    def as_retriever(self, search_kwargs: dict = None) -> RetrieverLike:
        """
        This method is used to create a retriever for the storage tool.
        It is used to retrieve documents from the storage tool.
        """
        pass
