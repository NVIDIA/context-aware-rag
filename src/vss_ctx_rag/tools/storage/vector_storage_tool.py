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

"""vector_storage_tool.py: Abstract base class for vector storage tools."""

from abc import abstractmethod
from typing import List, Dict, Any

from vss_ctx_rag.tools.storage.storage_tool import StorageTool


class VectorStorageTool(StorageTool):
    """
    Abstract base class for vector storage tools that provide common vector database operations.

    This class extends StorageTool with vector-specific operations that are common
    across different vector database implementations like Milvus and Elasticsearch.
    """

    @abstractmethod
    def search(self, search_query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """Perform similarity search in the vector storage system.

        Args:
            search_query: The query text to search for
            top_k: Number of top results to return

        Returns:
            List of metadata dictionaries for the most similar documents
        """
        pass

    @abstractmethod
    def drop_data(self, expr: str = None):
        """Delete data from the vector storage system based on an expression/query.

        Args:
            expr: Expression or query to filter which data to delete.
                 Implementation-specific format (e.g., Milvus expr, Elasticsearch query)
        """
        pass

    @abstractmethod
    def drop_collection(self):
        """Drop the entire collection/index and recreate it.

        This should reset the vector storage to an empty state.
        """
        pass

    @staticmethod
    @abstractmethod
    def _escape(val: str) -> str:
        """Escape special characters in a string for safe use in queries.

        Args:
            val: String value to escape

        Returns:
            Escaped string safe for use in database queries
        """
        pass
