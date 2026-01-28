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

import asyncio
import os
from typing import ClassVar, Optional, Dict, List
try:
    from typing import override
except ImportError:
    # Python < 3.12
    def override(func):
        return func

from langchain_core.retrievers import RetrieverLike
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.tools.storage.storage_tool import DBConfig
from vss_ctx_rag.tools.storage.vector_storage_tool import VectorStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger


@register_tool_config("qdrant")
class QdrantDBConfig(DBConfig):
    """Configuration for Qdrant vector database.

    Attributes:
        host: Qdrant server host (default: localhost)
        port: Qdrant server port (default: 6333)
        api_key: Optional API key for Qdrant Cloud
        collection_name: Name of the Qdrant collection
        prefer_grpc: Use gRPC instead of HTTP (default: False)
        custom_metadata: Custom metadata to add to all documents
        user_specified_collection_name: User-specified collection name for dynamic switching
    """
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "embedding": ["embedding"],
    }

    api_key: Optional[str] = None
    prefer_grpc: Optional[bool] = False
    custom_metadata: Optional[dict] = {}
    user_specified_collection_name: Optional[str] = None


@register_tool(config=QdrantDBConfig)
class QdrantDBTool(VectorStorageTool):
    """Handler for Qdrant vector database which stores embeddings for RAG retrieval.

    This implementation provides vector storage and similarity search capabilities
    using Qdrant, following the same patterns as MilvusDBTool for consistency.

    Implements VectorStorageTool class.
    """

    def __init__(
        self,
        name="vector_db",
        tools=None,
        config=None,
    ) -> None:
        super().__init__(name, config, tools)

        # Initialize Qdrant client - support both URL and host/port connection methods
        client_kwargs = {
            "api_key": self.config.params.api_key,
            "prefer_grpc": self.config.params.prefer_grpc,
        }

        # Check if host looks like a URL (contains :// or starts with https:)
        host = self.config.params.host
        if host and ("://" in host or host.startswith("https:")):
            # Use URL-based connection (for Qdrant Cloud)
            client_kwargs["url"] = host
            logger.info(f"Connecting to Qdrant using URL: {host}")
        else:
            # Use host/port connection (for local Qdrant)
            client_kwargs["host"] = host
            client_kwargs["port"] = int(self.config.params.port) if self.config.params.port else 6333
            logger.info(f"Connecting to Qdrant at {host}:{client_kwargs['port']}")

        self.client = QdrantClient(**client_kwargs)

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "\n-"],
        )

        self.embedding = self.get_tool("embedding").embedding

        self.collection_name = self.config.params.collection_name.replace("-", "_")
        self.current_collection_name = self.config.params.collection_name.replace(
            "-", "_"
        )
        self.is_user_specified_collection_name = False
        self.custom_metadata = {}

        self._drop_old_default = os.getenv(
            "VIA_CTX_RAG_ENABLE_RET", "True"
        ).lower() not in [
            "true",
            "1",
        ]

        # Initialize the vector store
        self._vector_store = self._create_vector_store(
            self.collection_name, self._drop_old_default
        )
        self._current_vector_store = self._vector_store

        self.update_tool(self.config, tools)

    def _create_vector_store(
        self, collection_name: str, recreate_collection: bool = False
    ) -> QdrantVectorStore:
        """Create a Qdrant vector store instance.

        Args:
            collection_name: Name of the collection to create/use
            recreate_collection: If True, drop and recreate the collection

        Returns:
            QdrantVectorStore instance
        """
        if recreate_collection:
            try:
                self.client.delete_collection(collection_name=collection_name)
                logger.info(f"Dropped existing Qdrant collection: {collection_name}")
            except Exception as e:
                logger.debug(f"Collection {collection_name} does not exist or could not be deleted: {e}")

        # Check if collection exists
        collection_exists = False
        try:
            self.client.get_collection(collection_name=collection_name)
            collection_exists = True
            logger.debug(f"Collection {collection_name} already exists")
        except Exception:
            logger.info(f"Collection {collection_name} does not exist, creating it")
            # Create collection manually with proper vector configuration
            # Get embedding dimension from a sample embedding
            try:
                sample_embedding = self.embedding.embed_query("test")
                vector_size = len(sample_embedding)

                self.client.create_collection(
                    collection_name=collection_name,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(f"Created collection {collection_name} with vector size {vector_size}")
            except Exception as create_error:
                logger.warning(f"Failed to create collection: {create_error}")
                # Let QdrantVectorStore handle it

        return QdrantVectorStore(
            client=self.client,
            collection_name=collection_name,
            embedding=self.embedding,
        )

    def get_current_vector_store(self) -> QdrantVectorStore:
        """Get the currently active vector store."""
        return self._current_vector_store

    @override
    def update_tool(
        self,
        config: QdrantDBConfig,
        tools: Optional[Dict[str, Dict[str, Tool]]] = None,
    ):
        """
        Updates the QdrantDBTool configuration from a Pydantic config.

        Args:
            config: Configuration containing database settings
            tools: Optional dictionary of available tools

        Raises:
            ValueError: If required configuration is not set
        """
        try:
            if not config.params.host:
                raise ValueError("Qdrant host not set in database configuration.")
            if not config.params.port:
                raise ValueError("Qdrant port not set in database configuration.")

            user_specified_collection_name = (
                config.params.user_specified_collection_name
            )
            if user_specified_collection_name is None:
                self.is_user_specified_collection_name = False
                self.current_collection_name = self.collection_name
                self._current_vector_store = self._vector_store
                return

            custom_metadata = config.params.custom_metadata
            self.is_user_specified_collection_name = True
            if (
                user_specified_collection_name == self.current_collection_name
                and custom_metadata == self.custom_metadata
            ):
                return  # No changes needed

            self._current_vector_store = self._create_vector_store(
                user_specified_collection_name, recreate_collection=False
            )
            self.current_collection_name = user_specified_collection_name
            self.custom_metadata = custom_metadata

        except Exception as e:
            logger.error(f"Error updating Qdrant configuration: {e}")
            raise e

    def add_summary(self, summary: str, metadata: dict):
        """Add a single summary document to Qdrant.

        Args:
            summary: The text content to store
            metadata: Dictionary containing metadata for the document

        Returns:
            List of IDs for the added documents
        """
        with Metrics(
            "qdrant/add caption", "blue", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"summary": summary, "metadata": metadata})

            source = metadata.get("source", None) or metadata.get("file", None) or ""

            processed_metadata = metadata.copy()
            for key, value in processed_metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = str(value)
            processed_metadata.update(self.custom_metadata)

            metadata = {
                "source": source,
                "content_metadata": processed_metadata,
            }
            doc = Document(page_content=summary, metadata=metadata)
            logger.debug(
                f"Adding document to Qdrant collection '{self.current_collection_name}': {doc}"
            )

            try:
                return self.get_current_vector_store().add_documents([doc])
            except Exception as e:
                tm.error(e)
                logger.error(
                    f"Error adding documents to Qdrant: {e}, metadata: {metadata}"
                )
                raise e

    async def aadd_summary(self, summary: str, metadata: dict):
        """Async method to add a summary document to Qdrant.

        Args:
            summary: The text content to store
            metadata: Dictionary containing metadata for the document

        Returns:
            List of IDs for the added documents
        """
        with Metrics(
            "qdrant/add caption", "blue", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"summary": summary, "metadata": metadata})
            doc = Document(page_content=summary, metadata=metadata)
            return await self.get_current_vector_store().aadd_documents([doc])

    def add_summaries(self, batch_summary: list[str], batch_metadata: list[dict]):
        """Add multiple summary documents to Qdrant in batch.

        Args:
            batch_summary: List of text contents to store
            batch_metadata: List of metadata dictionaries (must match batch_summary length)

        Raises:
            ValueError: If batch_summary and batch_metadata lengths don't match
        """
        with Metrics(
            "Qdrant/AddSummaries", "yellow", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"batch_summary": batch_summary, "batch_metadata": batch_metadata})
            if len(batch_summary) != len(batch_metadata):
                raise ValueError(
                    "Incorrect param. The length of batch_summary batch and "
                    "metadata batch should match."
                )
            docs = []
            for i in range(len(batch_summary)):
                docs.append(
                    Document(page_content=batch_summary[i], metadata=batch_metadata[i])
                )
            document_chunks = self.text_splitter.split_documents(docs)
            self.get_current_vector_store().add_documents(document_chunks)

    @staticmethod
    def _escape(val: str) -> str:
        """Escape special characters in a string for safe use in Qdrant filters.

        Args:
            val: String value to escape

        Returns:
            Escaped string safe for use in Qdrant filters
        """
        # Qdrant uses JSON-like filters, so we escape backslashes and quotes
        return val.replace("\\", "\\\\").replace('"', '\\"')

    async def aget_text_data(self, start_batch_index=0, end_batch_index=-1, uuid=""):
        """Async method to retrieve text data for a range of batch indices.

        Args:
            start_batch_index: Starting batch index (inclusive)
            end_batch_index: Ending batch index (inclusive, -1 for no upper limit)
            uuid: UUID to filter results

        Returns:
            List of dictionaries containing document data and metadata
        """
        safe_uuid = self._escape(uuid)

        try:
            await asyncio.sleep(0.001)

            # Build Qdrant filter conditions
            must_conditions = [
                {
                    "key": "content_metadata.doc_type",
                    "match": {"value": "caption_summary"}
                },
                {
                    "key": "content_metadata.batch_i",
                    "range": {"gte": start_batch_index}
                }
            ]

            if end_batch_index != -1:
                must_conditions.append({
                    "key": "content_metadata.batch_i",
                    "range": {"lte": end_batch_index}
                })

            if safe_uuid:
                must_conditions.append({
                    "key": "content_metadata.uuid",
                    "match": {"value": uuid}
                })

            filter_dict = {"must": must_conditions}

            logger.debug(
                f"Getting text data from Qdrant collection: {self.current_collection_name}"
            )
            logger.debug(f"Filter: {filter_dict}")

            # Scroll through all matching points
            results = []
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.current_collection_name,
                    scroll_filter=filter_dict,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as e:
                logger.warning(f"Failed to filter by metadata (may need payload index): {e}")
                # Try without filter to get all points
                try:
                    scroll_result = self.client.scroll(
                        collection_name=self.current_collection_name,
                        limit=100,
                        with_payload=True,
                        with_vectors=False,
                    )
                except Exception as scroll_error:
                    logger.error(f"Failed to scroll collection: {scroll_error}")
                    return []

            points, next_page_offset = scroll_result

            while points:
                for point in points:
                    payload = point.payload
                    # Flatten content_metadata with the rest of the metadata
                    result = {
                        **{
                            k: v
                            for k, v in payload.items()
                            if k != "content_metadata"
                        },
                        **(
                            payload.get("content_metadata", {})
                            if isinstance(payload.get("content_metadata"), dict)
                            else {}
                        ),
                    }
                    results.append(result)

                if next_page_offset is None:
                    break

                scroll_result = self.client.scroll(
                    collection_name=self.current_collection_name,
                    scroll_filter=filter_dict,
                    limit=100,
                    offset=next_page_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                points, next_page_offset = scroll_result

            return results

        except Exception as e:
            logger.warning(f"Error getting text data from Qdrant: {e}")
            return []

    async def aget_max_batch_index(self, uuid: str = ""):
        """Get the maximum batch index for a given UUID.

        Args:
            uuid: UUID to filter results (empty string for all documents)

        Returns:
            Maximum batch_i value found
        """
        must_conditions = [
            {
                "key": "content_metadata.doc_type",
                "match": {"value": "caption_summary"}
            }
        ]

        if uuid:
            safe_uuid = self._escape(uuid)
            must_conditions.append({
                "key": "content_metadata.uuid",
                "match": {"value": uuid}
            })

        filter_dict = {"must": must_conditions}

        # Scroll through all matching points to find max batch_i
        max_batch_index = -1
        try:
            scroll_result = self.client.scroll(
                collection_name=self.current_collection_name,
                scroll_filter=filter_dict,
                limit=100,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as e:
            logger.warning(f"Failed to filter by metadata (may need payload index): {e}")
            # Try without filter to get all points
            try:
                scroll_result = self.client.scroll(
                    collection_name=self.current_collection_name,
                    limit=100,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as scroll_error:
                logger.error(f"Failed to scroll collection: {scroll_error}")
                return 0

        points, next_page_offset = scroll_result

        while points:
            for point in points:
                content_metadata = point.payload.get("content_metadata", {})
                if isinstance(content_metadata, dict):
                    batch_i = content_metadata.get("batch_i", -1)
                    max_batch_index = max(max_batch_index, batch_i)

            if next_page_offset is None:
                break

            scroll_result = self.client.scroll(
                collection_name=self.current_collection_name,
                scroll_filter=filter_dict,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_page_offset = scroll_result

        return max_batch_index

    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_start_time: Optional[float] = None,
        min_end_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        chunk_id: Optional[int] = None,
        uuid: Optional[str] = None,
    ):
        """Filter chunks based on time ranges and metadata.

        Args:
            min_start_time: Minimum start time (NTP float)
            max_start_time: Maximum start time (NTP float)
            min_end_time: Minimum end time (NTP float)
            max_end_time: Maximum end time (NTP float)
            camera_id: Camera ID to filter by
            chunk_id: Chunk ID to filter by
            uuid: UUID to filter by

        Returns:
            List of matching documents with flattened metadata
        """
        must_conditions = [
            {
                "key": "content_metadata.doc_type",
                "match": {"value": "caption"}
            }
        ]

        if min_start_time is not None:
            must_conditions.append({
                "key": "content_metadata.start_ntp_float",
                "range": {"gte": min_start_time}
            })
        if max_start_time is not None:
            must_conditions.append({
                "key": "content_metadata.start_ntp_float",
                "range": {"lte": max_start_time}
            })
        if min_end_time is not None:
            must_conditions.append({
                "key": "content_metadata.end_ntp_float",
                "range": {"gte": min_end_time}
            })
        if max_end_time is not None:
            must_conditions.append({
                "key": "content_metadata.end_ntp_float",
                "range": {"lte": max_end_time}
            })
        if camera_id is not None:
            must_conditions.append({
                "key": "content_metadata.camera_id",
                "match": {"value": camera_id}
            })
        if chunk_id is not None:
            must_conditions.append({
                "key": "content_metadata.chunk_id",
                "match": {"value": chunk_id}
            })
        if uuid is not None:
            must_conditions.append({
                "key": "content_metadata.uuid",
                "match": {"value": uuid}
            })

        filter_dict = {"must": must_conditions}

        results = []
        scroll_result = self.client.scroll(
            collection_name=self.current_collection_name,
            scroll_filter=filter_dict,
            limit=100,
            with_payload=True,
            with_vectors=False,
        )

        points, next_page_offset = scroll_result

        while points:
            for point in points:
                payload = point.payload
                result = {
                    **{
                        k: v
                        for k, v in payload.items()
                        if k != "content_metadata"
                    },
                    **(
                        payload.get("content_metadata", {})
                        if isinstance(payload.get("content_metadata"), dict)
                        else {}
                    ),
                }
                results.append(result)

            if next_page_offset is None:
                break

            scroll_result = self.client.scroll(
                collection_name=self.current_collection_name,
                scroll_filter=filter_dict,
                limit=100,
                offset=next_page_offset,
                with_payload=True,
                with_vectors=False,
            )
            points, next_page_offset = scroll_result

        return results

    def search(self, search_query, top_k=1):
        """Perform similarity search in Qdrant.

        Args:
            search_query: Query text to search for
            top_k: Number of top results to return

        Returns:
            List of metadata dictionaries for the most similar documents
        """
        search_results = self.get_current_vector_store().similarity_search(
            search_query, k=top_k
        )
        return [result.metadata for result in search_results]

    def query(self, query, params: dict = {}):
        """Execute a raw query against Qdrant.

        Args:
            query: Query filter in Qdrant format
            params: Additional parameters for the query

        Returns:
            List of matching points with their payloads
        """
        try:
            scroll_result = self.client.scroll(
                collection_name=self.current_collection_name,
                scroll_filter=query if isinstance(query, dict) else None,
                limit=params.get("limit", 100),
                with_payload=True,
                with_vectors=False,
            )
            points, _ = scroll_result
            return [point.payload for point in points]
        except Exception as e:
            logger.warning(f"Error querying Qdrant collection: {e}")
            return []

    def drop_data(self, expr: str = None):
        """Delete data from Qdrant based on a filter expression.

        Args:
            expr: Filter expression (not used in current implementation - deletes all)
        """
        try:
            # Delete all points in the collection
            self.client.delete(
                collection_name=self.current_collection_name,
                points_selector={"filter": {}}
            )
            logger.info(f"Dropped data from Qdrant collection: {self.current_collection_name}")
        except Exception as e:
            logger.warning(f"Error dropping data from Qdrant: {e}")

    def drop_collection(self):
        """Drop the entire Qdrant collection and recreate it."""
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            logger.info(f"Dropped Qdrant collection: {self.collection_name}")
        except Exception as e:
            logger.warning(f"Error dropping Qdrant collection: {e}")

        # Recreate the default vector store
        self._vector_store = self._create_vector_store(
            self.collection_name, recreate_collection=False
        )
        self._current_vector_store = self._vector_store
        self.current_collection_name = self.collection_name
        self.is_user_specified_collection_name = False

    def reset(self, state: Optional[dict] = None):
        """Reset the storage system, optionally for a specific UUID.

        Args:
            state: Dictionary containing reset parameters (uuid, erase_db, etc.)
        """
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        if state is None:
            state = {}
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return

        delete_external_collection = state.get("delete_external_collection", False)

        # Build filter for deletion
        if uuid and not erase_db:
            filter_dict = {
                "must": [
                    {
                        "key": "content_metadata.uuid",
                        "match": {"value": self._escape(uuid)}
                    }
                ]
            }
        else:
            filter_dict = {}  # Delete all

        if self.is_user_specified_collection_name:
            if delete_external_collection:
                try:
                    self.client.delete(
                        collection_name=self.current_collection_name,
                        points_selector={"filter": filter_dict} if filter_dict else {}
                    )
                except Exception as e:
                    logger.warning(f"Error deleting from user-specified collection: {e}")
        else:
            try:
                self.client.delete(
                    collection_name=self.current_collection_name,
                    points_selector={"filter": filter_dict} if filter_dict else {}
                )
            except Exception as e:
                logger.warning(f"Error deleting from collection: {e}")

        self.is_user_specified_collection_name = False
        # Always revert to the default collection
        self._current_vector_store = self._vector_store
        self.current_collection_name = self.collection_name

    def as_retriever(self, search_kwargs: dict = None) -> RetrieverLike:
        """Create a LangChain retriever for the Qdrant database.

        Args:
            search_kwargs: Optional search parameters (k, filter, etc.)

        Returns:
            A LangChain RetrieverLike object for document retrieval
        """
        if search_kwargs is None:
            search_kwargs = {}
        return self.get_current_vector_store().as_retriever(search_kwargs=search_kwargs)
