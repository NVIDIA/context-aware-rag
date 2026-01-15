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
from typing import override, ClassVar

from langchain_core.retrievers import RetrieverLike
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_milvus import Milvus
from pymilvus import MilvusException, connections
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.tools.storage.storage_tool import DBConfig
from vss_ctx_rag.tools.storage.vector_storage_tool import VectorStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from typing import Optional, Dict, List
from pymilvus import Collection


@register_tool_config("milvus")
class MilvusDBConfig(DBConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "embedding": ["embedding"],
    }

    custom_metadata: Optional[dict] = {}
    user_specified_collection_name: Optional[str] = None


@register_tool(config=MilvusDBConfig)
class MilvusDBTool(VectorStorageTool):
    """Handler for Milvus DB which stores the video embeddings mapped using
    the summary text embeddings which can be used for retrieval.

    Implements StorageHandler class
    """

    def __init__(
        self,
        name="vector_db",
        tools=None,
        config=None,
    ) -> None:
        super().__init__(name, config, tools)

        self.connection = {
            "uri": f"http://{self.config.params.host}:{self.config.params.port}"
        }

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

        self._collection = Milvus(
            embedding_function=self.embedding,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=self._drop_old_default,
        )
        self._current_collection = self._collection

        self._pymilvus_collection = None
        self._pymilvus_current_collection = None

        self.update_tool(self.config, tools)

    def get_current_collection(self) -> Milvus:
        """Get the currently active collection."""
        return self._current_collection

    def get_current_pymilvus_collection(self) -> Collection:
        """Get the currently active pymilvus collection."""
        if self._pymilvus_current_collection is None:
            if (
                self._pymilvus_collection is None
                or self._pymilvus_collection.name != self.current_collection_name
            ):
                try:
                    connections._fetch_handler("default")
                except Exception:
                    connections.connect(
                        alias="default",
                        host=self.config.params.host,
                        port=self.config.params.port,
                    )

                new_collection = Collection(self.current_collection_name)
                new_collection.load()
                self._pymilvus_current_collection = new_collection

                if self.current_collection_name == self.collection_name:
                    self._pymilvus_collection = new_collection
            else:
                self._pymilvus_current_collection = self._pymilvus_collection
        return self._pymilvus_current_collection

    @override
    def update_tool(
        self,
        config: MilvusDBConfig,
        tools: Optional[Dict[str, Dict[str, Tool]]] = None,
    ):
        """
        Updates the MilvusDBTool configuration from a Pydantic config.

        Args:
            config: Configuration containing database settings
        Raises:
            ValueError: If required configuration is not set
        """

        try:
            if not config.params.host:
                raise ValueError("Milvus host not set in database configuration.")
            if not config.params.port:
                raise ValueError("Milvus port not set in database configuration.")

            user_specified_collection_name = (
                config.params.user_specified_collection_name
            )
            if user_specified_collection_name is None:
                self.is_user_specified_collection_name = False
                self.current_collection_name = self.collection_name
                self._current_collection = self._collection
                self._pymilvus_current_collection = self._pymilvus_collection
                return
            custom_metadata = config.params.custom_metadata
            self.is_user_specified_collection_name = True
            if (
                user_specified_collection_name == self.current_collection_name
                and custom_metadata == self.custom_metadata
            ):
                return  # No changes needed

            self._current_collection = Milvus(
                embedding_function=self.embedding,
                connection_args=self.connection,
                collection_name=user_specified_collection_name,
                auto_id=True,
                drop_old=False,
            )
            self.current_collection_name = user_specified_collection_name
            self.custom_metadata = custom_metadata
            self._pymilvus_current_collection = None

        except Exception as e:
            logger.error(f"Error updating Milvus configuration: {e}")
            raise e

    def add_summary(self, summary: str, metadata: dict):
        with Metrics(
            "milvusdb/add caption", "blue", span_kind=Metrics.SPAN_KIND["TOOL"]
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
                f"Adding document to MILVUS collection '{self.current_collection_name}': {doc}"
            )
        try:
            return self.get_current_collection().add_documents([doc])
        except MilvusException as e:
            tm.error(e)
            logger.error(
                f"Invalid metadata while adding documents to Milvus: {metadata}"
            )
            raise e

    async def aadd_summary(self, summary: str, metadata: dict):
        with Metrics(
            "milvusdb/add caption", "blue", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"summary": summary, "metadata": metadata})
            doc = Document(page_content=summary, metadata=metadata)
            return await self.get_current_collection().aadd_documents([doc])

    @staticmethod
    def _escape(val: str) -> str:
        return val.replace("\\", "\\\\").replace("'", "\\'")

    async def aget_text_data(self, start_batch_index=0, end_batch_index=-1, uuid=""):
        # TODO(sl): make this truly async
        safe_uuid = self._escape(uuid)

        try:
            await asyncio.sleep(0.001)
            expr = f"content_metadata['doc_type'] == 'caption_summary' and \
                    content_metadata['batch_i'] >= {start_batch_index}"
            if end_batch_index != -1:
                expr += f" and content_metadata['batch_i'] <= {end_batch_index}"
            if safe_uuid:
                expr += f" and content_metadata['uuid'] == '{safe_uuid}'"
            logger.debug(
                f"Getting text data from MILVUS COLLECTION: {self.current_collection_name}"
            )
            logger.debug(f"Expression: {expr}")

            results = self.get_current_pymilvus_collection().query(
                expr=expr,
                output_fields=["*"],
            )

            # pks = self.vector_db.get_pks(expr=filter)
            # results = self.vector_db.get_by_ids(pks)
            # Donot include primary key pk in the returned metadata
            # Obtain content_metadata and flatten it with the rest of the metadata
            return [
                {
                    **{
                        k: v
                        for k, v in result.items()
                        if k != "pk" and k != "content_metadata"
                    },
                    **(
                        result.get("content_metadata", {})
                        if isinstance(result.get("content_metadata"), dict)
                        else {}
                    ),
                }
                for result in results
            ]
        except Exception as e:
            logger.warning(f"Error getting text data: {e}")
            return []

    async def aget_max_batch_index(self, uuid: str = ""):
        if uuid:
            safe_uuid = self._escape(uuid)
            expr = f"content_metadata[\"uuid\"] == '{safe_uuid}' and content_metadata[\"doc_type\"] == 'caption_summary'"
        else:
            expr = "content_metadata[\"doc_type\"] == 'caption_summary'"

        searched_metadata = self.get_current_pymilvus_collection().query(
            expr=expr,
            output_fields=["content_metadata"],
        )
        return max(
            [
                batch_index["content_metadata"]["batch_i"]
                for batch_index in searched_metadata
            ]
        )

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
        expr = "content_metadata['doc_type'] == 'caption'"

        if min_start_time is not None:
            expr += f" and content_metadata['start_ntp_float'] >= {min_start_time}"
        if max_start_time is not None:
            expr += f" and content_metadata['start_ntp_float'] <= {max_start_time}"
        if min_end_time is not None:
            expr += f" and content_metadata['end_ntp_float'] >= {min_end_time}"
        if max_end_time is not None:
            expr += f" and content_metadata['end_ntp_float'] <= {max_end_time}"
        if camera_id is not None:
            expr += f" and content_metadata['camera_id'] == '{camera_id}'"
        if chunk_id is not None:
            expr += f" and content_metadata['chunk_id'] == {chunk_id}"
        if uuid is not None:
            expr += f" and content_metadata['uuid'] == '{uuid}'"

        results = self.get_current_pymilvus_collection().query(
            expr=expr,
            output_fields=["*"],
        )
        return [
            {
                **{
                    k: v
                    for k, v in result.items()
                    if k != "pk" and k != "content_metadata" and k != "vector"
                },
                **(
                    result.get("content_metadata", {})
                    if isinstance(result.get("content_metadata"), dict)
                    else {}
                ),
            }
            for result in results
        ]

    def search(self, search_query, top_k=1):
        search_results = self.get_current_collection().similarity_search(
            search_query, k=top_k
        )
        return [result.metadata for result in search_results]

    def query(self, query, params: dict = {}):
        try:
            search_results = self.get_current_pymilvus_collection().query(
                query, output_fields=["*"]
            )
        except Exception as e:
            logger.warning(f"Error querying pymilvus collection: {e}")
            search_results = []
        return search_results

    def drop_data(self, expr="pk > 0"):
        try:
            self.get_current_pymilvus_collection().delete(expr=expr)
            self.get_current_pymilvus_collection().flush()
        except Exception as e:
            logger.warning(f"Error dropping data: {e}")

    def drop_collection(self):
        self._collection = Milvus(
            embedding_function=self.embedding,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=True,
        )
        self._current_collection = self._collection
        self.current_collection_name = self.collection_name
        self.is_user_specified_collection_name = False

    def reset(self, state: Optional[dict] = None):
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        if state is None:
            state = {}
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return
        delete_external_collection = state.get("delete_external_collection", False)
        if uuid and not erase_db:
            expr = f"content_metadata[\"uuid\"] == '{self._escape(uuid)}'"
        else:
            expr = "pk > 0"
        if self.is_user_specified_collection_name:
            if delete_external_collection:
                self.drop_data(expr)
        else:
            self.drop_data(expr)
        self.is_user_specified_collection_name = False
        # Always revert to the default collection
        self._current_collection = self._collection
        self.current_collection_name = self.collection_name

    def as_retriever(self, search_kwargs: dict = None) -> RetrieverLike:
        """
        This method is used to create a retriever for the Milvus database.
        It is used to retrieve documents from the Milvus database.
        """
        if search_kwargs is None:
            search_kwargs = {}
        return self.get_current_collection().as_retriever(search_kwargs=search_kwargs)
