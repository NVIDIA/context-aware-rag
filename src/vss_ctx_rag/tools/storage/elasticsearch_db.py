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

import asyncio
from typing import override, ClassVar, Optional, Dict, List, Any
import os
from langchain_core.retrievers import RetrieverLike
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_elasticsearch.vectorstores import ElasticsearchStore

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.tools.storage.storage_tool import DBConfig
from vss_ctx_rag.tools.storage.vector_storage_tool import VectorStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger


@register_tool_config("elasticsearch")
class ElasticsearchDBConfig(DBConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "embedding": ["embedding"],
    }


@register_tool(config=ElasticsearchDBConfig)
class ElasticsearchDBTool(VectorStorageTool):
    """Handler for Elasticsearch DB which stores the video embeddings mapped using
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

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "\n-"],
        )

        self.update_tool(self.config, tools)

    @override
    def update_tool(
        self,
        config: ElasticsearchDBConfig,
        tools: Optional[Dict[str, Dict[str, Tool]]] = None,
    ):
        """
        Updates the ElasticsearchDBTool configuration from a Pydantic config.

        Args:
            config: Configuration containing database settings
        """

        if not config.params.host:
            raise ValueError("Elasticsearch host not set in database configuration.")
        if not config.params.port:
            raise ValueError("Elasticsearch port not set in database configuration.")

        self.es_url = f"http://{config.params.host}:{config.params.port}"

        self.embedding = self.get_tool("embedding").embedding

        self.index_name = self.config.params.collection_name

        self._vector_store = ElasticsearchStore(
            index_name=self.index_name,
            embedding=self.embedding,
            es_url=self.es_url,
            strategy=ElasticsearchStore.ApproxRetrievalStrategy(hybrid=True, rrf=False),
            distance_strategy="COSINE",
        )

    def add_summary(self, summary: str, metadata: dict):
        with Metrics(
            "elasticsearch/add caption", "blue", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"summary": summary, "metadata": metadata})
            # Extract source before processing metadata to avoid duplication
            source = metadata.get("source", None) or metadata.get("file", None) or "N/A"

            # Create a copy to avoid modifying the original
            processed_metadata = metadata.copy()

            for key, value in processed_metadata.items():
                if isinstance(value, list):
                    processed_metadata[key] = str(value)
                elif isinstance(value, str) and value == "":
                    processed_metadata[key] = None

            metadata = {
                "source": source,
                "content_metadata": processed_metadata,
            }
            doc = Document(page_content=summary, metadata=metadata)
            logger.debug(
                f"Adding document to Elasticsearch index '{self.index_name}': {doc}"
            )

        try:
            return self._vector_store.add_documents([doc])
        except Exception as e:
            tm.error(e)
            logger.error(
                f"Invalid metadata while adding documents to Elasticsearch: {metadata}"
            )
            raise e

    def add_summaries(self, batch_summary: list[str], batch_metadata: list[dict]):
        with Metrics(
            "Elasticsearch/AddSummaries", "yellow", span_kind=Metrics.SPAN_KIND["TOOL"]
        ) as tm:
            tm.input({"batch_summary": batch_summary, "batch_metadata": batch_metadata})
            if len(batch_summary) != len(batch_metadata):
                raise ValueError(
                    "Incorrect param. The length of batch_summary batch and\
                    metadata batch should match."
                )
            docs = []
            for i in range(len(batch_summary)):
                docs.append(
                    Document(page_content=batch_summary[i], metadata=batch_metadata[i])
                )
            document_chunks = self.text_splitter.split_documents(docs)
            self._vector_store.add_documents(document_chunks)

    @staticmethod
    def _escape(val: str) -> str:
        return val.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')

    async def aget_text_data(self, start_batch_index=0, end_batch_index=-1, uuid=""):
        # TODO(sl): make this truly async
        safe_uuid = self._escape(uuid)

        try:
            await asyncio.sleep(0.001)

            # Build Elasticsearch query
            must_conditions = [
                {
                    "term": {
                        "metadata.content_metadata.doc_type.keyword": "caption_summary"
                    }
                },
                {
                    "range": {
                        "metadata.content_metadata.batch_i": {"gte": start_batch_index}
                    }
                },
            ]

            if end_batch_index != -1:
                must_conditions.append(
                    {
                        "range": {
                            "metadata.content_metadata.batch_i": {
                                "lte": end_batch_index
                            }
                        }
                    }
                )

            if safe_uuid:
                must_conditions.append(
                    {"term": {"metadata.content_metadata.uuid.keyword": safe_uuid}}
                )

            query = {"query": {"bool": {"must": must_conditions}}}

            logger.debug(
                f"Getting text data from Elasticsearch index: {self.index_name}"
            )
            logger.debug(f"Query: {query}")

            es_client = self._vector_store.client
            response = es_client.search(
                index=self.index_name,
                body=query,
                size=10000,
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result = source["metadata"]

                text_content = source.get("text", "")

                flattened = {
                    "text": text_content,
                    **{k: v for k, v in result.items() if k != "content_metadata"},
                    **(
                        result.get("content_metadata", {})
                        if isinstance(result.get("content_metadata"), dict)
                        else {}
                    ),
                }
                results.append(flattened)

            return results
        except Exception as e:
            logger.warning(f"Error getting text data: {e}")
            return []

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
        try:
            # Build Elasticsearch query
            must_conditions = [
                {
                    "term": {
                        "metadata.content_metadata.doc_type.keyword": "caption_summary"
                    }
                },
            ]

            if min_start_time is not None:
                must_conditions.append(
                    {
                        "range": {
                            "metadata.content_metadata.start_ntp_float": {
                                "gte": min_start_time
                            }
                        }
                    }
                )
            if max_start_time is not None:
                must_conditions.append(
                    {
                        "range": {
                            "metadata.content_metadata.start_ntp_float": {
                                "lte": max_start_time
                            }
                        }
                    }
                )
            if min_end_time is not None:
                must_conditions.append(
                    {
                        "range": {
                            "metadata.content_metadata.end_ntp_float": {
                                "gte": min_end_time
                            }
                        }
                    }
                )
            if max_end_time is not None:
                must_conditions.append(
                    {
                        "range": {
                            "metadata.content_metadata.end_ntp_float": {
                                "lte": max_end_time
                            }
                        }
                    }
                )
            if camera_id is not None:
                must_conditions.append(
                    {"term": {"metadata.content_metadata.camera_id.keyword": camera_id}}
                )
            if chunk_id is not None:
                must_conditions.append(
                    {"term": {"metadata.content_metadata.chunk_id.keyword": chunk_id}}
                )
            if uuid:
                safe_uuid = self._escape(uuid)
                must_conditions.append(
                    {"term": {"metadata.content_metadata.uuid.keyword": safe_uuid}}
                )

            query = {"query": {"bool": {"must": must_conditions}}}

            logger.debug(
                f"Getting text data from Elasticsearch index: {self.index_name}"
            )
            logger.debug(f"Query: {query}")

            es_client = self._vector_store.client
            response = es_client.search(
                index=self.index_name,
                body=query,
                size=10000,
            )

            results = []
            for hit in response["hits"]["hits"]:
                source = hit["_source"]
                result = source["metadata"]

                text_content = source.get("text", "")

                flattened = {
                    "text": text_content,
                    **{k: v for k, v in result.items() if k != "content_metadata"},
                    **(
                        result.get("content_metadata", {})
                        if isinstance(result.get("content_metadata"), dict)
                        else {}
                    ),
                }
                results.append(flattened)

            return results
        except Exception as e:
            logger.warning(f"Error getting text data: {e}")
            return []

    async def aget_max_batch_index(self, uuid: str = ""):
        try:
            must_conditions = [
                {
                    "term": {
                        "metadata.content_metadata.doc_type.keyword": "caption_summary"
                    }
                }
            ]

            if uuid:
                safe_uuid = self._escape(uuid)
                must_conditions.append(
                    {"term": {"metadata.content_metadata.uuid.keyword": safe_uuid}}
                )

            query = {
                "query": {"bool": {"must": must_conditions}},
                "aggs": {
                    "max_batch_i": {
                        "max": {"field": "metadata.content_metadata.batch_i"}
                    }
                },
                "size": 0,
            }

            es_client = self._vector_store.client
            response = es_client.search(index=self.index_name, body=query)

            max_batch_i = response["aggregations"]["max_batch_i"]["value"]
            return int(max_batch_i) if max_batch_i is not None else 0
        except Exception as e:
            logger.warning(f"Error getting max batch index: {e}")
            return 0

    def search(self, search_query, top_k=1):
        search_results = self._vector_store.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    def query(self, query, params: dict = {}):
        try:
            es_client = self._vector_store.client
            search_results = es_client.search(
                index=self.index_name, body=query, **params
            )
            return search_results["hits"]["hits"]
        except Exception as e:
            logger.warning(f"Error querying Elasticsearch index: {e}")
            return []

    def drop_data(self, query=None):
        try:
            es_client = self._vector_store.client
            if query is None:
                # Delete all documents
                query = {"query": {"match_all": {}}}

            es_client.delete_by_query(index=self.index_name, body=query)
        except Exception as e:
            logger.warning(f"Error dropping data: {e}")

    def drop_collection(self):
        try:
            es_client = self._vector_store.client
            es_client.indices.delete(index=self.index_name, ignore=[400, 404])

            # Recreate the vector store
            self._vector_store = ElasticsearchStore(
                index_name=self.index_name,
                embedding=self.embedding,
                es_url=self.es_url,
                strategy=ElasticsearchStore.ApproxRetrievalStrategy(
                    hybrid=True, rrf=False
                ),
                distance_strategy="COSINE",
            )
        except Exception as e:
            logger.warning(f"Error dropping collection: {e}")

    def reset(self, state: Optional[dict] = None):
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        if state is None:
            state = {}
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return

        if uuid and not erase_db:
            # Delete documents with specific UUID
            query = {
                "query": {
                    "term": {
                        "metadata.content_metadata.uuid.keyword": self._escape(uuid)
                    }
                }
            }
        else:
            # Delete all documents
            query = {"query": {"match_all": {}}}

        self.drop_data(query)

    def as_retriever(self, search_kwargs: dict = None) -> RetrieverLike:
        """
        This method is used to create a retriever for the Elasticsearch database.
        It is used to retrieve documents from the Elasticsearch database.
        """
        if search_kwargs is None:
            search_kwargs = {}
        return self._vector_store.as_retriever(search_kwargs=search_kwargs)
