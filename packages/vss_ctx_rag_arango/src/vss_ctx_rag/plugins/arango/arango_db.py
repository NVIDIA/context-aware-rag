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

import os
import traceback
from typing import List, Tuple, ClassVar, Optional, Any

import nx_arangodb as nxadb
from arango import ArangoClient
from adbnx_adapter import ADBNX_Adapter, ADBNX_Controller_Full_Cycle
from typing import Dict
import hashlib

from adbnx_adapter.typings import NxData, NxId

from langchain_arangodb import ArangoVector
from langchain_core.runnables import chain

from vss_ctx_rag.functions.rag.graph_rag.constants import (
    get_retrieval_query,
)
from vss_ctx_rag.plugins.arango.networkx_db import (
    NetworkXGraphDB,
)
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_EMBEDDING_DIMENSION,
    DEFAULT_TRAVERSAL_STRATEGY,
    GNN_TRAVERSAL_STRATEGY,
)
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.tools.storage.storage_tool import DBConfig
from typing import override


@register_tool_config("arango")
class ArangoDBConfig(DBConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {"embedding": ["embedding"]}
    write_batch_size: int = 50000
    multi_channel: bool = False
    traversal_strategy: str = DEFAULT_TRAVERSAL_STRATEGY


class CustomController(ADBNX_Controller_Full_Cycle):
    def __init__(self, collection_name: str):
        self.collection_name = collection_name

    def _identify_networkx_node(
        self, nx_node_id: NxId, nx_node: NxData, adb_v_cols: List[str]
    ) -> str:
        return f"{self.collection_name}_{str(nx_node_id).split('/')[0]}"

    def _identify_networkx_edge(
        self,
        nx_edge: NxData,
        from_node_id: NxId,
        to_node_id: NxId,
        nx_map: Dict[NxId, str],
        adb_e_cols: List[str],
    ) -> str:
        if str(from_node_id).split("/")[0] == str(to_node_id).split("/")[0] == "Entity":
            return f"{self.collection_name}_LINKS_TO"
        elif (
            str(from_node_id).split("/")[0] == str(to_node_id).split("/")[0] == "Chunk"
        ):
            if str(nx_edge["type"]) == "NEXT_CHUNK":
                return f"{self.collection_name}_NEXT_CHUNK"
            # elif str(nx_edge["type"]) == "SIMILAR":
            #     return f"{self.collection_name}_SIMILAR"
        elif str(nx_edge["type"]) == "FIRST_CHUNK":
            return f"{self.collection_name}_PART_OF"
        elif str(nx_edge["type"]) == "IN_SUMMARY":
            return f"{self.collection_name}_IN_SUMMARY"
        else:
            return f"{self.collection_name}_{str(nx_edge['type'])}"

    def _keyify_networkx_node(
        self, i: int, nx_node_id: NxId, nx_node: NxData, col: str
    ) -> str:
        return self._string_to_arangodb_key_helper(str(nx_node_id).split("/")[1])

    def _keyify_networkx_edge(
        self,
        i: int,
        nx_edge: NxData,
        from_node_id: NxId,
        to_node_id: NxId,
        nx_map: Dict[NxId, str],
        col: str,
    ) -> str:
        from_key = self._string_to_arangodb_key_helper(str(from_node_id).split("/")[1])
        to_key = self._string_to_arangodb_key_helper(str(to_node_id).split("/")[1])
        return self._string_to_arangodb_key_helper(f"{from_key}-{col}-{to_key}")


@register_tool(config=ArangoDBConfig)
class ArangoGraphDB(NetworkXGraphDB):
    """
    StorageTool for the ArangoDB graph.
    Uses the NetworkX graph internally (via the base class NetworkXGraphDB)
    while adding ArangoDB-specific functionality such as AQL query execution.
    """

    def __init__(
        self,
        name="arango_db",
        tools=None,
        config=None,
    ) -> None:
        """
        Initialize the ArangoGraphDB class.

        Args:
            collection_name: Name of the collection.
            host: Host of the ArangoDB server.
            port: Port of the ArangoDB server.
            write_batch_size: Write batch size.
            name: Name of the graph.

        Returns:
            None
        """
        super().__init__(
            name=name,
            tools=tools,
            config=config,
        )
        self.graph_online = None
        self.config = config
        self.embedding = self.get_tool("embedding")
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        """
        Create an ArangoGraphDB instance from configuration.

        Args:
            config: Configuration containing database and embedding settings

        Returns:
            ArangoGraphDB: Configured instance

        Raises:
            ValueError: If required configuration is not set
        """
        super().update_tool(config, tools)

        self.collection_name = config.params.collection_name
        self.write_batch_size = config.params.write_batch_size
        self.multi_channel = config.params.multi_channel
        self.traversal_strategy = config.params.traversal_strategy

        if self.multi_channel:
            self.collection_name = "default"

        self.collection_name = self.collection_name.replace("-", "_")
        self.embedding_field = "embedding"
        self.entity_collection = f"{self.collection_name}_Entity"
        self.chunk_collection = f"{self.collection_name}_Chunk"
        self.community_collection = f"{self.collection_name}_Community"
        self.document_collection = f"{self.collection_name}_Document"
        self.subtitle_collection = f"{self.collection_name}_Subtitle"

        self.entity_edge_collections = [
            f"{self.collection_name}_LINKS_TO",
            f"{self.collection_name}_HAS_ENTITY",
        ]

        self.part_of_collection = f"{self.collection_name}_PART_OF"
        self.next_chunk_collection = f"{self.collection_name}_NEXT_CHUNK"

        self.edge_definitions = [
            {
                "edge_collection": f"{self.collection_name}_HAS_ENTITY",
                "from_vertex_collections": [self.chunk_collection],
                "to_vertex_collections": [self.entity_collection],
            },
            {
                "edge_collection": self.part_of_collection,
                "from_vertex_collections": [self.chunk_collection],
                "to_vertex_collections": [self.document_collection],
            },
            {
                "edge_collection": f"{self.collection_name}_LINKS_TO",
                "from_vertex_collections": [self.entity_collection],
                "to_vertex_collections": [self.entity_collection],
            },
            {
                "edge_collection": self.next_chunk_collection,
                "from_vertex_collections": [self.chunk_collection],
                "to_vertex_collections": [self.chunk_collection],
            },
            {
                "edge_collection": f"{self.collection_name}_IN_SUMMARY",
                "from_vertex_collections": [self.chunk_collection],
                "to_vertex_collections": [self.community_collection],
            },
            {
                "edge_collection": f"{self.collection_name}_SUMMARY_OF",
                "from_vertex_collections": [self.community_collection],
                "to_vertex_collections": [self.document_collection],
            },
            # {
            #     "edge_collection": f"{self.collection_name}_SIMILAR",
            #     "from_vertex_collections": [self.chunk_collection],
            #     "to_vertex_collections": [self.chunk_collection],
            # },
        ]
        logger.debug(
            f"Initializing ArangoDB client with host: {config.params.host}, port: {config.params.port} and collection name: {self.collection_name}"
        )
        if config.params.username and config.params.password:
            self.db = ArangoClient(
                hosts=f"http://{config.params.host}:{config.params.port}"
            ).db(username=config.params.username, password=config.params.password)
        else:
            self.db = ArangoClient(
                hosts=f"http://{config.params.host}:{config.params.port}"
            ).db()

        if (
            os.environ.get("VSS_CTX_RAG_ENABLE_RET", "false").lower()
            in [
                "false",
                "0",
            ]
            and self.db.has_graph(self.collection_name)
            and not self.multi_channel
        ):
            self.db.delete_graph(self.collection_name, drop_collections=True)

    def upload_graph(self) -> None:
        """
        Upload the graph to ArangoDB.
        """
        logger.info(f"Uploading graph to ArangoDB: {self.collection_name}")

        if not self.db.has_graph(self.collection_name):
            self.db.create_graph(
                name=self.collection_name,
                edge_definitions=self.edge_definitions,
            )
        else:
            logger.info(
                f"Graph {self.collection_name} already exists, skipping creation"
            )
        self.adapter = ADBNX_Adapter(
            self.db, controller=CustomController(self.collection_name)
        )

        self.adapter.networkx_to_arangodb(
            name=self.collection_name,
            nx_graph=self.graph,
            batch_size=1000,
            on_duplicate="update",
            use_async=False,  # asynchronous DB insertions
        )

        self.graph_online = nxadb.MultiDiGraph(
            name=self.collection_name, db=self.db
        )  # no need to specify `incoming_graph_data` anymore

    def fetch_index_id(self, collection_name: str, index_name: str) -> Optional[str]:
        """
        Returns the id of an index for a collection/
        """
        indexes = self.db.collection(collection_name).indexes()
        for index in indexes:
            if index["name"] == index_name:
                return index["id"]

        return None

    def delete_index_by_name(self, collection_name: str, index_name: str) -> bool:
        """
        Delete the vector index for the graph.
        """
        index_id = self.fetch_index_id(collection_name, index_name)
        if index_id:
            self.db.collection(collection_name).delete_index(index_id)
            return True

        return False

    def create_vector_index(
        self, overwrite: bool = False, index_name: str = "vector_cosine"
    ) -> None:
        """
        Create a vector index for the graph.

        Returns:
            None
        """
        with Metrics("NXGraphExtraction/VectorIndex", "blue"):
            self.create_chunk_vector_index(overwrite, index_name)
            self.create_entity_vector_index(overwrite, index_name)

    def create_chunk_vector_index(
        self, overwrite: bool = False, index_name: str = "vector_cosine"
    ) -> None:
        """
        Create a vector index for the chunk collection.
        """
        with Metrics("NXGraphExtraction/VectorIndex/Chunk", "blue"):
            if overwrite:
                self.delete_index_by_name(self.chunk_collection, index_name)
            else:
                index_id = self.fetch_index_id(self.chunk_collection, index_name)
                if index_id:
                    logger.info(
                        f"Vector index {index_name} already exists for collection {self.chunk_collection}. Use **overwrite** to delete and recreate."
                    )
                    return

            logger.info(
                f"Creating vector index {index_name} for collection: {self.chunk_collection}"
            )

            chunk_count = len(
                [
                    node
                    for node, data in self.graph.nodes(data=True)
                    if data.get("type") == "Chunk"
                ]
            )
            chunk_nlists = min(10, max(1, chunk_count // 2)) if chunk_count > 0 else 1

            logger.info(
                f"Creating chunk vector index with {chunk_nlists} clusters for {chunk_count} documents"
            )
            if chunk_count > 0:
                self.db.collection(self.chunk_collection).add_index(
                    {
                        "type": "vector",
                        "fields": [self.embedding_field],
                        "name": index_name,
                        "params": {
                            "metric": "cosine",
                            "dimension": DEFAULT_EMBEDDING_DIMENSION,
                            "nLists": chunk_nlists,
                        },
                    }
                )

    def create_entity_vector_index(
        self, overwrite: bool = False, index_name: str = "vector_cosine"
    ) -> None:
        """
        Create a vector index for the entity collection.
        """
        with Metrics("NXGraphExtraction/VectorIndex/Entity", "blue"):
            if overwrite:
                self.delete_index_by_name(self.entity_collection, index_name)
            else:
                index_id = self.fetch_index_id(self.entity_collection, index_name)
                if index_id:
                    logger.info(
                        f"Vector index {index_name} already exists for collection {self.entity_collection}. Use **overwrite** to delete and recreate."
                    )
                    return

            logger.info(
                f"Creating vector index {index_name} for collection: {self.entity_collection}"
            )

            entity_count = len(
                [
                    node
                    for node, data in self.graph.nodes(data=True)
                    if data.get("label") == "__Entity__"
                ]
            )
            entity_nlists = (
                min(10, max(1, entity_count // 2)) if entity_count > 0 else 1
            )

            logger.info(
                f"Creating entity vector index with {entity_nlists} clusters for {entity_count} documents"
            )
            if entity_count > 0:
                self.db.collection(self.entity_collection).add_index(
                    {
                        "type": "vector",
                        "fields": [self.embedding_field],
                        "name": index_name,
                        "params": {
                            "metric": "cosine",
                            "dimension": DEFAULT_EMBEDDING_DIMENSION,
                            "nLists": entity_nlists,
                        },
                    }
                )

    def query(self, query: str, params: dict) -> list:
        """
        Execute an AQL query against ArangoDB and return the results.

        Args:
            aql_query: The AQL query to execute.
            bind_vars: A dictionary of bind parameters, e.g., {"query": [...]}

        Returns:
            A list of result documents.
        """
        try:
            with Metrics("NXGraphExtraction/AQLQuery", "blue"):
                if not self.graph_online:
                    logger.info("Graph not online, uploading graph")
                    self.upload_graph()
                cursor = self.graph_online.query(query, params)
                results = list(cursor)
                return results
        except Exception as e:
            logger.error("Failed executing AQL query: %s", e)
            raise e

    async def aget_max_batch_index(self, uuid: str = "") -> int:
        """
        Async method to get the maximum batch index from the Community collection.

        Args:
            uuid: The UUID to filter by (optional)

        Returns:
            int: The maximum batch index, or None if no summaries exist
        """
        try:
            if uuid:
                aql_query = f"""
                    FOR s IN {self.collection_name}_Community
                    FILTER s.uuid == @uuid
                    RETURN s.batch_i
                """
                params = {"uuid": uuid}
            else:
                aql_query = f"""
                    FOR s IN {self.collection_name}_Community
                    RETURN s.batch_i
                """
                params = {}

            result = self.query(aql_query, params)

            if result:
                batch_indices = [item for item in result if item is not None]
                return max(batch_indices) if batch_indices else None
            else:
                return None

        except Exception as e:
            logger.error(f"Error getting max batch index: {e}")
            return None

    async def finalize_graph_creation(self):
        """
        Finalize the graph creation process.
        """
        self.upload_graph()
        self.create_vector_index()

    def reset(self, state: dict = {}):
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return
        super().reset(state)
        if erase_db:
            graphs = self.db.graphs()
            logger.debug(f"Deleting all graphs: {graphs}")
            for graph in graphs:
                self.db.delete_graph(graph["name"], drop_collections=True)
            logger.info("Cleared the entire ArangoDB graph.")
        if self.graph_online is not None:
            try:
                if self.db.has_graph(self.collection_name):
                    self.db.delete_graph(self.collection_name, drop_collections=True)
            except Exception as e:
                logger.warning(
                    "Failed to delete graph '%s' from ArangoDB during reset: %s",
                    self.collection_name,
                    e,
                )
            finally:
                # Clear reference so that future truthiness checks do not trigger remote calls
                self.graph_online = None

    def format_source_docs(self, docs: list[dict]):
        formatted_docs = []
        for doc in docs:
            text_data = doc.get("text", "")
            if isinstance(text_data, dict):
                page_content = text_data.get("responseText", "")
                metadata = text_data.get("metadata", {})
            else:
                page_content = text_data
                metadata = doc.get("metadata", {})

            formatted_docs.append(
                {
                    "metadata": metadata,
                    "page_content": page_content,
                }
            )
        return formatted_docs

    def retrieve_documents(
        self,
        question: str,
        uuid: str = "default",
        multi_channel: bool = False,
        top_k: int = 5,
        retriever: str = "chunk",
    ) -> Tuple[List[str], List[str]]:
        """
        Retrieve the documents from the graph.

        Returns:
            List[str]: The formatted documents.
        """
        with Metrics(
            "GraphRetrieval/RetrieveDocuments",
            "blue",
            span_kind=Metrics.SPAN_KIND["RETRIEVER"],
        ) as tm:
            tm.input({"question": question})
            try:
                embedded_question = self.embedding.embed_query(question)

                documents = self.query(
                    get_retrieval_query(self.collection_name, retriever),
                    {
                        "query": embedded_question,
                        "topk_docs": top_k,
                    },
                )

                tm.output({"documents": documents})
                return self.format_documents(documents), self.format_source_docs(
                    documents
                )
            except Exception as e:
                logger.error(traceback.format_exc())
                error_message = f"Error retrieving documents: {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)

    def format_documents(self, documents: List[dict]) -> List[str]:
        """
        Format a list of documents into a string representation.

        For each document, prints:

        Document X Content:
        ## text goes here

        Entities
        from_id:-relationship_type:-to_id

        Args:
            documents (list): A list of document dictionaries.

        Returns:
            list: A list of formatted string representations for each document.
        """
        formatted_docs = []
        for doc_entry in documents:
            text = doc_entry.get("text", "")
            formatted_doc = f"Document start\nContent: {text}\nDocument end\n"
            formatted_docs.append(formatted_doc)

        return "\n\n".join(formatted_docs)

    def create_document_retriever(self):
        """
        Creates a document retriever from the graph.
        TODO: Implement when langchain supports ArangoDB retrieval.
        """
        pass

    def retrieve_documents_for_gnn(
        self, question: str, uuid: str, multi_channel: bool = False, top_k: int = None
    ) -> tuple[List[str], List[str], List[str], List[int], List[int], List[dict]]:
        """
        Retrieve documents from the graph formatted for Graph Neural Network processing.

        Args:
            question (str): The query string to search for.
            uuid (str): Unique identifier for the session/context.
            multi_channel (bool, optional): Whether to use multi-channel processing. Defaults to False.
            top_k (int, optional): Maximum number of documents to retrieve. Defaults to None.

        Returns:
            tuple: A 6-element tuple containing:
                - text_chunks_list (List[str]): List of relationship text descriptions
                - entities (List[str]): Unique list of all entities from the graph
                - relations (List[str]): List of relationship types between entities
                - entities_from_indices (List[int]): Source entity indices for relationships
                - entities_to_indices (List[int]): Target entity indices for relationships
                - formatted_source_docs (List[dict]): Source documents with metadata and content
        """
        with Metrics(
            "GraphRetrieval/RetrieveDocuments",
            "blue",
            span_kind=Metrics.SPAN_KIND["RETRIEVER"],
        ) as tm:
            tm.input({"question": question})
            embedded_question = self.embedding.embed_query(question)
            documents = self.query(
                get_retrieval_query(self.collection_name, GNN_TRAVERSAL_STRATEGY),
                {
                    "query": embedded_question,
                    "topk_docs": top_k,
                },
            )

            (
                chunks_list,  # Text chunks from retrieved docs
                text_entities_list,  # List of entities from retrieved docs
                text_relationship_list,  # List of relationships from retrieved docs
                edge_index_from,  # List of source entity indices for relationships. indices correspond to text_entities_list
                edge_index_to,  # List of target entity indices for relationships. indices correspond to text_entities_list
                raw_docs,
            ) = (
                *self.format_docs_for_gnn(documents),
                self.format_source_docs(documents),
            )

            payload_data = {
                "nodes": text_entities_list,
                "edges": text_relationship_list,
                "edge_indices": [edge_index_from, edge_index_to],
                "description": chunks_list,
            }
            return payload_data, raw_docs

    def format_docs_for_gnn(
        self, documents: List[dict]
    ) -> tuple[List[str], List[str], List[str], List[int], List[int]]:
        """
        Format a list of documents into components required for GNN processing.

        Extracts relationship texts, entities, and relations from the first document in the list.
        Creates a unique list of entities by combining source and target nodes from relationships.

        Args:
            documents (List[dict]): A list of document dictionaries. Only the first document is processed.
                                   Expected to contain keys: 'relTexts', 'sourceNodes', 'targetNodes', 'relations'

        Returns:
            tuple: A 5-element tuple containing:
                - text_chunks_list (List[str]): List of relationship text descriptions
                - entities (List[str]): Unique list of all entities (combined from source and target nodes)
                - relations (List[str]): List of relationship types
                - entities_from_indices (List[int]): List of source node entity indices from the above unique list
                - entities_to_indices (List[int]): List of target node entity indices from the above unique list
        """
        # All necessary info extracted below is repeated in all the docs
        # Hence we are extracting it from the first document only
        if documents and len(documents) > 0:
            text_chunks_list = documents[0].get("relTexts", [])
            unique_entities = set()
            source_entities = documents[0].get("sourceNodes", [])
            target_entities = documents[0].get("targetNodes", [])
            relations = documents[0].get("relations", [])
            unique_entities.update(source_entities)
            unique_entities.update(target_entities)
            entities = list(unique_entities)
            entities_from = [entities.index(entity) for entity in source_entities]
            entities_to = [entities.index(entity) for entity in target_entities]
            return text_chunks_list, entities, relations, entities_from, entities_to
        else:
            return [], [], [], [], []

    def as_retriever(self, search_kwargs: dict = None):
        """
        Creates a document retriever from the graph.
        """
        if search_kwargs is None:
            search_kwargs = {}

        text_field = "text"
        vector_index_name = "vector_cosine"
        keyword_index_name = "keyword_index"

        try:
            vector_index = ArangoVector(
                embedding=self.embedding,
                embedding_dimension=DEFAULT_EMBEDDING_DIMENSION,
                database=self.db,
                collection_name=self.chunk_collection,
                search_type="hybrid",
                embedding_field=self.embedding_field,
                text_field=text_field,
                vector_index_name=vector_index_name,
                keyword_index_name=keyword_index_name,
            )

            logger.info(
                f"Successfully retrieved ArangoVector index for collection '{self.chunk_collection}' and keyword index '{keyword_index_name}'"
            )

            @chain
            def retriever(query: str):
                docs_and_scores = vector_index.similarity_search_with_score(
                    query,
                    k=search_kwargs.get("key", 10),
                    score_threshold=search_kwargs.get("score_threshold", 0.8),
                )
                for doc, score in docs_and_scores:
                    doc.metadata["score"] = score
                return [doc for doc, _ in docs_and_scores]

            return retriever

        except Exception as e:
            logger.error(
                f"Error retrieving ArangoVector index for collection {self.chunk_collection}: {e}"
            )
            raise

    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter chunks based on various criteria.

        Args:
            min_start_time: Minimum start time for chunks
            max_end_time: Maximum end time for chunks
            camera_id: Camera ID to filter by
            uuid: Document UUID to filter by

        Returns:
            List of dictionaries containing chunk data
        """
        filters: List[str] = []
        params: Dict[str, Any] = {"@collection": self.chunk_collection}

        if min_start_time is not None:
            filters.append("c.start_time >= @min_start_time")
            params["min_start_time"] = min_start_time
        if max_end_time is not None:
            filters.append("c.end_time <= @max_end_time")
            params["max_end_time"] = max_end_time
        if camera_id:
            filters.append("c.camera_id == @camera_id")
            params["camera_id"] = camera_id
        if uuid:
            filters.append("c.uuid == @uuid")
            params["uuid"] = uuid

        filter_clause = f"FILTER {' AND '.join(filters)}" if filters else ""

        aql = f"""
            FOR c IN @@collection
            {filter_clause}
            RETURN {{
                text: c.text,
                start_time: c.start_time,
                end_time: c.end_time,
                chunk_id: c._key,
                camera_id: c.camera_id
            }}
        """

        try:
            result = self.query(aql, params)
            return result if result else []
        except Exception as e:
            logger.error(f"AQL Query failed: {str(e)}")
            return []

    def get_neighbors(self, node_id: int) -> List[Dict[str, Any]]:
        # Identify the collection the node belongs to
        if self.db.collection(self.entity_collection).has(str(node_id)):
            with_collection = self.entity_collection
            node_id = f"{self.entity_collection}/{node_id}"
        elif self.db.collection(self.chunk_collection).has(str(node_id)):
            with_collection = self.chunk_collection
            node_id = f"{self.chunk_collection}/{node_id}"
        elif self.db.collection(self.document_collection).has(str(node_id)):
            with_collection = self.document_collection
            node_id = f"{self.document_collection}/{node_id}"
        elif self.db.collection(self.community_collection).has(str(node_id)):
            with_collection = self.community_collection
            node_id = f"{self.community_collection}/{node_id}"
        else:
            logger.error(f"Node {node_id} not found in any collection")
            return []

        aql = f"""
            WITH {with_collection}
            FOR v,e,p IN 1..1 ANY @node_id GRAPH @graph_name
                LET edge_collection = PARSE_IDENTIFIER(e._id).collection
                FILTER edge_collection != @part_of_collection

                LET node_collection = PARSE_IDENTIFIER(v._id).collection

                LET connected_data = (
                    node_collection == @entity_collection ?
                    {{
                        id: v._key,
                        description: v.description,
                        name: v.name,
                        type: 'Entity'
                    }} :
                    node_collection == @chunk_collection ?
                    {{
                        id: v._key,
                        text: v.text,
                        start_time: v.start_time,
                        end_time: v.end_time,
                        type: 'Chunk'
                    }} :
                    {{
                        id: v._key,
                        type: node_collection
                    }}
                )

                RETURN {{
                    node: connected_data,
                    relationship: {{type: edge_collection}}
                }}
        """

        params = {
            "node_id": node_id,
            "graph_name": self.collection_name,
            "entity_collection": self.entity_collection,
            "chunk_collection": self.chunk_collection,
            "part_of_collection": self.part_of_collection,
        }

        result = self.query(aql, params)
        if not result:
            return []

        return result

    def get_next_chunks(self, chunk_id: int, number_of_hops: int = 1) -> Dict[str, Any]:
        aql = f"""
            WITH {self.chunk_collection}
            FOR v, e, p IN {number_of_hops}..{number_of_hops} OUTBOUND @chunk_id @@next_chunk_collection
                RETURN {{
                    connected_chunk: {{
                        id: v._key,
                        text: v.text,
                        start_time: v.start_time,
                        end_time: v.end_time
                    }}
                }}
        """

        params = {
            "chunk_id": f"{self.chunk_collection}/{chunk_id}",
            "@next_chunk_collection": self.next_chunk_collection,
        }

        result = self.query(aql, params)
        if not result:
            return {"connected_chunk": None}

        return result[0]

    def get_chunk_asset_dir(self, chunk_id) -> Optional[str]:
        aql = """
            FOR c IN @@collection
                FILTER c._key == @chunk_id
                RETURN c.asset_dir
        """

        params = {"@collection": self.chunk_collection, "chunk_id": str(chunk_id)}

        result = self.query(aql, params)
        if not result:
            return None

        return result[0]

    def get_chunk_time_range(self, chunk_id) -> Optional[Tuple[float, float]]:
        aql = """
            FOR c IN @@collection
                FILTER c._key == @chunk_id
                RETURN {start_time: c.start_time, end_time: c.end_time}
        """

        params = {"@collection": self.chunk_collection, "chunk_id": str(chunk_id)}

        result = self.query(aql, params)
        if not result:
            return None

        return result[0]["start_time"], result[0]["end_time"]

    def get_asset_dirs_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[str]:
        aql = """
            FOR c IN @@collection
                FILTER c.start_time >= @start_time AND c.end_time <= @end_time
                RETURN DISTINCT c.asset_dir
        """

        params = {
            "@collection": self.chunk_collection,
            "start_time": start_time,
            "end_time": end_time,
        }

        result = self.query(aql, params)
        if not result:
            return []

        return result

    def filter_subtitles_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        aql = """
            FOR s IN @@collection
                FILTER s.start_time >= @start_time AND s.end_time <= @end_time
                RETURN {text: s.text, start_time: s.start_time, end_time: s.end_time}
        """

        params = {
            "@collection": self.subtitle_collection,
            "start_time": start_time,
            "end_time": end_time,
        }

        result = self.query(aql, params)
        if not result:
            return []

        return result

    def get_duplicate_nodes_list(self, duplicate_score_value):
        aql = """
            LET similarGroups = (
                FOR e1 IN @@collection
                    FILTER e1.embedding AND e1.name

                    LET similar = (
                        FOR e2 IN @@collection
                            FILTER e1._key < e2._key
                            AND ATTRIBUTES(e1) == ATTRIBUTES(e2)

                            LET score = COSINE_SIMILARITY(e1.embedding, e2.embedding)
                            FILTER score > @duplicate_score_value
                            SORT score DESC

                            RETURN {
                                _id: e2._id,
                                _key: e2._key,
                                name: e2.name,
                                description: e2.description,
                                camera_id: e2.camera_id
                            }
                    )

                    FILTER LENGTH(similar) > 0

                    LET entity = {
                        _id: e1._id,
                        _key: e1._key,
                        name: e1.name,
                        description: e1.description,
                        camera_id: e1.camera_id
                    }

                    RETURN {entity, similar}
            )

            LET allSubsetRelations = (
                FOR group1 IN similarGroups
                    FOR group2 IN similarGroups
                        FILTER group1.entity._key != group2.entity._key
                        AND LENGTH(group1.similar) < LENGTH(group2.similar)

                        // Check if group1 is a subset of group2
                        LET group1Keys = (FOR item IN group1.similar RETURN item._key)
                        LET group2Keys = (FOR item IN group2.similar RETURN item._key)
                        LET missingKeys = MINUS(group1Keys, group2Keys)

                        FILTER LENGTH(missingKeys) == 0
                        RETURN {
                            subsetGroup: group1.entity._key,
                            supersetGroup: group2.entity._key
                        }
            )

            FOR group IN similarGroups
                // Check if this group is a subset of any other group
                LET isSubset = LENGTH(FOR rel IN allSubsetRelations FILTER rel.subsetGroup == group.entity._key RETURN 1) > 0

                // Only process non-subset groups
                FILTER NOT isSubset

                // Find all entities that should be merged into this group
                LET entitiesToMerge = (
                    FOR rel IN allSubsetRelations
                        FILTER rel.supersetGroup == group.entity._key
                        LET subsetGroup = FIRST(FOR sg IN similarGroups FILTER sg.entity._key == rel.subsetGroup RETURN sg)
                        RETURN subsetGroup.entity
                )

                // Combine original similar entities with merged entities (deduplicated)
                LET mergedSimilar = UNION_DISTINCT(group.similar, entitiesToMerge)

                RETURN { e: group.entity, similar: mergedSimilar}
        """

        params = {
            "@collection": self.entity_collection,
            "duplicate_score_value": duplicate_score_value,
        }

        nodes_list = self.query(aql, params)

        return nodes_list

    def merge_duplicate_nodes(self, duplicate_score_value: float):
        with Metrics("GraphRAG/ArangoDB/merge_duplicate_nodes", "yellow"):
            # Ensure Vector Index
            index_name = "vector_cosine"
            self.create_entity_vector_index(index_name=index_name)
            # Get Duplicate Nodes
            duplicate_nodes_list = self.get_duplicate_nodes_list(duplicate_score_value)
            logger.debug(f"Nodes list to merge: {len(duplicate_nodes_list)} groups")
            logger.debug(f"Nodes list to merge: {duplicate_nodes_list}")

            # Transform the data structure to match what the query expects
            rows_to_merge = []
            for node_group in duplicate_nodes_list:
                # Extract the main node and its similar nodes
                main_node = node_group.get("e", {})
                similar_nodes = node_group.get("similar", [])

                if not main_node or not similar_nodes or "_key" not in main_node:
                    logger.warning(f"Skipping invalid node group: {node_group}")
                    continue

                #######################################
                # NOTE: ArangoDB-specific Change      #
                # camera_id is added here to optimize #
                # merge_duplicate_nodes_query()       #
                #######################################

                first_element = {
                    "_id": main_node["_id"],
                    "_key": main_node["_key"],
                    "camera_id": main_node.get("camera_id"),
                }

                similar_elements = [
                    {
                        "_id": n["_id"],
                        "_key": n["_key"],
                        "camera_id": n.get("camera_id"),
                    }
                    for n in similar_nodes
                    if "_key" in n
                ]

                #######################################

                if not similar_elements:
                    logger.info(
                        f"No similar nodes to merge for {main_node.get('name', 'unknown')}"
                    )
                    continue

                # Log what we're merging for debugging
                logger.info(
                    f"Merging node {main_node.get('name', 'unknown')} with {len(similar_elements)} similar nodes"
                )
                logger.info(
                    f"Similar nodes: {[(n.get('name', 'unknown'), n.get('description', 'unknown')) for n in similar_nodes]}"
                )

                # Add to the rows to process
                rows_to_merge.append(
                    {
                        "firstElement": first_element,
                        "similarElements": similar_elements,
                    }
                )

            if not rows_to_merge:
                logger.warning("No valid node groups to merge")
                return {"totalMerged": 0}

            # Execute the merge query with the transformed data

            ####
            # Step 1) Update Edges, Remove Similar Nodes

            params = {
                "rows_to_merge": rows_to_merge,
                "@collection": self.entity_collection,
            }

            update_edges_aql = ""
            for i, edge_collection in enumerate(self.entity_edge_collections):
                update_edges_aql += f"""
                    LET update_{i} = (
                        FOR e IN @@edge_collection_{i}
                            FILTER e._from == similarNode._id OR e._to == similarNode._id
                            LET updateWith =  e._from == similarNode._id ? {{_from: mainNode._id}} : {{_to: mainNode._id}}
                            UPDATE e WITH updateWith IN @@edge_collection_{i}
                    )
                """

                params[f"@edge_collection_{i}"] = edge_collection

            aql = f"""
                FOR row IN @rows_to_merge

                    LET mainNode = row.firstElement
                    LET similarNodes = row.similarElements

                    LET migrate = (
                        FOR similarNode IN similarNodes
                            {update_edges_aql}

                            REMOVE similarNode._key IN @@collection OPTIONS {{ ignoreErrors: true }}
                    )

                    RETURN LENGTH(similarNodes) + 1
            """

            result = self.query(aql, params)
            ####

            ####
            # Step 2) Update Main Nodes with merged camera_ids

            update_nodes = []
            for row in rows_to_merge:
                main_node = row["firstElement"]
                main_camera_id = main_node["camera_id"]
                if not main_camera_id:
                    merged_camera_ids = set()
                elif isinstance(main_camera_id, list):
                    merged_camera_ids = set(main_camera_id)
                else:
                    merged_camera_ids = {main_camera_id}

                merged_camera_ids.update(
                    n["camera_id"] if isinstance(n["camera_id"], str) else x
                    for n in row["similarElements"]
                    if n["camera_id"]
                    for x in (
                        n["camera_id"]
                        if isinstance(n["camera_id"], list)
                        else [n["camera_id"]]
                    )
                )

                merged_camera_ids = list(merged_camera_ids)

                update_nodes.append(
                    {"_id": main_node["_id"], "camera_id": merged_camera_ids}
                )

            # NOTE: Consider batching this update operation if **update_nodes** is large.
            self.db.collection(self.entity_collection).update_many(update_nodes)
            ####

            # Log the result
            total_merged = result[0] if result else 0
            logger.info(f"Successfully merged {total_merged} nodes")

        return result

    def get_num_cameras(self) -> int:
        query = """
        FOR d IN @@collection
        FILTER d.camera_id != null
        COLLECT camera_id = d.camera_id
        RETURN LENGTH(1)
        """
        params = {"@collection": self.document_collection}
        result = self.query(query, params)
        return len(result) if result else 0

    def get_video_length(self):
        query = """
            FOR n IN @@chunk_collection
                LET nextEdges = (
                    FOR e IN @@next_chunk_collection
                    FILTER e._from == n._id
                    LIMIT 1
                    RETURN 1
                )
                FILTER LENGTH(nextEdges) == 0
                COLLECT camera_id = n.camera_id INTO group
                LET latest = MAX(group[*].n.end_time)
                RETURN { camera_id: camera_id, end_time: latest }
            """

        params = {
            "@chunk_collection": self.chunk_collection,
            "@next_chunk_collection": self.next_chunk_collection,
        }

        result = self.query(query, params)

        # Convert the result list to a map/dictionary format like the original Cypher
        camera_endtime_map = {}
        if result:
            for item in result:
                camera_endtime_map[item["camera_id"]] = item["end_time"]

        return camera_endtime_map

    def get_chunk_size(self) -> Dict[str, float]:
        query = """
        FOR c IN @@chunk_collection
            COLLECT camera_id = c.camera_id
            AGGREGATE maxChunkSize = MAX(c.end_time - c.start_time)
            RETURN {camera_id, maxChunkSize}
        """
        params = {"@chunk_collection": self.chunk_collection}
        result = self.query(query, params)

        # Transform list of records into dictionary format {"camera_id": chunk_size}
        if result:
            return {record["camera_id"]: record["maxChunkSize"] for record in result}
        return {}

    def get_chunk_camera_id(self, chunk_id):
        query = """
        FOR c IN @@chunk_collection
            FILTER c._key == @chunk_id
            RETURN {camera_id: c.camera_id}
        """
        params = {"@chunk_collection": self.chunk_collection, "chunk_id": chunk_id}
        result = self.query(query, params)
        return result[0]["camera_id"] if result else ""

    @override
    def update_knn(self):
        logger.info("KNN disabled for ArangoDB")

    def persist_subtitle_frames(self, subtitle_frames: list[dict]):
        with Metrics("GraphRAG/ArangoDB/persist_subtitle_frames", "yellow"):
            if not subtitle_frames:
                logger.info("No subtitle frames to persist")
                return
            try:
                for subtitle_frame in subtitle_frames:
                    key_material = f"{subtitle_frame['start_time']}|{subtitle_frame['end_time']}|{subtitle_frame['text']}".encode(
                        "utf-8"
                    )
                    key_hash = hashlib.sha1(key_material).hexdigest()
                    subtitle_frame["_key"] = key_hash

                self.db.collection(self.subtitle_collection).import_bulk(
                    subtitle_frames, on_duplicate="update"
                )
                logger.info(
                    f"Successfully persisted {len(subtitle_frames)} subtitle frames"
                )
            except Exception as e:
                logger.error(f"Failed to persist subtitle frames: {e}")
                raise

    def fetch_subtitle_for_embedding(self) -> List[Dict[str, Any]]:
        """Fetch subtitle nodes without embeddings for embedding processing."""
        with Metrics("GraphRAG/ArangoDB/fetch_subtitle_for_embedding", "green"):
            query = """
                LET subtitles = (
                    FOR s IN @@collection
                        FILTER s.embedding IS NULL AND s.text IS NOT NULL
                        RETURN {elementId: s._key, text: s.text}
                )

                RETURN subtitles
            """
            try:
                result = self.query(query, {"@collection": self.subtitle_collection})

                if not result:
                    return []

                return result[0]
            except Exception as e:
                logger.error(f"Failed to fetch subtitles for embedding: {e}")
                return []

    def persist_subtitle_embeddings(self, rows_with_embeddings: List[Dict]):
        with Metrics("GraphRAG/ArangoDB/persist_subtitle_embeddings", "yellow"):
            if not rows_with_embeddings:
                logger.info("No subtitle embeddings to persist")
                return

            aql = """
                FOR row IN @rows
                    UPDATE {_key: row.elementId} WITH {embedding: row.embedding} IN @@collection
            """

            params = {
                "rows": rows_with_embeddings,
                "@collection": self.subtitle_collection,
            }

            try:
                self.query(aql, params=params)
                logger.info(
                    f"Successfully persisted embeddings for {len(rows_with_embeddings)} subtitles"
                )
            except Exception as e:
                logger.error(f"Failed to persist subtitle embeddings: {e}")
                raise
