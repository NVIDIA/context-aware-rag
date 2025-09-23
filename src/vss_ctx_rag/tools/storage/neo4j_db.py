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
import time
import traceback
from typing import Any, Dict, List, Tuple, ClassVar, Optional

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    DocumentCompressorPipeline,
    EmbeddingsFilter,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_community.vectorstores import Neo4jVector
from langchain_core.runnables import chain
from neo4j.time import DateTime

from vss_ctx_rag.functions.rag.graph_rag.constants import (
    CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
    CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
    CHUNK_VECTOR_INDEX_NAME,
    CREATE_CHUNK_VECTOR_INDEX_QUERY,
    DROP_CHUNK_VECTOR_INDEX_QUERY,
    DROP_INDEX_QUERY,
    FILTER_LABELS,
    FULL_TEXT_QUERY,
    KEYWORD_SEARCH_FULL_TEXT_QUERY,
    KEYWORD_SEARCH_INDEX_DROP_QUERY,
    LABELS_QUERY,
    QUERY_TO_DELETE_UUID_GRAPH,
    VECTOR_GRAPH_SEARCH_QUERY,
    GNN_VECTOR_GRAPH_SEARCH_QUERY,
    VECTOR_SEARCH_TOP_K,
    ENTITY_SEARCH_QUERY_FORMATTED,
    SUBTITLE_SEARCH_FULL_TEXT_QUERY,
)

from vss_ctx_rag.functions.rag.graph_rag.planner_constants import (
    CHUNK_SEARCH_QUERY,
    PLANNER_ENTITY_SEARCH_QUERY,
    SUBTITLE_SEARCH_QUERY_FORMATTED,
)

from vss_ctx_rag.utils.globals import (
    DEFAULT_EMBEDDING_PARALLEL_COUNT,
    DEFAULT_CONCURRENT_EMBEDDING_LIMIT,
    DEFAULT_TRAVERSAL_STRATEGY,
    GNN_TRAVERSAL_STRATEGY,
)
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.utils import remove_lucene_chars
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.tools.storage.storage_tool import DBConfig
from langchain_core.documents import Document
import logging

neo4j_log = logging.getLogger("neo4j")
neo4j_log.setLevel(logging.CRITICAL)


@register_tool_config("neo4j")
class Neo4jDBConfig(DBConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {"embedding": ["embedding"]}

    traversal_strategy: str = DEFAULT_TRAVERSAL_STRATEGY
    embedding_parallel_count: int = DEFAULT_EMBEDDING_PARALLEL_COUNT


@register_tool(config=Neo4jDBConfig)
class Neo4jGraphDB(GraphStorageTool):
    def __init__(
        self,
        name="graph_db",
        tools=None,
        config=None,
    ) -> None:
        super().__init__(name, config, tools)
        self.embedding = self.get_tool("embedding").embedding
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "\n-"],
        )
        self.update_tool(self.config, tools)

    def _connect_to_neo4j(
        self,
        neo4j_host,
        neo4j_port,
        neo4j_username,
        neo4j_password,
        max_retries=5,
        delay_seconds=10,
    ):
        """Connect to Neo4j database with retry logic."""
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    f"Attempting to connect to Neo4j (attempt {attempt}/{max_retries}) at {neo4j_host}:{neo4j_port}"
                )
                self.graph_db = Neo4jGraph(
                    url=f"bolt://{neo4j_host}:{neo4j_port}",
                    username=neo4j_username,
                    password=neo4j_password,
                    sanitize=True,
                    refresh_schema=False,
                )
                logger.info("Successfully connected to Neo4j database.")
                break
            except Exception as e:
                logger.error(f"Neo4j connection attempt {attempt} failed: {e}")
                if attempt < max_retries:
                    logger.info(
                        f"Retrying Neo4j connection in {delay_seconds} seconds..."
                    )
                    time.sleep(delay_seconds)
                else:
                    logger.critical(
                        f"All {max_retries} attempts to connect to Neo4j failed."
                    )
                    raise

    def update_tool(self, config, tools=None):
        neo4j_host = config.params.host
        neo4j_port = config.params.port

        if not neo4j_host:
            raise ValueError("Neo4j host not set in database configuration.")
        if not neo4j_port:
            raise ValueError("Neo4j port not set in database configuration.")

        neo4j_username = config.params.username
        neo4j_password = config.params.password

        if not neo4j_username:
            raise ValueError("Neo4j username not set in database configuration.")
        if not neo4j_password:
            raise ValueError("Neo4j password not set in database configuration.")

        self._connect_to_neo4j(neo4j_host, neo4j_port, neo4j_username, neo4j_password)

        self.traversal_strategy = config.params.traversal_strategy
        self.embedding_parallel_count = config.params.embedding_parallel_count
        self.config = config
        self._embedding_semaphore = asyncio.Semaphore(
            DEFAULT_CONCURRENT_EMBEDDING_LIMIT
        )

        self.create_chunk_vector_index()

    def extract_cypher(self, text: str) -> str:
        """Extract Cypher code from a text.

        Args:
            text: Text to extract Cypher code from.

        Returns:
            Cypher code extracted from the text.
        """
        # The pattern to find Cypher code enclosed in triple backticks
        # pattern = r"```cypher(.*?)```"

        # # Find all matches in the input text
        # matches = re.findall(pattern, text, re.DOTALL)
        def find_between(s, first, last):
            try:
                start = s.index(first) + len(first)
                end = s.index(last, start)
                return s[start:end]
            except ValueError:
                return ""

        logger.debug("Generated Query: %s", text)
        start = "CYPHER_START"
        end = "CYPHER_END"
        result = find_between(text, start, end)
        logger.debug("Extracted Query: %s", result)

        return result if result else text

    def query(self, query, params: dict = {}):
        logger.debug(f"Query: {query}")
        try:
            result = self.graph_db.query(query, params)
            logger.debug(f"Query exec result: {result}")
            return result
        except Exception as e:
            logger.error("Neo4j Query failed %s", str(e))
            return None

    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        filters: List[str] = []
        params: Dict[str, Any] = {}

        if min_start_time is not None:
            filters.append("c.start_time >= toFloat($min_start_time)")
            params["min_start_time"] = min_start_time

        if max_end_time is not None:
            filters.append("c.end_time <= toFloat($max_end_time)")
            params["max_end_time"] = max_end_time
        if camera_id:
            filters.append("c.camera_id = $camera_id")
            params["camera_id"] = camera_id
        if uuid:
            filters.append("c.uuid = $uuid")
            params["uuid"] = uuid

        where_clause = f"WHERE {' AND '.join(filters)} " if filters else ""
        cypher = (
            "MATCH (c:Chunk) "
            + where_clause
            + "RETURN [el in collect(c) | {text: el.text, start_time: el.start_time, end_time: el.end_time, chunk_id: id(el), camera_id: el.camera_id}] as values"
        )
        result = self.query(cypher, params)
        if not result:
            return []
        row = result[0]
        return row.get("values", [])

    def get_neighbors(self, node_id: int) -> List[Dict[str, Any]]:
        cypher = (
            "MATCH (n)-[r]-(connected) "
            "WHERE id(n) = $node_id AND type(r) <> 'PART_OF' "
            "WITH n, r, connected, "
            "CASE WHEN '__Entity__' IN labels(connected) THEN {id: id(connected), description: connected.description, name: connected.name, type: 'Entity'} "
            "WHEN 'Chunk' IN labels(connected) THEN {id: id(connected), text: connected.text, start_time: connected.start_time, end_time: connected.end_time, type: 'Chunk'} "
            "ELSE {id: id(connected), type: head(labels(connected))} END AS connected_data "
            "RETURN collect(DISTINCT {node: connected_data, relationship: {type: type(r)}}) AS connected_info"
        )
        result = self.query(cypher, {"node_id": node_id})
        if not result:
            return []
        return result[0].get("connected_info", [])

    def get_next_chunks(self, chunk_id, number_of_hops: int = 1) -> Dict[str, Any]:
        cypher = (
            f"MATCH (start:Chunk)-[r:NEXT_CHUNK*{number_of_hops}]->(connected) "
            "WHERE id(start) = toInteger($chunk_id) "
            "RETURN {connected_chunk: {id: id(connected), text: connected.text, start_time: connected.start_time, end_time: connected.end_time}} AS result"
        )
        result = self.query(cypher, {"chunk_id": chunk_id})
        if not result:
            return {"connected_chunk": None}
        return result[0].get("result", {"connected_chunk": None})

    def get_chunk_asset_dir(self, chunk_id) -> Optional[str]:
        cypher = "MATCH (c:Chunk) WHERE id(c) = toInteger($chunk_id) RETURN c.asset_dir as asset_dir LIMIT 1"
        result = self.query(cypher, {"chunk_id": chunk_id})
        if result and result[0].get("asset_dir"):
            return result[0]["asset_dir"]
        return None

    def get_chunk_time_range(self, chunk_id) -> Optional[Tuple[float, float]]:
        cypher = "MATCH (c:Chunk) WHERE id(c) = toInteger($chunk_id) RETURN c.start_time as start_time, c.end_time as end_time LIMIT 1"
        result = self.query(cypher, {"chunk_id": chunk_id})
        if result:
            return result[0].get("start_time"), result[0].get("end_time")
        return None

    def get_asset_dirs_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[str]:
        cypher = "MATCH (c:Chunk) WHERE c.start_time >= toFloat($start_time) AND c.end_time <= toFloat($end_time) RETURN DISTINCT(c.asset_dir) as asset_dir"
        result = self.query(cypher, {"start_time": start_time, "end_time": end_time})
        if not result:
            return []
        return [row.get("asset_dir") for row in result if row.get("asset_dir")]

    def filter_subtitles_by_time_range(
        self, start_time: float, end_time: float
    ) -> List[Dict[str, Any]]:
        cypher = (
            "MATCH (s:Subtitle) WHERE s.start_time >= toFloat($start_time) AND s.end_time <= toFloat($end_time) "
            "RETURN [el in collect(s) | {text: el.text, start_time: el.start_time, end_time: el.end_time}] as values"
        )
        result = self.query(cypher, {"start_time": start_time, "end_time": end_time})
        if not result:
            return []
        return result[0].get("values", [])

    def datetime_encoder(self, obj):
        if isinstance(obj, DateTime):
            return obj.to_native().isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def aquery(self, query: str) -> List[Dict]:
        """Async wrapper around run_cypher_query"""
        return await asyncio.to_thread(self.query, query)

    async def aget_text_data(self, start_batch_index=0, end_batch_index=-1, uuid=""):
        """Async method to retrieve text data based on filter criteria.

        Args:
            start_batch_index: The start batch index
            end_batch_index: The end batch index
            uuid: The UUID of the document

        Returns:
            List of dictionaries containing the content and batch_i values
        """
        if self.graph_db:
            await asyncio.sleep(0.001)
            end_condition = (
                f"AND s.batch_i <= {end_batch_index}" if end_batch_index != -1 else ""
            )
            if uuid:
                query = f"MATCH (s:Summary) WHERE s.batch_i >= {start_batch_index} {end_condition} AND s.uuid = '{uuid}' RETURN s.content, s.batch_i"
            else:
                query = f"MATCH (s:Summary) WHERE s.batch_i >= {start_batch_index} {end_condition} RETURN s.content, s.batch_i"
            results = await self.aquery(query)
            # Format the results to include both content and batch_i
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {"text": result["s.content"], "batch_i": result["s.batch_i"]}
                )
            return formatted_results
        else:
            return []

    async def aget_max_batch_index(self, uuid: str = ""):
        if uuid:
            query = f"MATCH (s:Summary) WHERE s.uuid = '{uuid}' RETURN max(s.batch_i)"
        else:
            query = "MATCH (s:Summary) RETURN max(s.batch_i)"
        result = await self.aquery(query)
        return result[0]["max(s.batch_i)"]

    def add_summary(self, summary: str, metadata: dict):
        """Add a batch summary to the Neo4j database as a Summary node with metadata."""
        logger.debug(f"Adding summary {metadata['batch_i']} to Neo4j")
        query = "MERGE (s:Summary:__Community__ {content: $summary, batch_i: $batch_i, uuid: $uuid}) SET s += $metadata"
        params = {
            "summary": summary,
            "batch_i": metadata["batch_i"],
            "uuid": metadata["uuid"],
            "metadata": metadata,
        }
        self.query(query, params)

    def add_graph_documents_to_db(self, graph_documents: List[GraphDocument]):
        if graph_documents:
            with Metrics("GraphRAG/Neo4j/add_graph_documents", "green"):
                self.graph_db.add_graph_documents(
                    graph_documents, baseEntityLabel=True, include_source=False
                )

    def persist_chunk_data(
        self, batch_data: List[Dict], relationships: List[Dict], document_uuid: str
    ):
        with Metrics("GraphRAG/Neo4j/persist_chunk_data", "green"):
            query_create_chunks = """
                UNWIND $batch_data AS data
                MERGE (c:Chunk {id: data.id})
                SET c += data { .text, .camera_id, .position, .length, .uuid, .content_offset, .chunkIdx, .start_pts, .end_pts, .streamId, .file, .asset_dir, .pts_offset_ns, .start_time, .end_time }
                WITH c, data
                MATCH (d:Document {uuid: data.uuid, camera_id: data.camera_id})
                MERGE (c)-[:PART_OF]->(d)
            """
            self.graph_db.query(query_create_chunks, params={"batch_data": batch_data})

            first_chunk_rels = [
                rel for rel in relationships if rel["type"] == "FIRST_CHUNK"
            ]
            if first_chunk_rels:
                query_first_chunk = """
                    UNWIND $relationships AS relationship
                    MATCH (d:Document {uuid: $uuid})
                    MATCH (c:Chunk {id: relationship.chunk_id})
                    MERGE (d)-[:FIRST_CHUNK]->(c)
                """
                self.graph_db.query(
                    query_first_chunk,
                    params={"uuid": document_uuid, "relationships": first_chunk_rels},
                )

            next_chunk_rels = [
                rel for rel in relationships if rel["type"] == "NEXT_CHUNK"
            ]
            if next_chunk_rels:
                query_next_chunk = """
                    UNWIND $relationships AS relationship
                    MATCH (c:Chunk {id: relationship.current_chunk_id})
                    MATCH (pc:Chunk {id: relationship.previous_chunk_id})
                    MERGE (pc)-[:NEXT_CHUNK]->(c)
                """
                self.graph_db.query(
                    query_next_chunk, params={"relationships": next_chunk_rels}
                )

    def persist_summary_chunk_relationships(self, document_uuid: str):
        with Metrics("GraphRAG/Neo4j/persist_summary_relations", "green"):
            try:
                in_summary_query = """
                    MATCH (s:Summary)
                    WHERE s.linked_summary_chunks IS NOT NULL AND s.uuid = $uuid
                    UNWIND s.linked_summary_chunks AS chunk_idx
                    MATCH (c:Chunk {chunkIdx: chunk_idx, uuid: s.uuid})
                    MERGE (c)-[:IN_SUMMARY]->(s)
                """
                self.graph_db.query(in_summary_query, params={"uuid": document_uuid})
                logger.debug("Successfully merged IN_SUMMARY relationships.")

                summary_of_query = """
                    MATCH (d:Document {uuid: $uuid})
                    MATCH (s:Summary {uuid: $uuid})
                    MERGE (s)-[:SUMMARY_OF]->(d)
                """
                self.graph_db.query(summary_of_query, params={"uuid": document_uuid})
                logger.debug(
                    f"Successfully merged SUMMARY_OF relationships for document {document_uuid}."
                )

            except Exception as e:
                logger.warning(f"Could not merge IN_SUMMARY relationships: {e}")

    def persist_chunk_embeddings(self, data_for_embedding: List[Dict]):
        with Metrics("GraphRAG/Neo4j/persist_chunk_embeddings", "blue"):
            query_to_create_embedding = """
                UNWIND $data AS row
                MERGE (c:Chunk {id: row.chunkId})
                SET c.embedding = row.embedding
            """
            self.graph_db.query(
                query_to_create_embedding,
                params={"data": data_for_embedding},
            )

    def persist_chunk_entity_relationships(
        self, batch_data: List[Dict], document_uuid: str
    ):
        with Metrics("GraphRAG/Neo4j/merge_chunk_entity_rels", "yellow"):
            unwind_query = """
                UNWIND $batch_data AS data
                MATCH (c:Chunk {id: data.chunk_hash})
                CALL apoc.merge.node([data.node_type], {id: data.node_id}, {uuid: $uuid}) YIELD node AS n
                MERGE (c)-[:HAS_ENTITY]->(n)
            """
            self.graph_db.query(
                unwind_query, params={"batch_data": batch_data, "uuid": document_uuid}
            )

    def update_knn(self):
        with Metrics("GraphRAG/Neo4j/UpdateKNN", "blue"):
            try:
                index_info = self.graph_db.query(
                    """SHOW INDEXES YIELD name, type, labelsOrTypes
                       WHERE type = 'VECTOR' AND name = $index_name AND labelsOrTypes = ['Chunk']
                       RETURN count(*) > 0 AS indexExists""",
                    params={"index_name": CHUNK_VECTOR_INDEX_NAME},
                )
                index_exists = index_info and index_info[0]["indexExists"]
            except Exception as e:
                logger.error(
                    f"Failed to check for vector index '{CHUNK_VECTOR_INDEX_NAME}': {e}"
                )
                index_exists = False

            if not index_exists:
                logger.warning(
                    f"Vector index '{CHUNK_VECTOR_INDEX_NAME}' on Chunk(embedding) does not exist. Skipping KNN update."
                )
                return

            knn_min_score = float(os.environ.get("KNN_MIN_SCORE", 0.8))
            logger.info(
                f"Updating KNN graph using index '{CHUNK_VECTOR_INDEX_NAME}' with min score {knn_min_score}"
            )
            try:
                self.graph_db.query(
                    """MATCH (c:Chunk)
                        WHERE c.embedding IS NOT NULL AND count { (c)-[:SIMILAR]-() } < 5
                        CALL  db.index.vector.queryNodes('vector', 6, c.embedding) yield node, score
                        WHERE node <> c and score >= $score MERGE (c)-[rel:SIMILAR]-(node) SET rel.score = score
                    """,
                    {"score": float(knn_min_score)},
                )
                logger.info("KNN graph update query executed.")
            except Exception as e:
                logger.error(
                    f"Failed to update KNN graph: {e} {traceback.format_exc()}"
                )

    def fetch_subtitle_for_embedding(self) -> List[Dict[str, Any]]:
        with Metrics("GraphRAG/Neo4j/fetch_subtitle_for_embedding", "green"):
            query = """
                    MATCH (s:Subtitle)
                    WHERE s.embedding IS NULL AND s.text IS NOT NULL
                    RETURN elementId(s) AS elementId, s.text AS text
                    """
            try:
                result = self.graph_db.query(query)
                return [
                    {"elementId": record["elementId"], "text": record["text"]}
                    for record in result
                ]
            except Exception as e:
                logger.error(f"Failed to fetch subtitles for embedding: {e}")
                return []

    def fetch_entities_needing_embedding(self) -> List[Dict[str, Any]]:
        with Metrics("GraphRAG/Neo4j/fetch_entities_for_embedding", "green"):
            query = """
                    MATCH (e)
                    WHERE NOT (e:Chunk OR e:Document OR e:Summary) AND e.embedding IS NULL AND e.name IS NOT NULL
                    RETURN elementId(e) AS elementId, e.name + coalesce(' ' + e.description, '') AS text
                    """
            try:
                result = self.graph_db.query(query)
                return [
                    {"elementId": record["elementId"], "text": record["text"]}
                    for record in result
                ]
            except Exception as e:
                logger.error(f"Failed to fetch entities for embedding: {e}")
                return []

    def persist_subtitle_embeddings(self, rows_with_embeddings: List[Dict]):
        with Metrics("GraphRAG/Neo4j/persist_subtitle_embeddings", "yellow"):
            query = """
                UNWIND $rows AS row
                MATCH (s) WHERE elementId(s) = row.elementId
                CALL db.create.setNodeVectorProperty(s, "embedding", row.embedding)
                """
            try:
                self.graph_db.query(query, params={"rows": rows_with_embeddings})
            except Exception as e:
                logger.error(f"Failed to persist subtitle embeddings: {e}")
                raise

    def persist_entity_embeddings(self, rows_with_embeddings: List[Dict]):
        with Metrics("GraphRAG/Neo4j/persist_entity_embeddings", "yellow"):
            query = """
                UNWIND $rows AS row
                MATCH (e) WHERE elementId(e) = row.elementId
                CALL db.create.setNodeVectorProperty(e, "embedding", row.embedding)
                """
            try:
                self.graph_db.query(query, params={"rows": rows_with_embeddings})
            except Exception as e:
                if "Unknown function 'db.create.setNodeVectorProperty'" in str(e):
                    logger.warning(
                        "APOC function db.create.setNodeVectorProperty not found. Falling back to standard SET."
                    )
                    fallback_query = """
                         UNWIND $rows AS row
                         MATCH (e) WHERE elementId(e) = row.elementId
                         SET e.embedding = row.embedding
                         """
                    self.graph_db.query(
                        fallback_query, params={"rows": rows_with_embeddings}
                    )
                else:
                    logger.error(f"Failed to persist entity embeddings: {e}")
                    raise

    def fetch_summaries_needing_embedding(self) -> List[Dict[str, Any]]:
        with Metrics("GraphRAG/Neo4j/fetch_summaries_for_embedding", "green"):
            query = """
                    MATCH (s:Summary)
                    WHERE s.embedding IS NULL AND s.content IS NOT NULL
                    RETURN elementId(s) AS id, s.content AS content
                    """
            try:
                result = self.graph_db.query(query)
                return [
                    {"id": record["id"], "content": record["content"]}
                    for record in result
                ]
            except Exception as e:
                logger.error(f"Failed to fetch summaries for embedding: {e}")
                return []

    def persist_summary_embeddings(self, summaries_with_embeddings: List[Dict]):
        with Metrics("GraphRAG/Neo4j/persist_summary_embeddings", "yellow"):
            query = """
                UNWIND $summaries AS summary
                MATCH (s:Summary) WHERE elementId(s) = summary.id
                CALL db.create.setNodeVectorProperty(s, "embedding", summary.embedding)
                """
            try:
                self.graph_db.query(
                    query, params={"summaries": summaries_with_embeddings}
                )
            except Exception as e:
                logger.error(f"Failed to persist summary embeddings: {e}")
                raise

    def create_document_node(self, document_uuid: str, camera_id: str = ""):
        with Metrics("GraphRAG/Neo4j/create_document_node", "blue"):
            params = {"uuid": document_uuid, "camera_id": camera_id}
            query = "MERGE (d:Document {uuid: $uuid, camera_id: $camera_id}) SET d.uuid = $uuid, d.camera_id = $camera_id"
            self.graph_db.query(query, params=params)

    async def finalize_graph_creation(self):
        with Metrics("GraphRAG/Neo4j/finalize_graph_creation", "red"):
            logger.info("Finalizing Neo4j graph: Creating indexes.")
            await asyncio.to_thread(self.create_vector_fulltext_indexes)
            logger.info("Neo4j indexes created.")

    def create_vector_index(self, index_type):
        with Metrics("GraphExtraction/VectorIndex", "blue"):
            drop_query = ""
            query = ""

            if index_type == CHUNK_VECTOR_INDEX_NAME:
                drop_query = DROP_CHUNK_VECTOR_INDEX_QUERY
                query = CREATE_CHUNK_VECTOR_INDEX_QUERY.format(
                    index_name=CHUNK_VECTOR_INDEX_NAME,
                )
            else:
                logger.error(f"Invalid index type provided: {index_type}")
                return

            try:
                logger.info("Starting the process to create vector index.")
                try:
                    self.graph_db.query(drop_query)
                except Exception as e:
                    logger.error(f"Failed to drop index: {e}")
                    return

                try:
                    self.graph_db.query(query)
                except Exception as e:
                    logger.error(f"Failed to create {index_type} vector index: {e}")
                    return
            except Exception as e:
                logger.error(
                    "An error occurred while creating the vector index.", exc_info=True
                )
                logger.error(f"Error details: {str(e)}")

    def create_vector_fulltext_indexes(self):
        types = ["entities", "keyword", "subtitles"]
        logger.info("Starting the process of creating full-text indexes.")

        for index_type in types:
            try:
                logger.info(f"Creating a full-text index for type '{index_type}'.")
                self.create_fulltext(index_type)
                logger.info(
                    f"Full-text index for type '{index_type}' created successfully."
                )
            except Exception as e:
                logger.error(
                    f"Failed to create full-text index for type '{index_type}': {e}"
                )

        try:
            logger.info(
                f"Creating a vector index for type '{CHUNK_VECTOR_INDEX_NAME}'."
            )
            self.create_vector_index(CHUNK_VECTOR_INDEX_NAME)
            logger.info("Vector index for chunk created successfully.")
        except Exception as e:
            logger.error(
                f"Failed to create vector index for '{CHUNK_VECTOR_INDEX_NAME}': {e}"
            )

        logger.info("Full-text and vector index creation process completed.")

    def create_fulltext(self, index_type):
        with Metrics("GraphRAG/aprocess-doc/create-fulltext:", "red"):
            try:
                try:
                    if index_type == "entities":
                        drop_query = DROP_INDEX_QUERY.format(index_name=index_type)
                    elif index_type == "keyword":
                        drop_query = KEYWORD_SEARCH_INDEX_DROP_QUERY.format(
                            index_name=index_type
                        )
                    elif index_type == "subtitles":
                        drop_query = DROP_INDEX_QUERY.format(index_name=index_type)
                    self.graph_db.query(drop_query)
                except Exception as e:
                    logger.error(f"Failed to drop {index_type} index: {e}")
                    return
                try:
                    if index_type == "entities":
                        result = self.graph_db.query(LABELS_QUERY)
                        labels = [record["label"] for record in result]

                        for label in FILTER_LABELS:
                            if label in labels:
                                labels.remove(label)
                        if labels:
                            labels_str = ":" + "|".join(
                                [f"`{label}`" for label in labels]
                            )
                        else:
                            logger.info(
                                "Full text index is not created as labels are empty"
                            )
                            return
                except Exception as e:
                    logger.error(f"Failed to fetch labels: {e}")
                    return
                try:
                    if index_type == "entities":
                        fulltext_query = FULL_TEXT_QUERY.format(
                            index_name=index_type, labels_str=labels_str
                        )
                    elif index_type == "keyword":
                        fulltext_query = KEYWORD_SEARCH_FULL_TEXT_QUERY.format(
                            index_name=index_type
                        )
                    elif index_type == "subtitles":
                        fulltext_query = SUBTITLE_SEARCH_FULL_TEXT_QUERY.format(
                            index_name=index_type
                        )

                    self.graph_db.query(fulltext_query)
                except Exception as e:
                    logger.error(f"Failed to create full-text index {index_type}: {e}")
                    return
            except Exception as e:
                logger.error(f"An error occurred during the session: {e}")

    async def create_entity_embedding(self):
        rows = []
        logger.debug(f"Embedding parallel count: {self.embedding_parallel_count}")
        with Metrics("GraphExtraction/FetchEntEmbd", "green"):
            rows = self.fetch_entities_for_embedding()
        for i in range(0, len(rows), self.embedding_parallel_count):
            await self.update_embeddings(rows[i : i + self.embedding_parallel_count])

    async def create_subtitle_embedding(self):
        rows = []
        logger.debug(f"Embedding parallel count: {self.embedding_parallel_count}")
        with Metrics("GraphExtraction/FetchEntEmbd", "green"):
            rows = self.fetch_subtitle_for_embedding()
        for i in range(0, len(rows), self.embedding_parallel_count):
            await self.update_embeddings(rows[i : i + self.embedding_parallel_count])

    def fetch_entities_for_embedding(self):
        query = """
                MATCH (e)
                WHERE NOT (e:Chunk OR e:Document) AND e.embedding IS NULL AND e.id IS NOT NULL
                RETURN elementId(e) AS elementId, e.id + " " + coalesce(e.description, "") AS text
                """
        result = self.graph_db.query(query)
        return [
            {"elementId": record["elementId"], "text": record["text"]}
            for record in result
        ]

    async def update_embeddings(self, rows):
        with Metrics("GraphExtraction/UpdatEmbding", "yellow"):
            logger.info("update embedding for entities")

            async def semaphore_controlled_embed(text):
                async with self._embedding_semaphore:
                    return await self.embedding.aembed_query(text)

            tasks = [
                asyncio.create_task(semaphore_controlled_embed(row["text"]))
                for row in rows
            ]
            results = await asyncio.gather(*tasks)
            for i, row in enumerate(rows):
                row["embedding"] = results[i]
            query = """
            UNWIND $rows AS row
            MATCH (e) WHERE elementId(e) = row.elementId
            CALL db.create.setNodeVectorProperty(e, "embedding", row.embedding)
            """
            return self.graph_db.query(query, params={"rows": rows})

    def create_chunk_vector_index(self):
        try:
            vector_index = self.graph_db.query(
                "SHOW INDEXES YIELD * WHERE labelsOrTypes = ['Chunk'] and type = 'VECTOR' AND name = 'vector' return options"
            )

            if not vector_index:
                vector_store = Neo4jVector(
                    embedding=self.embedding,
                    graph=self.graph_db,
                    node_label="Chunk",
                    embedding_node_property="embedding",
                    index_name="vector",
                )
                vector_store.create_new_index()
                logger.info("Index created successfully.")
            else:
                logger.info("Index already exist,Skipping creation.")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(
                e
            ) or "An equivalent index already exists" in str(e):
                logger.info("Vector index already exists, skipping creation.")
            else:
                raise

    def reset(self, state: dict = {}):
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return
        if uuid and not erase_db:
            logger.debug(f"Resetting graph for UUID: {uuid}")
            self.query(QUERY_TO_DELETE_UUID_GRAPH, params={"uuid": uuid})
        else:
            logger.warning("Deleting all nodes and relationships.")
            self.query("MATCH (n) DETACH DELETE n")
            logger.debug("Dropping all indexes.")
            result = self.query(
                "SHOW INDEXES YIELD name, type WHERE type IN ['VECTOR', 'FULLTEXT'] RETURN 'DROP INDEX ' + name AS dropCommand;"
            )
            for record in result:
                self.query(record["dropCommand"])

    def create_neo4j_vector(
        self, multi_channel: bool, retrieval_query: str, index_name: str
    ):
        try:
            keyword_index = "keyword"
            node_label = "Chunk"
            embedding_node_property = "embedding"
            text_node_properties = ["text"]

            vector_db = Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=self.graph_db,
                search_type="hybrid" if multi_channel else "vector",
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
                keyword_index_name=keyword_index,
            )
            logger.info(
                f"Successfully retrieved Neo4jVector Fulltext index '{index_name}' and keyword index '{keyword_index}'"
            )
        except Exception as e:
            logger.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
            raise
        return vector_db

    def get_neo4j_retriever(
        self,
        retriever_type: str = "chunk",
        top_k: int = None,
        multi_channel: bool = False,
        uuid: str = None,
    ):
        """Get a Neo4j retriever based on the specified type.

        Args:
            retriever_type: Type of retriever to create ("chunk", "entity", or "subtitle")
            top_k: Number of documents to retrieve
            multi_channel: Whether to retrieve documents for multiple channels
            uuid: UUID of the user

        Returns:
            Neo4jVector retriever instance

        Raises:
            ValueError: If retriever_type is not supported
            Exception: If there's an error creating the retriever
        """
        with Metrics("GraphRetrieval/Neo4jRetriever", "blue"):
            # Configuration for different retriever types
            retriever_configs = {
                "chunk": {
                    "index_name": "vector",
                    "retrieval_query": VECTOR_GRAPH_SEARCH_QUERY,
                    "vector_creator": self.create_neo4j_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
                },
                "planner_chunk": {
                    "index_name": "vector",
                    "retrieval_query": CHUNK_SEARCH_QUERY,
                    "vector_creator": self.create_neo4j_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
                },
                "entity": {
                    "index_name": "entity_vector",
                    "retrieval_query": ENTITY_SEARCH_QUERY_FORMATTED,
                    "vector_creator": self.create_neo4j_entity_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": 0.6,
                },
                "planner_entity": {
                    "index_name": "entity_vector",
                    "retrieval_query": PLANNER_ENTITY_SEARCH_QUERY,
                    "vector_creator": self.create_neo4j_entity_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": 0.6,
                },
                "subtitle": {
                    "index_name": "subtitle_vector",
                    "retrieval_query": SUBTITLE_SEARCH_QUERY_FORMATTED,
                    "vector_creator": self.create_neo4j_subtitle_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": 0.6,
                },
                "gnn_chunk": {
                    "index_name": "vector",
                    "retrieval_query": GNN_VECTOR_GRAPH_SEARCH_QUERY,
                    "vector_creator": self.create_neo4j_vector,
                    "default_k": VECTOR_SEARCH_TOP_K,
                    "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
                },
            }

            if retriever_type.lower() not in retriever_configs:
                logger.warning(
                    f"Unsupported retriever type: {retriever_type}. Defaulting to chunk search"
                )
                retriever_type = "chunk"

            config = retriever_configs[retriever_type.lower()]

            try:
                index_name = config["index_name"]
                retrieval_query = config["retrieval_query"]
                vector_db = config["vector_creator"](
                    multi_channel, retrieval_query, index_name
                )
                search_k = top_k or config["default_k"]
                # Build base search kwargs
                search_kwargs = {
                    "k": search_k,
                    "score_threshold": config["score_threshold"],
                }

                # Add UUID filtering for non-subtitle retrievers
                if retriever_type.lower() != "subtitle":
                    if not multi_channel:
                        search_kwargs.update(
                            {"filter": {"uuid": uuid}, "params": {"uuid": uuid}}
                        )
                    else:
                        search_kwargs.update({"params": {"uuid": None}})

                return vector_db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs=search_kwargs,
                )
            except Exception as e:
                logger.error(
                    f"Error retrieving Neo4jVector index {index_name} or creating retriever: {e}"
                )
                raise Exception(
                    f"An error occurred while retrieving the Neo4jVector index or creating the retriever. "
                    f"Please drop and create a new vector index '{index_name}': {e}"
                ) from e

    def create_neo4j_subtitle_vector(
        self, multi_channel: bool, retrieval_query: str, index_name: str
    ):
        try:
            node_label = "Subtitle"
            embedding_node_property = "embedding"
            text_node_properties = ["text"]
            vector_db = Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=self.graph_db,
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
            )
            logger.info(f"Successfully retrieved Neo4jVector Index '{index_name}'")
        except Exception as e:
            if "An equivalent index already exists" in str(
                e
            ) or "EquivalentSchemaRuleAlreadyExists" in str(e):
                logger.info(f"using existing index {index_name}")
                vector_db = Neo4jVector.from_existing_index(
                    embedding=self.embedding,
                    index_name=index_name,
                    graph=self.graph_db,
                )
                logger.info(f"Successfully retrieved Neo4jVector Index '{index_name}'")
            else:
                logger.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
                raise
        return vector_db

    def create_neo4j_entity_vector(
        self, multi_channel: bool, retrieval_query: str, index_name: str
    ):
        try:
            node_label = "__Entity__"
            embedding_node_property = "embedding"
            text_node_properties = ["name"]

            vector_db = Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=self.graph_db,
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
            )
            logger.info(f"Successfully retrieved Neo4jVector Index '{index_name}'")
        except Exception as e:
            if "An equivalent index already exists" in str(
                e
            ) or "EquivalentSchemaRuleAlreadyExists" in str(e):
                logger.info(f"using existing index {index_name}")
                vector_db = Neo4jVector.from_existing_index(
                    embedding=self.embedding,
                    index_name=index_name,
                    graph=self.graph_db,
                    text_node_property="name",
                )
                logger.info(f"Successfully retrieved Neo4jVector Index '{index_name}'")
            else:
                logger.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
                raise
        return vector_db

    def persist_subtitle_frames(self, subtitle_frames: list[dict]):
        with Metrics("GraphRAG/Neo4j/persist_subtitle_frames", "yellow"):
            query = """
                UNWIND $subtitle_frames AS subtitle_frame
                MERGE (s:Subtitle {start_time: subtitle_frame.start_time, end_time: subtitle_frame.end_time, text: subtitle_frame.text})
                SET s.start_time = subtitle_frame.start_time, s.end_time = subtitle_frame.end_time, s.text = subtitle_frame.text
            """
            self.graph_db.query(query, params={"subtitle_frames": subtitle_frames})

    def create_document_retriever(self, neo4j_retriever):
        with Metrics("GraphRetrieval/CreateDocRetChain", "blue"):
            try:
                logger.info("Starting to create document retriever chain")

                embeddings_dimension = int(
                    os.environ.get("CA_RAG_EMBEDDINGS_DIMENSION", 2048)
                )
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=embeddings_dimension,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", "\n-"],
                )
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.embedding,
                    similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
                )
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[splitter, embeddings_filter]
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=pipeline_compressor, base_retriever=neo4j_retriever
                )
                return compression_retriever
            except Exception as e:
                logger.error(
                    f"Error creating document retriever chain: {e}", exc_info=True
                )
                raise

    def format_docs(self, documents):
        """Format retrieved documents for downstream processing.

        Args:
            documents: List of retrieved documents

        Returns:
            tuple: (formatted_docs, sources, entities, node_details)
        """
        with Metrics("chat/format documents", "green"):
            sorted_documents = sorted(
                documents,
                key=lambda doc: doc.state.get("query_similarity_score", 0),
                reverse=True,
            )
            formatted_docs = list()
            sources = set()
            entities = dict()

            for doc in sorted_documents:
                try:
                    entities = (
                        doc.metadata["entities"]
                        if "entities" in doc.metadata.keys()
                        else entities
                    )

                    formatted_doc = (
                        f"Document start\nContent: {doc.page_content}\nDocument end\n"
                    )
                    formatted_docs.append(formatted_doc)

                except Exception as e:
                    logger.error(f"Error formatting document: {e}")

            result = {
                "sources": list(sources),
            }

            return "\n\n".join(formatted_docs), sources, entities, result

    def format_source_docs(self, docs: list[Document]):
        formatted_docs = []
        for doc in docs:
            formatted_docs.append(
                {
                    "metadata": {  # TODO: add other metadata
                        "asset_dirs": doc.metadata.get("asset_dirs", None),
                        "length": doc.metadata.get("length", None),
                    },
                    "page_content": doc.page_content,
                }
            )
        return formatted_docs

    def retrieve_documents(
        self,
        question: str,
        uuid: str,
        multi_channel: bool = False,
        top_k: int = 5,
        retriever: str = "chunk",
    ) -> Tuple[List[str], List[str]]:
        """Retrieve documents from graph database and format them.

        Args:
            question (str): Question to retrieve documents for
            uuid (str): UUID of the user
            multi_channel (bool, optional): Whether to retrieve documents for multiple channels
            top_k (int, optional): Number of documents to retrieve
            retriever (str, optional): Type of retriever to use ("chunk", "entity", or "subtitle")

        Returns:
            str: Formatted documents text

        Raises:
            RuntimeError: If there is an error retrieving documents
        """
        with Metrics(
            "GraphRetrieval/RetrieveDocuments",
            "blue",
            span_kind=Metrics.SPAN_KIND["RETRIEVER"],
        ) as tm:
            tm.input({"question": question})
            try:
                neo4j_retriever = self.get_neo4j_retriever(
                    retriever_type=retriever,
                    top_k=top_k,
                    multi_channel=multi_channel,
                    uuid=uuid,
                )
                doc_retriever = self.create_document_retriever(neo4j_retriever)
                question = remove_lucene_chars(question)
                docs = doc_retriever.invoke(question)
                formatted_docs, sources, entitydetails, sources_and_chunks = (
                    self.format_docs(docs)
                )

                result = {"sources": list(), "nodedetails": dict(), "entities": dict()}
                # node_details = {"entitydetails": list()}
                entities = {"entityids": list(), "relationshipids": list()}

                result["sources"] = sources_and_chunks["sources"]
                entities.update(entitydetails)

                source_docs = self.format_source_docs(docs)

                return formatted_docs, source_docs

            except Exception as e:
                logger.error(traceback.format_exc())
                error_message = f"Error retrieving documents: {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message) from e

    def retrieve_documents_for_gnn(
        self,
        question: str,
        uuid: str,
        multi_channel: bool = False,
        top_k: int = None,
        retriever="gnn_chunk",
    ) -> tuple[List[str], List[str], List[str], List[int], List[int], List[dict]]:
        """Retrieve documents from graph database and format them.

        Args:
            question (str): Question to retrieve documents for
            uuid (str): UUID of the user
            multi_channel (bool, optional): Whether to retrieve documents for multiple channels
            top_k (int, optional): Number of documents to retrieve

        Returns:
            list[str] :

        Raises:
            RuntimeError: If there is an error retrieving documents
        """
        with Metrics("Retrieve documents for GNN", "green"):
            try:
                neo_4j_retriever = self.get_neo4j_retriever(
                    GNN_TRAVERSAL_STRATEGY, top_k, multi_channel, uuid
                )
                doc_retriever = self.create_document_retriever(neo_4j_retriever)
                docs = doc_retriever.invoke(question)
                (
                    chunks_list,
                    text_entities_list,
                    text_relationship_list,
                    edge_index_from,
                    edge_index_to,
                ) = self.format_docs_for_gnn(docs)
                source_docs = self.format_source_docs(docs)

                payload_data = {
                    "nodes": text_entities_list,
                    "edges": text_relationship_list,
                    "edge_indices": [edge_index_from, edge_index_to],
                    "description": chunks_list,
                }
                return payload_data, source_docs
            except Exception as e:
                logger.error(traceback.format_exc())
                error_message = f"Error retrieving documents: {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)

    def format_docs_for_gnn(self, documents):
        """Format retrieved documents for downstream processing.

        Args:
            documents: List of retrieved documents

        Returns:
            tuple: (chunks_list, text_entities_list, text_relationship_list, edge_index_from, edge_index_to)
        """
        with Metrics("chat/format documents for GNN", "green"):
            prompt_token_cutoff = 28
            sorted_documents = sorted(
                documents,
                key=lambda doc: doc.state.get("query_similarity_score", 0),
                reverse=True,
            )
            documents = sorted_documents[:prompt_token_cutoff]
            text_entities_list = []  # nodes of the retrieved subgraph
            text_relationship_list = []  # edges of the retrieved subgraph
            edge_index_from = []  # list of source nodes of the above edges
            edge_index_to = []  # list of target nodes of the above edges
            chunks_list = []
            if documents and len(documents) > 0:
                doc0 = documents[0]
                sourceNodes = doc0.metadata.get("sourceNode", None)
                if not (sourceNodes and len(sourceNodes) > 0):
                    raise RuntimeError("Retrieved subgraph is invalid")
                edges = doc0.metadata.get("edge", None)
                if not (edges and len(edges) > 0):
                    raise RuntimeError("Retrieved subgraph is invalid")
                targetNodes = doc0.metadata.get("targetNode", None)
                if not (targetNodes and len(targetNodes) > 0):
                    raise RuntimeError("Retrieved subgraph is invalid")
                if len(sourceNodes) != len(targetNodes):
                    raise RuntimeError("Retrieved subgraph is invalid")
                if len(sourceNodes) != len(edges):
                    raise RuntimeError("Retrieved subgraph is invalid")
                N = len(sourceNodes)
                logger.debug(f"SOURCE NODES {sourceNodes}\n")
                logger.debug(f"EDGES {edges}\n")
                logger.debug(f"TARGET {targetNodes}\n")
                # Relationships are present in the form
                # sourceNode[i] edge[i] targetNode[i]
                # text_entities_list should contain all unique nodes
                text_relationship_list = edges
                for i in range(N):
                    if sourceNodes[i] not in text_entities_list:
                        text_entities_list.append(sourceNodes[i])
                    # Append the position of the source node to the list
                    edge_index_from.append(text_entities_list.index(sourceNodes[i]))

                    if targetNodes[i] not in text_entities_list:
                        text_entities_list.append(targetNodes[i])
                    # Append the position of the target node
                    edge_index_to.append(text_entities_list.index(targetNodes[i]))
                chunks_list = documents[0].metadata.get("chunkdetails", [])
            return (
                chunks_list,
                text_entities_list,
                text_relationship_list,
                edge_index_from,
                edge_index_to,
            )

    def as_retriever(self, search_kwargs: dict = None):
        """
        This method is used to create a retriever for the Neo4j database.
        It is used to retrieve documents from the Neo4j database.
        """
        if search_kwargs is None:
            search_kwargs = {}
        index_name = "image_qna_vec_index"

        try:
            keyword_index = "keyword"
            node_label = "Chunk"
            embedding_node_property = "embedding"
            text_node_properties = ["text"]

            vector_index = Neo4jVector.from_existing_graph(
                embedding=self.embedding,
                index_name=index_name,
                graph=self.graph_db,
                search_type="hybrid",
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
                keyword_index_name=keyword_index,
            )
            logger.info(
                f"Successfully retrieved Neo4jVector Fulltext index '{index_name}' and keyword index '{keyword_index}'"
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
            logger.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
            raise

    def get_duplicate_nodes_list(self, duplicate_score_value):
        query_duplicate_nodes = """
                MATCH (n:!Chunk&!Document) with n
                WHERE n.embedding is not null and n.name is not null // and size(toString(n.name)) > 3
                WITH n ORDER BY count {{ (n)--() }} DESC, size(toString(n.name)) DESC // updated
                WITH collect(n) as nodes
                UNWIND nodes as n
                WITH n, [other in nodes
                // only one pair, same labels e.g. Person with Person
                WHERE elementId(n) < elementId(other) and labels(n) = labels(other)
                // at least embedding similarity of X
                AND
                vector.similarity.cosine(other.embedding, n.embedding) > $duplicate_score_value
                ] as similar
                WHERE size(similar) > 0
                // remove duplicate subsets
                with collect([n]+similar) as all
                CALL(all) {{ with all
                    unwind all as nodes
                    with nodes, all
                    // skip current entry if it's smaller and a subset of any other entry
                    where none(other in all where other <> nodes and size(other) > size(nodes) and size(apoc.coll.subtract(nodes, other))=0)
                    return head(nodes) as n, tail(nodes) as similar
                }}
                OPTIONAL MATCH (doc:Document)<-[:PART_OF]-(c:Chunk)-[:HAS_ENTITY]->(n)
                {return_statement}
                """
        return_query_duplicate_nodes = """
                RETURN n {.*, embedding:null, elementId:elementId(n), labels:labels(n)} as e,
                [s in similar | s {.id, .name, .camera_id, .description, labels:labels(s), elementId: elementId(s)}] as similar,
                collect(distinct doc.uuid) as documents, count(distinct c) as chunkConnections
                ORDER BY e.name ASC
                LIMIT 100
                """
        param = {"duplicate_score_value": duplicate_score_value}
        logger.debug(f"Duplicate score value: {duplicate_score_value}")
        nodes_list = self.graph_db.query(
            query_duplicate_nodes.format(return_statement=return_query_duplicate_nodes),
            params=param,
        )
        return nodes_list

    def merge_duplicate_nodes(self, duplicate_score_value: float):
        with Metrics("GraphRAG/Neo4j/merge_duplicate_nodes", "yellow"):
            duplicate_nodes_list = self.get_duplicate_nodes_list(duplicate_score_value)
            logger.debug(f"Nodes list to merge: {len(duplicate_nodes_list)} groups")
            logger.debug(f"Nodes list to merge: {duplicate_nodes_list}")
            # Transform the data structure to match what the Cypher query expects
            rows_to_merge = []
            for node_group in duplicate_nodes_list:
                # Extract the main node and its similar nodes
                main_node = node_group.get("e", {})
                similar_nodes = node_group.get("similar", [])

                if not main_node or not similar_nodes or "elementId" not in main_node:
                    logger.warning(f"Skipping invalid node group: {node_group}")
                    continue

                # Extract element IDs for the query
                first_element_id = main_node["elementId"]
                similar_element_ids = [
                    n["elementId"] for n in similar_nodes if "elementId" in n
                ]

                if not similar_element_ids:
                    logger.info(
                        f"No similar nodes to merge for {main_node.get('name', 'unknown')}"
                    )
                    continue

                # Log what we're merging for debugging
                logger.info(
                    f"Merging node {main_node.get('name', 'unknown')} with {len(similar_element_ids)} similar nodes"
                )
                logger.info(
                    f"Similar nodes: {[(n.get('name', 'unknown'), n.get('description', 'unknown')) for n in similar_nodes]}"
                )

                # Add to the rows to process
                rows_to_merge.append(
                    {
                        "firstElementId": first_element_id,
                        "similarElementIds": similar_element_ids,
                    }
                )

            if not rows_to_merge:
                logger.warning("No valid node groups to merge")
                return {"totalMerged": 0}

            # Execute the merge query with the transformed data
            query = """
            UNWIND $rows AS row
            CALL (row){ with row
                MATCH (first) WHERE elementId(first) = row.firstElementId
                MATCH (rest) WHERE elementId(rest) IN row.similarElementIds
                WITH first, collect(rest) as rest
                WITH [first] + rest as nodes
                CALL apoc.refactor.mergeNodes(nodes,
                {properties:{`.*`: "discard", camera_id: "combine"}, mergeRels:true, produceSelfRel:false, preserveExistingSelfRels:false, singleElementAsArray:true})
                YIELD node
                RETURN size(nodes) as mergedCount
            }
            RETURN sum(mergedCount) as totalMerged
            """
            param = {"rows": rows_to_merge}
            result = self.graph_db.query(query, params=param)

            # Log the result
            total_merged = result[0]["totalMerged"] if result else 0
            logger.info(f"Successfully merged {total_merged} nodes")

        return result

    def get_num_cameras(self) -> int:
        query = """
        MATCH (d:Document)
        RETURN COUNT(DISTINCT d.camera_id) as num_cameras
        """
        result = self.graph_db.query(query)
        return result[0]["num_cameras"] if result else 0

    def get_video_length(self):
        query = """
                MATCH (n:Chunk)
                WHERE NOT (n)-[:NEXT_CHUNK]->()
                WITH collect({key: n.camera_id, value: n.end_time}) AS pairs
                RETURN reduce(
                    acc = {}, entry IN pairs |
                    apoc.map.setKey(acc, entry.key, entry.value)
                ) AS camera_endtime_map"""
        result = self.graph_db.query(query)
        return result[0]["camera_endtime_map"] if result else 0

    def get_chunk_size(self) -> Dict[str, float]:
        query = """
        MATCH (c:Chunk)
        RETURN c.camera_id as camera_id, MAX(c.end_time - c.start_time) as maxChunkSize
        ORDER BY c.camera_id
        """
        result = self.graph_db.query(query)

        # Transform list of records into dictionary format {"camera_id": chunk_size}
        if result:
            return {record["camera_id"]: record["maxChunkSize"] for record in result}
        return {}

    def get_chunk_camera_id(self, chunk_id):
        query = """
        MATCH (c:Chunk)
        WHERE id(c) = toInteger($chunk_id)
        RETURN c.camera_id as camera_id
        """
        result = self.query(query, params={"chunk_id": chunk_id})
        return result[0]["camera_id"] if result else ""
