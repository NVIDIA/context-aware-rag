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

import os
import asyncio
import hashlib
import time
from typing import List

from langchain_community.graphs.graph_document import GraphDocument

from langchain_text_splitters import TokenTextSplitter
from langchain.docstore.document import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector

from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.utils.ctx_rag_batcher import Batch
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    CHUNK_VECTOR_INDEX_NAME,
    DROP_CHUNK_VECTOR_INDEX_QUERY,
    CREATE_CHUNK_VECTOR_INDEX_QUERY,
    DROP_INDEX_QUERY,
    HYBRID_SEARCH_INDEX_DROP_QUERY,
    FULL_TEXT_QUERY,
    LABELS_QUERY,
    HYBRID_SEARCH_FULL_TEXT_QUERY,
    FILTER_LABELS,
)
from vss_ctx_rag.utils.globals import DEFAULT_EMBEDDING_PARALLEL_COUNT


class GraphExtraction:
    """Handles extraction and processing of graph-based knowledge representations.

    This class manages the extraction of structured knowledge from text into a graph format,
    handling document processing, entity extraction, and graph construction.
    """

    def __init__(
        self,
        batcher,
        uuid,
        llm,
        graph: Neo4jGraphDB,
        embedding_parallel_count: int = DEFAULT_EMBEDDING_PARALLEL_COUNT,
    ):
        self.graph_db = graph
        self.transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=[],
            node_properties=False,
            relationship_properties=False,
            ignore_tool_usage=True,
        )
        self.uuid = uuid
        self.batcher = batcher
        self.create_chunk_vector_index()
        self.cleaned_graph_documents_list = []
        self.previous_chunk_id = 0
        self.last_position = 0
        self.embedding_parallel_count = embedding_parallel_count

    def handle_backticks_nodes_relationship_id_type(
        self, graph_document_list: List[GraphDocument]
    ):
        with TimeMeasure("GraphRAG/aprocess-doc/graph-create/handle-backticks", "blue"):
            for graph_document in graph_document_list:
                # Clean node id and types
                cleaned_nodes = []
                for node in graph_document.nodes:
                    node.properties = {"uuid": self.uuid}
                    if node.type.strip() and node.id.strip():
                        node.type = node.type.replace("`", "")
                        cleaned_nodes.append(node)
                # Clean relationship id types and source/target node id and types
                cleaned_relationships = []
                for rel in graph_document.relationships:
                    if (
                        rel.type.strip()
                        and rel.source.id.strip()
                        and rel.source.type.strip()
                        and rel.target.id.strip()
                        and rel.target.type.strip()
                    ):
                        rel.type = rel.type.replace("`", "")
                        rel.source.type = rel.source.type.replace("`", "")
                        rel.target.type = rel.target.type.replace("`", "")
                        cleaned_relationships.append(rel)
                graph_document.relationships = cleaned_relationships
                graph_document.nodes = cleaned_nodes
            return graph_document_list

    def merge_relationship_between_chunk_and_entites(self):
        with TimeMeasure(
            "GraphRAG/aprocess-doc/graph-create/merge-relationships", "yellow"
        ):
            batch_data = []
            logger.debug("Create HAS_ENTITY relationship between chunks and entities")
            for graph_doc in self.cleaned_graph_documents_list:
                for node in graph_doc.nodes:
                    query_data = {
                        "hash": graph_doc.source.metadata["hash"],
                        "node_type": node.type,
                        "node_id": node.id,
                    }
                    batch_data.append(query_data)

            if batch_data:
                unwind_query = """
                            UNWIND $batch_data AS data
                            MATCH (c:Chunk {id: data.hash})
                            CALL apoc.merge.node([data.node_type], {id: data.node_id}) YIELD node AS n
                            MERGE (c)-[:HAS_ENTITY]->(n)
                        """
                self.graph_db.graph_db.query(
                    unwind_query, params={"batch_data": batch_data}
                )

    async def update_embedding_chunks(self, chunkId_chunkDoc_list):
        with TimeMeasure(
            "GraphRAG/aprocess-doc/graph-create/update-embedding-chunks", "blue"
        ):
            data_for_query = []
            logger.info("update embedding and vector index for chunks")
            tasks = [
                asyncio.create_task(
                    self.graph_db.embeddings.aembed_query(
                        row["chunk_doc"].source.page_content
                    )
                )
                for row in chunkId_chunkDoc_list
            ]
            results = await asyncio.gather(*tasks)

            for i, row in enumerate(chunkId_chunkDoc_list):
                data_for_query.append(
                    {"chunkId": row["chunk_id"], "embeddings": results[i]}
                )

            query_to_create_embedding = """
                UNWIND $data AS row
                MATCH (d:Document {uuid: $uuid})
                MERGE (c:Chunk {id: row.chunkId})
                SET c.embedding = row.embeddings
                MERGE (c)-[:PART_OF]->(d)
            """
            self.graph_db.graph_db.query(
                query_to_create_embedding,
                params={"uuid": self.uuid, "data": data_for_query},
            )

    def create_relation_between_chunks(self) -> list:
        logger.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
        with TimeMeasure("GraphRAG/aprocess-doc/graph-create/create-relation", "green"):
            self.cleaned_graph_documents_list = sorted(
                self.cleaned_graph_documents_list,
                key=lambda doc: doc.source.metadata.get("chunkIdx", 0),
            )

            current_chunk_id = self.previous_chunk_id
            lst_chunks_including_hash = []
            batch_data = []
            relationships = []
            offset = 0
            for i, chunk in enumerate(self.cleaned_graph_documents_list):
                value_for_hash = chunk.source.page_content + self.uuid
                page_content_sha1 = hashlib.sha1(value_for_hash.encode())
                self.previous_chunk_id = current_chunk_id
                current_chunk_id = page_content_sha1.hexdigest()
                self.last_position = self.last_position + 1
                if i > 0:
                    offset += len(
                        self.cleaned_graph_documents_list[i - 1].source.page_content
                    )
                if i == 0 and chunk.source.metadata.get("chunkIdx", 0) == 0:
                    firstChunk = True
                else:
                    firstChunk = False
                metadata = {
                    "position": self.last_position,
                    "length": len(chunk.source.page_content),
                    "content_offset": offset,
                    "hash": current_chunk_id,
                    "chunkIdx": chunk.source.metadata["chunkIdx"],
                }
                self.cleaned_graph_documents_list[i].source.metadata.update(metadata)
                chunk_document = Document(
                    page_content=chunk.source.page_content, metadata=metadata
                )

                chunk_data = {
                    "id": current_chunk_id,
                    "pg_content": chunk_document.page_content,
                    "position": self.last_position,
                    "length": chunk_document.metadata["length"],
                    "uuid": self.uuid,
                    "previous_id": self.previous_chunk_id,
                    "content_offset": offset,
                    "chunkIdx": chunk_document.metadata["chunkIdx"],
                }

                if (
                    "start_ntp_float" in chunk.source.metadata
                    and "end_ntp_float" in chunk.source.metadata
                ):
                    chunk_data["start_time"] = chunk.source.metadata["start_ntp_float"]
                    chunk_data["end_time"] = chunk.source.metadata["end_ntp_float"]

                batch_data.append(chunk_data)

                lst_chunks_including_hash.append(
                    {"chunk_id": current_chunk_id, "chunk_doc": chunk}
                )

                # create relationships between chunks
                if firstChunk:
                    relationships.append(
                        {"type": "FIRST_CHUNK", "chunk_id": current_chunk_id}
                    )
                else:
                    relationships.append(
                        {
                            "type": "NEXT_CHUNK",
                            "previous_chunk_id": self.previous_chunk_id,  # ID of previous chunk
                            "current_chunk_id": current_chunk_id,
                        }
                    )
            self.previous_chunk_id = current_chunk_id

            query_to_create_chunk_and_PART_OF_relation = """
                UNWIND $batch_data AS data
                MERGE (c:Chunk {id: data.id})
                SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.uuid=data.uuid, c.content_offset=data.content_offset
                WITH data, c
                SET c.start_time = CASE WHEN data.start_time IS NOT NULL THEN data.start_time END,
                    c.end_time = CASE WHEN data.end_time IS NOT NULL THEN data.end_time END,
                    c.chunkIdx = data.chunkIdx
                WITH data, c
                MATCH (d:Document {uuid: data.uuid})
                MERGE (c)-[:PART_OF]->(d)
            """
            self.graph_db.graph_db.query(
                query_to_create_chunk_and_PART_OF_relation,
                params={"batch_data": batch_data},
            )

            query_to_create_FIRST_relation = """
                UNWIND $relationships AS relationship
                MATCH (d:Document {uuid: $uuid})
                MATCH (c:Chunk {id: relationship.chunk_id})
                FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                        MERGE (d)-[:FIRST_CHUNK]->(c))
                """
            self.graph_db.graph_db.query(
                query_to_create_FIRST_relation,
                params={"uuid": self.uuid, "relationships": relationships},
            )

            query_to_create_NEXT_CHUNK_relation = """
                UNWIND $relationships AS relationship
                MATCH (c:Chunk {id: relationship.current_chunk_id})
                WITH c, relationship
                MATCH (pc:Chunk {id: relationship.previous_chunk_id})
                FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                        MERGE (c)<-[:NEXT_CHUNK]-(pc))
                """
            self.graph_db.graph_db.query(
                query_to_create_NEXT_CHUNK_relation,
                params={"relationships": relationships},
            )

            return lst_chunks_including_hash

    def get_combined_chunks(self, chunkId_chunkDoc_list):
        with TimeMeasure("GraphRAG/aprocess-doc/graph-create/combine-chunks", "yellow"):
            text_splitter = TokenTextSplitter(chunk_size=300, chunk_overlap=10)
            return text_splitter.split_documents(chunkId_chunkDoc_list)

    def update_KNN_graph(self):
        """
        Update the graph node with SIMILAR relationship where embedding scrore match
        """
        with TimeMeasure("GraphExtraction/UpdateKNN", "blue"):
            index = self.graph_db.graph_db.query(
                """show indexes yield * where type = 'VECTOR' and name = 'vector'"""
            )
            # logger.info(f'show index vector: {index}')
            knn_min_score = os.environ.get("KNN_MIN_SCORE", 0.8)
            if len(index) > 0:
                logger.info("update KNN graph")
                self.graph_db.graph_db.query(
                    """
                    MATCH (c:Chunk)
                        WHERE c.embedding IS NOT NULL AND count { (c)-[:SIMILAR]-() } < 5
                        CALL db.index.vector.queryNodes('vector', 6, c.embedding) yield node, score
                        WHERE node <> c and score >= $score MERGE (c)-[rel:SIMILAR]-(node) SET rel.score = score
                    """,
                    {"score": float(knn_min_score)},
                )
            else:
                logger.info("Vector index does not exist, So KNN graph not update")

    def create_vector_index(self, index_type):
        with TimeMeasure("GraphExtraction/VectorIndex", "blue"):
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
                    start_step = time.time()
                    self.graph_db.graph_db.query(drop_query)
                    logger.info(
                        f"Dropped existing index (if any) in {time.time() - start_step:.2f} seconds."
                    )
                except Exception as e:
                    logger.error(f"Failed to drop index: {e}")
                    return

                try:
                    start_step = time.time()
                    self.graph_db.graph_db.query(query)
                    logger.info(
                        f"Created vector index in {time.time() - start_step:.2f} seconds."
                    )
                except Exception as e:
                    logger.error(f"Failed to create vector index: {e}")
                    return
            except Exception as e:
                logger.error(
                    "An error occurred while creating the vector index.", exc_info=True
                )
                logger.error(f"Error details: {str(e)}")

    def create_vector_fulltext_indexes(self):
        types = ["entities", "hybrid"]
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

    def create_fulltext(self, type):
        with TimeMeasure("GraphRAG/aprocess-doc/create-fulltext:", "red"):
            try:
                try:
                    start_step = time.time()
                    if type == "entities":
                        drop_query = DROP_INDEX_QUERY
                    elif type == "hybrid":
                        drop_query = HYBRID_SEARCH_INDEX_DROP_QUERY

                    self.graph_db.graph_db.query(drop_query)
                    logger.info(
                        f"Dropped existing index (if any) in {time.time() - start_step:.2f} seconds."
                    )
                except Exception as e:
                    logger.error(f"Failed to drop index: {e}")
                    return
                try:
                    if type == "entities":
                        start_step = time.time()
                        result = self.graph_db.graph_db.query(LABELS_QUERY)
                        labels = [record["label"] for record in result]

                        for label in FILTER_LABELS:
                            if label in labels:
                                labels.remove(label)
                        if labels:
                            labels_str = ":" + "|".join(
                                [f"`{label}`" for label in labels]
                            )
                            logger.info(
                                f"Fetched labels in {time.time() - start_step:.2f} seconds."
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
                    start_step = time.time()
                    if type == "entities":
                        fulltext_query = FULL_TEXT_QUERY.format(labels_str=labels_str)
                    elif type == "hybrid":
                        fulltext_query = HYBRID_SEARCH_FULL_TEXT_QUERY

                    self.graph_db.graph_db.query(fulltext_query)
                    logger.info(
                        f"Created full-text index in {time.time() - start_step:.2f} seconds."
                    )
                except Exception as e:
                    logger.error(f"Failed to create full-text index: {e}")
                    return
            except Exception as e:
                logger.error(f"An error occurred during the session: {e}")

    async def create_entity_embedding(self):
        rows = []
        logger.debug(f"Embedding parallel count: {self.embedding_parallel_count}")
        with TimeMeasure("GraphExtraction/FetchEntEmbd", "green"):
            rows = self.fetch_entities_for_embedding()
        for i in range(0, len(rows), self.embedding_parallel_count):
            await self.update_embeddings(rows[i : i + self.embedding_parallel_count])

    def fetch_entities_for_embedding(self):
        query = """
                MATCH (e)
                WHERE NOT (e:Chunk OR e:Document) AND e.embedding IS NULL AND e.id IS NOT NULL
                RETURN elementId(e) AS elementId, e.id + " " + coalesce(e.description, "") AS text
                """
        result = self.graph_db.graph_db.query(query)
        return [
            {"elementId": record["elementId"], "text": record["text"]}
            for record in result
        ]

    async def update_embeddings(self, rows):
        with TimeMeasure("GraphExtraction/UpdatEmbding", "yellow"):
            logger.info("update embedding for entities")
            tasks = [
                asyncio.create_task(self.graph_db.embeddings.aembed_query(row["text"]))
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
            return self.graph_db.graph_db.query(query, params={"rows": rows})

    def create_chunk_vector_index(self):
        try:
            vector_index = self.graph_db.graph_db.query(
                "SHOW INDEXES YIELD * WHERE labelsOrTypes = ['Chunk'] and type = 'VECTOR' AND name = 'vector' return options"
            )

            if not vector_index:
                vector_store = Neo4jVector(
                    embedding=self.graph_db.embeddings,
                    graph=self.graph_db.graph_db,
                    node_label="Chunk",
                    embedding_node_property="embedding",
                    index_name="vector",
                )
                vector_store.create_new_index()
                logger.info("Index created successfully.")
            else:
                logger.info("Index already exist,Skipping creation.")
        except Exception as e:
            if "EquivalentSchemaRuleAlreadyExists" in str(e):
                logger.info("Vector index already exists, skipping creation.")
            else:
                raise

    async def apost_process(self):
        with TimeMeasure("GraphRAG/aprocess-doc/graph-create/postprocessing", "green"):
            logger.debug("Post process GRAG")

            # Setting the document node
            params = {}
            query = "MERGE(d:Document {uuid :$props.uuid}) SET d += $props"
            params["uuid"] = self.uuid
            param = {"props": params}
            self.graph_db.graph_db.query(query, param)

            chunkId_chunkDoc_list = self.create_relation_between_chunks()
            await self.update_embedding_chunks(chunkId_chunkDoc_list)
            self.merge_relationship_between_chunk_and_entites()
            await asyncio.to_thread(self.update_KNN_graph)
            await asyncio.to_thread(self.create_vector_fulltext_indexes)
            await self.create_entity_embedding()
            self.cleaned_graph_documents_list.clear()
            logger.info("Graph created")

    async def acreate_graph(self, batch: Batch):
        with TimeMeasure("GraphRAG/aprocess-doc/graph-create:", "yellow"):
            docs = [
                Document(page_content=doc, metadata=metadata)
                for doc, _, metadata in batch.as_list()
            ]
            combined_chunk_document_list = self.get_combined_chunks(docs)

            with TimeMeasure("GraphRAG/aprocess-doc/graph-create/convert", "blue"):
                graph_documents = await self.transformer.aconvert_to_graph_documents(
                    combined_chunk_document_list
                )
            cleaned_graph_documents = self.handle_backticks_nodes_relationship_id_type(
                graph_documents
            )
            self.cleaned_graph_documents_list.extend(cleaned_graph_documents)
            with TimeMeasure(
                "GraphRAG/aprocess-doc/graph-create/add-graph-documents", "green"
            ):
                self.graph_db.graph_db.add_graph_documents(
                    cleaned_graph_documents, baseEntityLabel=True
                )

    def reset(self):
        """Reset the graph extraction state.

        Clears all temporary data structures and resets position tracking
        for a fresh extraction process.
        """
        self.cleaned_graph_documents_list.clear()
        self.previous_chunk_id = 0
        self.last_position = 0
