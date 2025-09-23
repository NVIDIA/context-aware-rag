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
from typing import Any, Dict, List

import cupy as cp
import networkx as nx
import numpy as np
from cuml.neighbors import NearestNeighbors
from langchain_community.graphs.graph_document import GraphDocument

from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger


class NetworkXGraphDB(GraphStorageTool):
    """
    StorageTool for the NetworkX graph.
    """

    def __init__(
        self,
        name="networkx_db",
        tools=None,
        config=None,
    ) -> None:
        """
        Initialize the NetworkXGraphDB class.

        Args:
            collection_name: Name of the collection.
            name: Name of the graph.

        Instance variables:
            self.collection_name: Name of the collection.
            self.graph: NetworkX graph.

        Returns:
            None
        """
        super().__init__(name, config, tools)
        self.config = config
        self.embedding = self.get_tool("embedding")
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        try:
            self.collection_name = config.params.collection_name
            self.graph = nx.MultiDiGraph(name=self.collection_name)
            logger.info(
                f"Initialized nx-cugraph-cu12 Graph with collection name {self.collection_name}"
            )
        except Exception as e:
            logger.error("Failed to initialize nx-cugraph-cu12 Graph: %s", e)
            raise

    def upsert_node(self, entity_id: str, data: dict) -> None:
        """
        Upsert a node into the graph.

        Args:
            entity_id: ID of the entity.
            data: Data to update the node with.

        Returns:
            None
        """
        if self.graph.has_node(entity_id):
            self.graph.nodes[entity_id].update(data)
        else:
            self.graph.add_node(entity_id, **data)

    def upsert_edge(self, source: str, target: str, data: dict) -> None:
        """
        Upsert an edge into the graph.

        Args:
            source: Source node.
            target: Target node.
            data: Data to update the edge with.

        Returns:
            None
        """
        self.graph.add_edge(source, target, **data)

    def has_node(self, node_id: str) -> bool:
        """
        Check if a node exists in the graph.

        Args:
            node_id: ID of the node.

        Returns:
            True if the node exists, False otherwise.
        """
        return self.graph.has_node(node_id)

    def has_edge(self, source: str, target: str) -> bool:
        """
        Check if an edge exists in the graph.

        Args:
            source: Source node.
            target: Target node.

        Returns:
            True if the edge exists, False otherwise.
        """
        return self.graph.has_edge(source, target)

    def get_nodes(self, data: bool = True) -> dict:
        """
        Get the nodes of graph with/without data

        Args:
            None

        Returns:
            Data of the node.
        """
        return self.graph.nodes(data=data)

    def get_node_by_id(self, node_id: str) -> dict:
        """
        Get the node of graph with data by id

        Args:
            node_id: ID of the node.

        Returns:
            Data of the node.
        """
        return self.graph.nodes[node_id]

    def reset(self, state: dict = {}):
        """
        Clears the graph or selectively removes nodes and edges associated with a specific UUID.

        Args:
            uuid: If provided, only nodes with this UUID attribute will be removed.
                  If empty or None, the entire graph will be cleared.
        """
        if os.getenv("VSS_CTX_RAG_ENABLE_RET", "False").lower() in ["true", "1"]:
            return
        uuid = state.get("uuid", "")
        erase_db = state.get("erase_db", False)
        if not uuid and not erase_db:
            return
        if uuid and not erase_db:
            nodes_to_remove = [
                node
                for node, data in self.graph.nodes(data=True)
                if data.get("uuid") == uuid
            ]
            self.graph.remove_nodes_from(nodes_to_remove)
            logger.info(f"Removed {len(nodes_to_remove)} nodes with uuid '{uuid}'.")
        elif erase_db:
            self.graph.clear()
            logger.info("Cleared the entire NetworkX graph.")

    def add_graph_documents(self, graph_documents: List[GraphDocument]) -> None:
        """
        Add or update graph documents into the NetworkX graph.

        Each GraphDocument is expected to contain:
            - nodes: A list of nodes with an 'id' attribute and a 'properties' dict.
            - relationships: A list of relationships with a 'type', and source and target node references.

        Args:
            graph_documents: List of GraphDocument objects.
            baseEntityLabel: If True, the base entity label will be added to the nodes.
        Returns:
            None
        """
        for doc in graph_documents:
            for node in doc.nodes:
                node.properties.update(
                    {"id": node.id, "type": node.type, "label": "__Entity__"}
                )
                self.upsert_node("Entity/" + node.id, node.properties)
            # Process relationships (edges) in the document.
            for rel in doc.relationships:
                source_id = rel.source.id
                target_id = rel.target.id

                edge_attributes = {"type": rel.type}
                if hasattr(rel, "properties"):
                    edge_attributes.update(rel.properties)
                self.upsert_edge(
                    "Entity/" + source_id, "Entity/" + target_id, edge_attributes
                )

        logger.info(
            f"Added {len(graph_documents)} entities and relationships to the NetworkX graph."
        )

    def add_summary(self, summary: str, metadata: dict):
        """Add a batch summary as a node in the internal NetworkX graph with metadata."""
        try:
            logger.debug(f"Adding summary {metadata['batch_i']} to ArangoDB")
            metadata["type"] = "Community"
            metadata["text"] = summary
            metadata["label"] = "__Community__"
            if "chunkIdx" in metadata:
                summary_id = f"Community/summary_{metadata['batch_i']}"
            else:
                logger.error(f"No chunkIdx in metadata: {metadata}")

            self.upsert_node(summary_id, metadata)
            logger.info("Summary node added to internal NetworkX graph.")
        except Exception as e:
            logger.error("Failed to add summary node to internal NetworkX graph: %s", e)
            raise e

    def add_graph_documents_to_db(self, graph_documents: List[Dict]):
        """Persists initial nodes/relationships using NetworkX add_graph_documents."""
        if graph_documents:
            with Metrics("NXGraphRAG/add_graph_documents", "green"):
                self.add_graph_documents(graph_documents)

    def persist_chunk_data(
        self, batch_data: List[Dict], relationships: List[Dict], uuid: str
    ):
        """Persists chunk nodes and relationships using NetworkX operations."""
        with Metrics("NXGraphRAG/persist_chunk_data", "green"):
            for data in batch_data:
                node_id = data["id"]
                node_attrs = {
                    "text": data["text"],
                    "type": "Chunk",
                    **{k: v for k, v in data.items() if k not in ["id", "text"]},
                }
                self.upsert_node("Chunk/" + node_id, node_attrs)
                self.upsert_edge(
                    "Chunk/" + node_id, "Document/" + uuid, {"type": "PART_OF"}
                )

            for rel in relationships:
                if rel["type"] == "FIRST_CHUNK":
                    self.upsert_edge(
                        "Document/" + uuid,
                        "Chunk/" + rel["chunk_id"],
                        {"type": "FIRST_CHUNK"},
                    )
                elif rel["type"] == "NEXT_CHUNK":
                    self.upsert_edge(
                        "Chunk/" + rel["previous_chunk_id"],
                        "Chunk/" + rel["current_chunk_id"],
                        {"type": "NEXT_CHUNK"},
                    )

    def persist_summary_chunk_relationships(self, uuid: str):
        """Persists IN_SUMMARY relationships using NetworkX."""
        with Metrics("NXGraphRAG/persist_summary_relations", "green"):
            summary_nodes = [
                (node_id, data)
                for node_id, data in self.get_nodes()
                if data.get("type", "").lower() == "community"
                and "chunkIdx" in data
                and data.get("uuid", "") == uuid
            ]

            summary_node_map = {}
            for summary_id, summary_data in summary_nodes:
                chunk_indices = summary_data.get("linked_summary_chunks", [])
                if isinstance(chunk_indices, (int, str)):
                    chunk_indices = [chunk_indices]
                for chunk_idx in chunk_indices:
                    summary_node_map[chunk_idx] = summary_id

            chunk_nodes_to_link = [
                (node_id, data)
                for node_id, data in self.get_nodes()
                if data.get("type", "").lower() == "chunk"
                and data.get("uuid", "") == uuid
                and "chunkIdx" in data
            ]

            link_count = 0
            for chunk_id, chunk_data in chunk_nodes_to_link:
                chunk_idx = chunk_data.get("chunkIdx", [])
                if chunk_idx in summary_node_map:
                    summary_id = summary_node_map[chunk_idx]

                    self.upsert_edge(chunk_id, summary_id, {"type": "IN_SUMMARY"})
                    link_count += 1

            logger.debug(f"Merged {link_count} IN_SUMMARY relationships.")

            summary_of_link_count = 0
            for summary_id, _ in summary_nodes:
                self.upsert_edge(summary_id, "Document/" + uuid, {"type": "SUMMARY_OF"})
                summary_of_link_count += 1

            logger.debug(
                f"Added {summary_of_link_count} SUMMARY_OF relationships to document {uuid}."
            )

    def persist_chunk_embeddings(self, data_for_embedding: List[Dict]):
        """Updates chunk embeddings using NetworkX node attributes."""
        with Metrics("NXGraphRAG/persist_chunk_embeddings", "blue"):
            update_count = 0
            for row in data_for_embedding:
                chunk_id = row["chunkId"]
                embedding = row["embedding"]
                self.upsert_node("Chunk/" + chunk_id, {"embedding": embedding})
                update_count += 1

            logger.debug(f"Updated embeddings for {update_count} chunk nodes.")

    def persist_chunk_entity_relationships(self, batch_data: List[Dict], uuid: str):
        """Creates HAS_ENTITY relationships using NetworkX operations."""
        with Metrics("NXGraphRAG/persist_chunk_entity_rels", "yellow"):
            link_count = 0
            for data in batch_data:
                chunk_node_id = data["chunk_hash"]
                entity_node_id = data["node_id"]
                self.upsert_edge(
                    "Chunk/" + chunk_node_id,
                    "Entity/" + entity_node_id,
                    {"type": "HAS_ENTITY"},
                )
                link_count += 1
            logger.debug(f"Merged {link_count} HAS_ENTITY relationships.")

    def update_knn(self):
        """Updates the KNN graph using cuML NearestNeighbors on NetworkX data."""
        with Metrics("NXGraphRAG/UpdateKNN", "blue"):
            knn_min_score = float(os.environ.get("KNN_MIN_SCORE", 0.75))
            logger.info(f"Updating KNN graph with min score {knn_min_score}")
            chunk_nodes_with_embeddings = [
                (n, data["embedding"])
                for n, data in self.get_nodes()
                if data.get("type", "").lower() == "chunk"
                and data.get("embedding") is not None
            ]

            if not chunk_nodes_with_embeddings:
                logger.warning("No Chunk nodes with embeddings found for KNN update.")
                return

            node_ids = [n for n, _ in chunk_nodes_with_embeddings]
            embeddings_list = [emb for _, emb in chunk_nodes_with_embeddings]

            valid_embeddings = []
            valid_node_ids = []
            for node_id, emb in zip(node_ids, embeddings_list):
                try:
                    emb_np = np.array(emb, dtype=np.float32)
                    if np.linalg.norm(emb_np) > 1e-6:
                        valid_embeddings.append(emb_np)
                        valid_node_ids.append(node_id)
                    else:
                        logger.debug(
                            f"Skipping node {node_id} with zero vector for KNN."
                        )
                except Exception as e:
                    logger.warning(
                        f"Error processing embedding for node {node_id}: {e}. Skipping."
                    )

            if len(valid_embeddings) < 2:
                logger.warning(
                    f"Insufficient valid embeddings ({len(valid_embeddings)}) for KNN update."
                )
                return

            embeddings_array = cp.array(valid_embeddings)
            num_samples = embeddings_array.shape[0]

            top_k = min(int(cp.ceil(cp.sqrt(num_samples)).get()), num_samples)

            logger.info(f"Calculating KNN for {num_samples} nodes with k={top_k}")
            try:
                nn_model = NearestNeighbors(n_neighbors=top_k, metric="cosine")
                nn_model.fit(embeddings_array)
                distances, indices = nn_model.kneighbors(embeddings_array)

                distances = cp.asnumpy(distances)
                indices = cp.asnumpy(indices)

            except Exception as e:
                logger.error(f"cuML NearestNeighbors failed: {e}")
                logger.error(traceback.format_exc())
                return

            added_edges = 0
            for i, source_node_id in enumerate(valid_node_ids):
                for j in range(1, top_k):
                    neighbor_original_idx = indices[i, j]
                    neighbor_node_id = valid_node_ids[neighbor_original_idx]
                    score = 1.0 - distances[i, j]

                    if score >= knn_min_score:
                        edge_data = self.graph.get_edge_data(
                            source_node_id, neighbor_node_id
                        )
                        edge_type = (
                            edge_data[len(edge_data) - 1]["type"] if edge_data else None
                        )
                        if edge_type and edge_type != "SIMILAR":
                            self.upsert_edge(
                                source_node_id,
                                neighbor_node_id,
                                {"type": "SIMILAR", "score": float(score)},
                            )
                            added_edges += 1

            logger.info(
                f"KNN graph updated. Added/updated {added_edges} SIMILAR edges."
            )

    def fetch_entities_needing_embedding(self) -> List[Dict[str, Any]]:
        """Fetches entities without embeddings from the NetworkX graph."""
        with Metrics("NXGraphRAG/fetch_entities_for_embedding", "green"):
            entities = []
            for node_id, data in self.get_nodes():
                is_entity = (
                    data.get("type", "").lower()
                    not in ["chunk", "document", "community"]
                    and data.get("embedding") is None
                    and node_id is not None
                )
                if is_entity:
                    description = data.get("description", "")
                    name = data.get("name", "")
                    text = f"{name} {description}".strip()
                    if not text:
                        text = data.get("_key", "null")
                    entities.append({"elementId": str(node_id), "text": text})
            return entities

    def persist_entity_embeddings(self, rows_with_embeddings: List[Dict]):
        """Updates entity embeddings using NetworkX node attributes."""
        with Metrics("NXGraphRAG/persist_entity_embeddings", "yellow"):
            update_count = 0
            for row in rows_with_embeddings:
                node_id = row["elementId"]
                embedding = row["embedding"]

                self.upsert_node(node_id, {"embedding": embedding})
                update_count += 1

            logger.debug(f"Updated embeddings for {update_count} entity nodes.")

    def fetch_summaries_needing_embedding(self) -> List[Dict[str, Any]]:
        """Fetches summary nodes without embeddings from the NetworkX graph."""
        with Metrics("NXGraphRAG/fetch_summaries_for_embedding", "green"):
            summaries = []
            for node_id, data in self.get_nodes():
                if (
                    data.get("type", "").lower() == "community"
                    and data.get("embedding") is None
                    and data.get("text", "") is not None
                ):
                    summaries.append({"id": str(node_id), "content": data["text"]})
            return summaries

    def persist_summary_embeddings(self, summaries_with_embeddings: List[Dict]):
        """Updates summary embeddings using NetworkX node attributes."""
        with Metrics("NXGraphRAG/persist_summary_embeddings", "yellow"):
            update_count = 0
            for summary in summaries_with_embeddings:
                summary_id = summary["id"]
                embedding = summary["embedding"]

                self.upsert_node(summary_id, {"embedding": embedding})
                update_count += 1

            logger.debug(f"Updated embeddings for {update_count} summary nodes.")

    def create_document_node(self, uuid: str, camera_id: str = "default"):
        """Ensures the main Document node exists in the NetworkX graph."""
        with Metrics("NXGraphRAG/create_document_node", "blue"):
            self.upsert_node(
                "Document/" + uuid,
                {
                    "type": "Document",
                    "uuid": uuid,
                    "camera_id": camera_id,
                },
            )

    async def aget_text_data(self, start_batch_index=0, end_batch_index=-1, uuid=""):
        """Async method to retrieve text data based on filter criteria.

        Args:
            start_batch_index: The start batch index
            end_batch_index: The end batch index
            uuid: The UUID of the document

        Returns:
            List of dictionaries containing the content and batch_i values
        """
        import asyncio

        # Add a small sleep to make this function truly async
        await asyncio.sleep(0.001)

        results = []
        for _, data in self.get_nodes(data=True):
            if (
                data.get("type", "").lower() == "community"
                and "batch_i" in data
                and data["batch_i"] >= start_batch_index
                and (end_batch_index == -1 or data["batch_i"] <= end_batch_index)
                and "text" in data
            ):
                if uuid and data.get("uuid", "") == uuid:
                    results.append({"text": data["text"], "batch_i": data["batch_i"]})
                elif not uuid:
                    results.append({"text": data["text"], "batch_i": data["batch_i"]})
        return results
