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
import hashlib
import traceback
from typing import Dict, List
import os
import json
import re

from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import httpx
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import openai
from requests.exceptions import ConnectionError

from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_batcher import Batch, Batcher
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_EMBEDDING_PARALLEL_COUNT,
    DEFAULT_CONCURRENT_EMBEDDING_LIMIT,
)


class GraphIngestion:
    """
    Class for handling extraction and processing of graph-based knowledge representations.

    This class manages the extraction of structured knowledge from text into a graph format,
    handling document processing, entity extraction, and graph construction, while deferring
    database-specific operations to subclasses.
    """

    def __init__(
        self,
        batcher: Batcher,
        duplicate_score_value: float,
        llm,
        graph_db: GraphStorageTool,
        embedding_parallel_count: int = DEFAULT_EMBEDDING_PARALLEL_COUNT,
        node_types: List[str] = ["Person", "Vehicle", "Location", "Object"],
        relationship_types: List[str] = [],
        deduplicate_nodes: bool = False,
        disable_entity_description: bool = True,
        disable_entity_extraction: bool = False,
        chunk_size: int = 500,
        chunk_overlap: int = 10,
    ):
        self.graph_db = graph_db
        self.embedding = self.graph_db.embedding
        self.llm = llm
        if disable_entity_description:
            node_properties = False
            relationship_properties = False
            node_types = []
            relationship_types = []
            ignore_tool_usage = True
            # logger.warning(
            #     "Entity description is disabled, node_types and relationship_types will be ignored"
            # )
        else:
            node_properties = ["description"]
            relationship_properties = ["description"]
            ignore_tool_usage = False

        self.transformer = LLMGraphTransformer(
            llm=llm,
            allowed_nodes=node_types,
            allowed_relationships=relationship_types,
            node_properties=node_properties,
            relationship_properties=relationship_properties,
            ignore_tool_usage=ignore_tool_usage,
        )
        self.disable_entity_description = disable_entity_description
        self.disable_entity_extraction = disable_entity_extraction
        self.duplicate_score_value = duplicate_score_value
        self.batcher = batcher
        self.cleaned_graph_documents_list: List[GraphDocument] = []
        self.previous_chunk_id: str = "0"
        self.last_position: int = 0
        self.embedding_parallel_count: int = embedding_parallel_count
        self.deduplicate_nodes = deduplicate_nodes
        self.subtitles_path: str = ""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedding_semaphore = asyncio.Semaphore(
            DEFAULT_CONCURRENT_EMBEDDING_LIMIT
        )

    def handle_backticks_nodes_relationship_id_type(
        self, graph_document_list: List[GraphDocument]
    ) -> List[GraphDocument]:
        """Removes backticks from node and relationship types."""
        with Metrics("GraphRAG/Base/handle-backticks", "blue"):
            for graph_document in graph_document_list:
                cleaned_nodes = []
                for node in graph_document.nodes:
                    node.properties.update(
                        {
                            "uuid": graph_document.source.metadata.get(
                                "uuid", "default"
                            ),
                            "camera_id": graph_document.source.metadata.get(
                                "camera_id", ""
                            )
                            if graph_document.source.metadata.get("camera_id", "")
                            != "default"
                            else "",
                        }
                    )
                    if node.type.strip() and node.id.strip():
                        node.type = node.type.replace("`", "")
                        cleaned_nodes.append(node)
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

    def get_split_chunk_document_list(self, docs: List[Document]) -> List[Document]:
        """Splits documents into smaller chunks using a TokenTextSplitter."""
        with Metrics("GraphRAG/Base/combine-chunks", "yellow"):
            text_splitter = RecursiveCharacterTextSplitter(
                separators=["\n\n", "\n"],
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            return text_splitter.split_documents(docs)

    def replace_node_ids_with_names(self, graph_documents: List[GraphDocument]):
        """Replace node IDs with hashed values while preserving original IDs as names.

        Args:
            graph_documents: List of GraphDocument objects to process

        Returns:
            The same graph documents with updated node IDs/names
        """
        with Metrics("GraphRAG/aprocess-doc/graph-create/replace-node-ids", "blue"):
            logger.info("Replacing node IDs with hashed values")
            for graph_doc in graph_documents:
                # Keep track of ID mappings
                id_mapping = {}

                # First pass: create ID mappings
                for node in graph_doc.nodes:
                    old_id = node.id

                    description = node.properties.get("description", "")
                    node_type = node.type if node.type else "Entity"

                    # Generate a new ID based on hash of description + name + uuid
                    hash_input = f"{description}_{node_type}_{old_id}_{graph_doc.source.metadata.get("uuid", "default")}"
                    hash_obj = hashlib.sha1(hash_input.encode())
                    new_id = hash_obj.hexdigest()

                    # Store the mapping
                    id_mapping[old_id] = new_id

                    # Save the old ID as a name property
                    node.properties.update({"name": old_id})

                    # Update the node ID
                    node.id = new_id

                # Second pass: update all relationships
                for rel in graph_doc.relationships:
                    if rel.source.id in id_mapping:
                        rel.source.id = id_mapping[rel.source.id]
                    if rel.target.id in id_mapping:
                        rel.target.id = id_mapping[rel.target.id]

            return graph_documents

    async def aconvert_to_graph_documents(self, documents):
        """Custom implementation that converts documents to graph documents and replaces node IDs with meaningful names.

        This method wraps the original LLMGraphTransformer.aconvert_to_graph_documents
        and adds node ID to name conversion.

        Args:
            documents: List of Document objects to convert to graph format

        Returns:
            List of GraphDocument objects with descriptive node names
        """
        with Metrics("GraphRAG/aprocess-doc/graph-create/custom-convert", "blue"):
            logger.info(
                "Converting documents to graph format with descriptive node names"
            )

            with Metrics("GraphRAG/aprocess-doc/graph-create/convert", "blue"):

                @retry(
                    stop=stop_after_attempt(5),  # More attempts for network issues
                    wait=wait_exponential(
                        multiplier=2, min=4, max=60
                    ),  # Longer backoff for network
                    retry=retry_if_exception_type(
                        (
                            # Network/HTTP errors
                            httpx.ConnectError,
                            httpx.TimeoutException,
                            httpx.ReadTimeout,
                            httpx.ConnectTimeout,
                            # OpenAI specific errors
                            openai.APITimeoutError,
                            openai.APIConnectionError,
                            openai.RateLimitError,
                            # General connection errors
                            ConnectionError,
                            TimeoutError,
                        )
                    ),
                    reraise=True,
                )
                async def transformer_with_retry(documents):
                    """Wrapper function for transformer with retry logic"""
                    return await self.transformer.aconvert_to_graph_documents(documents)

                # First, use the original transformer to convert documents to graph format
                graph_documents = await transformer_with_retry(documents)

            with Metrics("GraphRAG/aprocess-doc/graph-create/clean", "blue"):
                cleaned_graph_documents = (
                    self.handle_backticks_nodes_relationship_id_type(graph_documents)
                )
            # Then replace the node IDs with descriptive names
            return self.replace_node_ids_with_names(cleaned_graph_documents)

    async def _create_empty_graph_documents(
        self, documents: List[Document]
    ) -> List[GraphDocument]:
        """Creates empty graph documents when entity extraction is disabled."""
        return [
            GraphDocument(
                source=doc,
                nodes=[],
                relationships=[],
            )
            for doc in documents
        ]

    async def _extract_entities_with_fallback(
        self, documents: List[Document]
    ) -> List[GraphDocument]:
        """Extracts entities from documents with fallback to disabled entity description on failure."""
        try:
            with Metrics("GraphRAG/Base/convert_to_graph_documents", "blue"):
                return await self.aconvert_to_graph_documents(documents)
        except Exception as e:
            # Check for specific error code 400 (Bad Request)
            error_message = str(e)
            if "Error code: 400" in error_message or "BadRequestError" in error_message:
                logger.warning(
                    f"Disabling entity description, LLM doesn't support structured output: {e}{traceback.format_exc()}"
                )
            else:
                logger.error(
                    f"Error in _extract_entities_with_fallback: {e}{traceback.format_exc()}"
                )
                raise e

            # Fallback: reconfigure transformer and retry
            self._configure_fallback_transformer()
            self.disable_entity_description = True

            with Metrics("GraphRAG/Base/convert_to_graph_documents", "blue"):
                return await self.aconvert_to_graph_documents(documents)

    def _configure_fallback_transformer(self):
        """Configures transformer for fallback mode when entity description fails."""
        self.transformer = LLMGraphTransformer(
            llm=self.llm,
            allowed_nodes=[],
            allowed_relationships=[],
            node_properties=False,
            relationship_properties=False,
            ignore_tool_usage=True,
        )

    def _convert_batch_to_documents(self, batch: Batch) -> List[Document]:
        """Converts batch items to Document objects, filtering out empty content."""
        docs = []
        for doc, _, metadata in batch.as_list():
            if doc != ".":
                docs.append(Document(page_content=doc, metadata=metadata))
        return docs

    async def acreate_graph(self, batch: Batch):
        """Processes a batch of documents to extract graph structures and adds them."""
        with Metrics("GraphRAG/Base/acreate_graph", "yellow"):
            try:
                # Convert batch to documents
                docs = self._convert_batch_to_documents(batch)
                if not docs:
                    logger.info("No valid documents in batch, skipping graph creation")
                    return

                # Split documents into chunks
                split_chunk_document_list = self.get_split_chunk_document_list(docs)

                # Extract graph documents based on configuration
                if self.disable_entity_extraction:
                    cleaned_graph_documents = await self._create_empty_graph_documents(
                        split_chunk_document_list
                    )
                else:
                    cleaned_graph_documents = (
                        await self._extract_entities_with_fallback(
                            split_chunk_document_list
                        )
                    )

                # Store and persist the results
                self.cleaned_graph_documents_list.extend(cleaned_graph_documents)

                with Metrics("GraphRAG/Base/add_graph_documents_to_db", "green"):
                    self.graph_db.add_graph_documents_to_db(cleaned_graph_documents)

            except Exception as e:
                logger.error(f"Error in acreate_graph: {e}{traceback.format_exc()}")
                raise e

    def create_relation_between_chunks(self, uuid: str) -> list:
        """
        Generates chunk data and relationship structures based on the processed documents.
        Calls abstract methods to persist this data in the specific database.
        Returns a list containing chunk IDs and the original chunk document objects.
        """
        logger.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
        with Metrics("GraphRAG/Base/create_relation_between_chunks", "green"):
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
                value_for_hash = chunk.source.page_content + chunk.source.metadata.get(
                    "uuid", "default"
                )
                page_content_sha1 = hashlib.sha1(value_for_hash.encode())
                previous_id = current_chunk_id
                current_chunk_id = page_content_sha1.hexdigest()

                self.last_position += 1
                if i > 0:
                    offset += len(
                        self.cleaned_graph_documents_list[i - 1].source.page_content
                    )

                is_first_chunk = (
                    i == 0 and chunk.source.metadata.get("chunkIdx", 0) == 0
                )

                metadata = {
                    "position": self.last_position,
                    "length": len(chunk.source.page_content),
                    "content_offset": offset,
                    "hash": current_chunk_id,
                    **chunk.source.metadata,
                }

                self.cleaned_graph_documents_list[i].source.metadata.update(metadata)
                chunk.source.metadata.update(metadata)

                chunk_document = Document(
                    page_content=chunk.source.page_content, metadata=metadata
                )

                chunk_data = {
                    "id": current_chunk_id,
                    "text": chunk_document.page_content,
                    "position": self.last_position,
                    "length": chunk_document.metadata["length"],
                    "uuid": chunk_document.metadata.get("uuid", "default"),
                    "camera_id": chunk_document.metadata.get("camera_id", "")
                    if chunk_document.metadata.get("camera_id", "") != "default"
                    else "",
                    "content_offset": offset,
                    "chunkIdx": chunk_document.metadata.get("chunkIdx", ""),
                    "start_pts": chunk_document.metadata.get("start_pts", ""),
                    "end_pts": chunk_document.metadata.get("end_pts", ""),
                    "streamId": chunk_document.metadata.get("streamId", ""),
                    "file": chunk_document.metadata.get("file", ""),
                    "asset_dir": chunk_document.metadata.get("asset_dir", ""),
                    "pts_offset_ns": chunk_document.metadata.get("pts_offset_ns", ""),
                    "start_time": chunk_document.metadata.get("start_ntp_float", ""),
                    "end_time": chunk_document.metadata.get("end_ntp_float", ""),
                    "subtitles_path": chunk_document.metadata.get("subtitles_path", ""),
                }
                chunk_data = {k: v for k, v in chunk_data.items() if v is not None}
                self.subtitles_path = chunk_document.metadata.get("subtitles_path", "")
                batch_data.append(chunk_data)

                lst_chunks_including_hash.append(
                    {"chunk_id": current_chunk_id, "chunk_doc": chunk}
                )

                if is_first_chunk:
                    relationships.append(
                        {"type": "FIRST_CHUNK", "chunk_id": current_chunk_id}
                    )
                else:
                    relationships.append(
                        {
                            "type": "NEXT_CHUNK",
                            "previous_chunk_id": previous_id,
                            "current_chunk_id": current_chunk_id,
                        }
                    )

                if "summary_id" in chunk.source.metadata:
                    relationships.append(
                        {
                            "type": "IN_SUMMARY",
                            "chunk_id": current_chunk_id,
                            "summary_id": chunk.source.metadata["summary_id"],
                        }
                    )

            self.previous_chunk_id = current_chunk_id

            if batch_data:
                self.graph_db.persist_chunk_data(
                    batch_data,
                    relationships,
                    uuid,
                )

            self.graph_db.persist_summary_chunk_relationships(uuid)

            return lst_chunks_including_hash

    async def update_embedding_chunks(self, chunkId_chunkDoc_list: List[Dict]):
        """Calculates embeddings for chunks and persists them using abstract methods."""
        with Metrics("GraphRAG/Base/update_embedding_chunks", "blue"):
            if not chunkId_chunkDoc_list:
                logger.info("No chunks to update embeddings for.")
                return

            data_for_embedding = []
            logger.info(
                f"Calculating embeddings for {len(chunkId_chunkDoc_list)} chunks..."
            )

            async def semaphore_controlled_embed(content):
                async with self._embedding_semaphore:
                    return await self.embedding.aembed_query(content)

            tasks = [
                asyncio.create_task(
                    semaphore_controlled_embed(row["chunk_doc"].source.page_content)
                )
                for row in chunkId_chunkDoc_list
            ]
            results = await asyncio.gather(*tasks)

            for i, row in enumerate(chunkId_chunkDoc_list):
                data_for_embedding.append(
                    {"chunkId": row["chunk_id"], "embedding": results[i]}
                )

            if data_for_embedding:
                logger.info(f"Persisting {len(data_for_embedding)} chunk embeddings.")
                self.graph_db.persist_chunk_embeddings(data_for_embedding)

    def merge_relationship_between_chunk_and_entites(self):
        """Creates HAS_ENTITY relationships between chunks and entities using abstract methods."""
        with Metrics("GraphRAG/Base/merge_chunk_entity_relationships", "yellow"):
            batch_data = []
            logger.debug(
                "Preparing HAS_ENTITY relationship data between chunks and entities"
            )
            for graph_doc in self.cleaned_graph_documents_list:
                chunk_hash = graph_doc.source.metadata.get("hash")
                if not chunk_hash:
                    logger.warning(
                        f"Chunk hash missing for source: {graph_doc.source.page_content[:50]}... Skipping."
                    )
                    continue

                for node in graph_doc.nodes:
                    query_data = {
                        "chunk_hash": chunk_hash,
                        "node_type": node.type,
                        "node_id": node.id,
                    }
                    batch_data.append(query_data)

            if batch_data:
                logger.info(f"Merging {len(batch_data)} HAS_ENTITY relationships.")
                self.graph_db.persist_chunk_entity_relationships(
                    batch_data, graph_doc.source.metadata.get("uuid", "default")
                )
            else:
                logger.debug("No HAS_ENTITY relationships to merge.")

    async def create_entity_embedding(self):
        """Fetches entities needing embeddings, calculates them, and persists them."""
        rows_to_embed = []
        logger.info("Starting entity embedding creation process.")
        with Metrics("GraphRAG/Base/FetchEntEmbd", "green"):
            rows_to_embed = self.graph_db.fetch_entities_needing_embedding()

        if not rows_to_embed:
            logger.info("No entities found requiring embedding.")
            return

        logger.info(
            f"Found {len(rows_to_embed)} entities needing embedding. Processing in batches."
        )
        all_updated_rows = []
        for i in range(0, len(rows_to_embed), self.embedding_parallel_count):
            batch = rows_to_embed[i : i + self.embedding_parallel_count]
            updated_rows = await self.update_embeddings_batch(batch)
            all_updated_rows.extend(updated_rows)

        if all_updated_rows:
            logger.info(f"Persisting embeddings for {len(all_updated_rows)} entities.")
            self.graph_db.persist_entity_embeddings(all_updated_rows)

    async def update_embeddings_batch(self, rows: List[Dict]) -> List[Dict]:
        """Calculates embeddings for a batch of entities."""
        with Metrics("GraphRAG/Base/UpdateEmbdingBatch", "yellow"):
            logger.info(f"Calculating embeddings for {len(rows)} entities in batch.")

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
            return rows

    async def create_summary_embeddings(self):
        """Fetches summaries needing embeddings, calculates them, and persists them."""
        summaries_to_embed = []
        logger.info("Starting summary embedding creation process.")
        with Metrics("GraphRAG/Base/FetchSummaryEmbd", "green"):
            summaries_to_embed = self.graph_db.fetch_summaries_needing_embedding()

        if not summaries_to_embed:
            logger.info("No summaries found requiring embedding.")
            return

        logger.info(f"Found {len(summaries_to_embed)} summaries needing embedding.")

        async def semaphore_controlled_embed(content):
            async with self._embedding_semaphore:
                return await self.embedding.aembed_query(content)

        tasks = [
            asyncio.create_task(semaphore_controlled_embed(node["content"]))
            for node in summaries_to_embed
            if node.get("content", "").strip()
        ]
        embeddings = await asyncio.gather(*tasks)

        summaries_with_embeddings = []
        for node, embedding in zip(summaries_to_embed, embeddings):
            summaries_with_embeddings.append({"id": node["id"], "embedding": embedding})

        if summaries_with_embeddings:
            logger.info(
                f"Persisting embeddings for {len(summaries_with_embeddings)} summaries."
            )
            self.graph_db.persist_summary_embeddings(summaries_with_embeddings)

    def parse_subtitle_time(self, time_str, split=":"):
        h, m, s_ms = time_str.split(":")
        s, ms = s_ms.split(split)
        return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000

    def load_subtitles(self, subtitles_path):
        if os.path.basename(subtitles_path).endswith(".srt"):
            if os.path.exists(subtitles_path):
                subtitles = {}
                with open(subtitles_path, "r", encoding="utf-8") as file:
                    content = file.read().split("\n\n")
                    for section in content:
                        if section.strip():
                            lines = section.split("\n")
                            if len(lines) >= 3:
                                time_range = lines[1].split(" --> ")
                                start_time = self.parse_subtitle_time(
                                    time_range[0], ","
                                )
                                end_time = self.parse_subtitle_time(time_range[1], ",")
                                text = " ".join(line for line in lines[2:])
                                subtitles[(start_time, end_time)] = text
        else:
            starting_timestamp_for_subtitles = 0
            with open(subtitles_path, "r", encoding="utf-8") as file:
                data_li = json.load(file)
            lvb_val = os.environ.get("lvb_val", "")
            if lvb_val:
                with open(os.environ["lvb_val"]) as file:
                    data_di = json.load(file)
                for data in data_di:
                    if (
                        os.path.basename(subtitles_path).replace("_en.json", "")
                        in data["video_id"]
                    ):
                        starting_timestamp_for_subtitles = data[
                            "starting_timestamp_for_subtitles"
                        ]
                        duration = data["duration"]
            else:
                logger.error(
                    "lvb_val is not set, Please set the lvb_val environment variable."
                )
                duration = 0
            subtitles = {}
            for dic in data_li:
                if "start" in dic and "end" in dic and "line" in dic:
                    start_time = (
                        self.parse_subtitle_time(dic["start"], ".")
                        - starting_timestamp_for_subtitles
                    )
                    end_time = (
                        self.parse_subtitle_time(dic["end"], ".")
                        - starting_timestamp_for_subtitles
                    )
                    subtitles[(start_time, end_time)] = dic["line"]
                elif "timestamp" in dic and "text" in dic:
                    if dic["timestamp"][0] is not None:
                        start_time = (
                            dic["timestamp"][0] - starting_timestamp_for_subtitles
                        )
                    else:
                        start_time = dic["timestamp"][0]
                    if dic["timestamp"][1] is not None:
                        end_time = (
                            dic["timestamp"][1] - starting_timestamp_for_subtitles
                        )
                    else:
                        end_time = duration - starting_timestamp_for_subtitles
                    subtitles[(start_time, end_time)] = dic["text"]
        return subtitles

    def extract_subtitles(self, subtitles_path):
        subtitles = self.load_subtitles(subtitles_path)
        subtitle_frames = []
        for (start_time, end_time), text in subtitles.items():
            pattern = r'<font color="white" size=".72c">(.*?)</font>'
            raw_text = re.findall(pattern, text)
            if end_time == "null" or end_time is None:
                end_time = start_time
            try:
                text = raw_text[0]
                if text != "":
                    subtitle_frames.append(
                        {
                            "start_time": float(start_time),
                            "end_time": float(end_time),
                            "text": text,
                        }
                    )
            except Exception:
                subtitle_frames.append(
                    {
                        "start_time": float(start_time),
                        "end_time": float(end_time),
                        "text": text,
                    }
                )

        return subtitle_frames

    async def create_subtitle_embedding(self):
        """Fetches subtitles needing embeddings, calculates them, and persists them."""
        rows_to_embed = []
        logger.info("Starting subtitle embedding creation process.")
        with Metrics("GraphRAG/Base/FetchSubEmbd", "green"):
            rows_to_embed = self.graph_db.fetch_subtitle_for_embedding()

        if not rows_to_embed:
            logger.info("No subtitles found requiring embedding.")
            return

        logger.info(
            f"Found {len(rows_to_embed)} subtitles needing embedding. Processing in batches."
        )
        all_updated_rows = []
        for i in range(0, len(rows_to_embed), self.embedding_parallel_count):
            batch = rows_to_embed[i : i + self.embedding_parallel_count]
            updated_rows = await self.update_embeddings_batch(batch)
            all_updated_rows.extend(updated_rows)

        if all_updated_rows:
            logger.info(f"Persisting embeddings for {len(all_updated_rows)} subtitles.")
            self.graph_db.persist_subtitle_embeddings(all_updated_rows)

    async def apost_process(self, uuid: str, camera_id: str = ""):
        """Orchestrates the post-processing steps after all batches are processed."""
        with Metrics("GraphRAG/Base/apost_process", "green"):
            logger.info("Starting post-processing...")
            if camera_id == "default":
                camera_id = ""

            self.graph_db.create_document_node(uuid, camera_id)

            chunkId_chunkDoc_list = self.create_relation_between_chunks(uuid)

            await self.update_embedding_chunks(chunkId_chunkDoc_list)
            await self.create_summary_embeddings()
            self.merge_relationship_between_chunk_and_entites()

            await asyncio.to_thread(self.graph_db.update_knn)
            await self.create_entity_embedding()
            await self.graph_db.finalize_graph_creation()
            if not self.disable_entity_description and self.deduplicate_nodes:
                self.graph_db.merge_duplicate_nodes(self.duplicate_score_value)
            if self.subtitles_path != "" and self.subtitles_path is not None:
                subtitle_frames = self.extract_subtitles(self.subtitles_path)
                self.graph_db.persist_subtitle_frames(subtitle_frames)
                await self.create_subtitle_embedding()

            self.cleaned_graph_documents_list.clear()
            logger.info("Graph post-processing complete.")

    def reset(self):
        """Resets the internal state for processing a new document/batch."""
        self.cleaned_graph_documents_list.clear()
        self.previous_chunk_id = "0"
        self.last_position = 0
        logger.debug("Graph extraction state reset.")
