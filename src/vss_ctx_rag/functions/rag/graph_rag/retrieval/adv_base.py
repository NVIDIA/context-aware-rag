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

import json
from typing import Any, Dict, List

from langchain_core.documents import Document

from vss_ctx_rag.functions.rag.graph_rag.constants import (
    QUESTION_ANALYSIS_PROMPT,
)
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K
from vss_ctx_rag.utils.utils import remove_think_tags


class AdvGraphRetrieval:
    def __init__(
        self,
        llm,
        graph: GraphStorageTool,
        multi_channel=False,
        uuid="default",
        top_k=DEFAULT_RAG_TOP_K,
        max_retries=None,
        prompt_config=None,
    ):
        logger.info("Initializing AdvGraphRetrieval")
        self.chat_llm = llm
        self.graph_db = graph
        self.top_k = top_k
        self.max_retries = max_retries if max_retries else 3
        self.multi_channel = multi_channel
        self.uuid = uuid
        self.prompt_config = prompt_config
        neo4j_retriever = self.graph_db.get_neo4j_retriever(
            top_k=self.top_k, uuid=self.uuid, multi_channel=self.multi_channel
        )
        self.doc_retriever = self.graph_db.create_document_retriever(neo4j_retriever)
        logger.info(f"Initialized with top_k={self.top_k}")

    async def get_all_entity_types(self) -> List[str]:
        """Fetch all distinct entity types (node labels) from the Neo4j database"""
        logger.info("Fetching all entity types from Neo4j")

        query = """
        CALL db.labels()
        YIELD label
        RETURN DISTINCT label
        ORDER BY label
        """

        try:
            result = await self.graph_db.aquery(query)
            entity_types = [record["label"] for record in result]
            logger.info(f"Found {len(entity_types)} entity types: {entity_types}")
            return entity_types
        except Exception as e:
            logger.error(f"Error fetching entity types: {e}")
            return []

    def _build_property_filters(self, properties: Dict) -> str:
        if not properties:
            return ""
        filters = []
        for key, value in properties.items():
            filters.append(f"n.{key} = '{value}'")
        return "WHERE " + " AND ".join(filters)

    async def retrieve_by_entity_type(
        self, entity_type: str, properties: Dict = None
    ) -> List[Dict]:
        """Retrieve nodes of specific type with optional property filters"""
        logger.info(
            f"Retrieving entities of type {entity_type} with properties {properties}"
        )
        query = f"""
        MATCH (n:{entity_type})
        {self._build_property_filters(properties) if properties else ""}
        RETURN n
        LIMIT {self.top_k or 10}
        """
        return await self.graph_db.aquery(query)

    async def retrieve_by_relationship(
        self,
        start_type: str,
        relationship: str,
        end_type: str,
        time_range: Dict = None,
    ) -> List[Dict]:
        """Retrieve relationships between node types with optional time filtering"""
        logger.info(
            f"Retrieving relationships {relationship} between "
            f"{start_type} and {end_type}"
        )
        time_filter = ""
        if time_range:
            time_filter = f"""
            WHERE r.start_timestamp >= {time_range.get("start", 0)}
            AND r.end_timestamp <= {time_range.get("end", "infinity")}
            """
            logger.info(f"Added time filter: {time_range}")

        query = f"""
        MATCH (start:{start_type})-[r:{relationship}]->(end:{end_type})
        {time_filter}
        RETURN start, r, end
        LIMIT {self.top_k or 10}
        """
        return await self.graph_db.aquery(query)

    async def retrieve_temporal_context(
        self, start_time: float, end_time: float
    ) -> List[Dict]:
        """Retrieve all events between start and end times"""
        logger.info(f"Retrieving temporal context between {start_time} and {end_time}")
        result = []
        temporal_filter = ""
        if start_time is None and end_time is None:
            return result
        if start_time is not None:
            temporal_filter = f"""
            AND toFloat(n.start_time) >= {start_time}
            """
        if end_time is not None:
            temporal_filter = (
                temporal_filter
                + f"""
            AND toFloat(n.end_time) <= {end_time}
            """
            )
        query = f"""
        MATCH (n: Chunk)
        WHERE n.start_time IS NOT NULL AND n.end_time IS NOT NULL
        {temporal_filter}
        RETURN n
            ORDER BY n.start_time
            LIMIT {self.top_k or 10}
            """
        result = await self.graph_db.aquery(query)
        return result

    async def retrieve_semantic_context(
        self,
        question: str,
        start_time: float = None,
        end_time: float = None,
        sort_by: str = None,
    ) -> List[Dict]:
        """Retrieve semantically similar content using vector similarity search"""
        logger.info(
            f"Retrieving semantic context for question: {question} "
            f"between {start_time} and {end_time}"
        )

        result = await self.doc_retriever.ainvoke(
            question,
        )
        # logger.info(f"Semantic search results raw: {result}")
        processed_results = []
        for doc in result:
            processed_results.append(
                {
                    "n": {
                        "text": doc.page_content,
                        "start_time": doc.metadata.get("start_time", ""),
                        "end_time": doc.metadata.get("end_time", ""),
                        "chunkIdx": doc.metadata.get("chunkIdx", ""),
                        "score": doc.state.get("query_similarity_score", 0),
                        "asset_dirs": doc.metadata.get("asset_dirs", []),
                    }
                }
            )
            if sort_by == "score":
                processed_results.sort(key=lambda x: x["n"]["score"], reverse=True)
            elif sort_by == "start_time":
                processed_results.sort(key=lambda x: x["n"]["start_time"])
            elif sort_by == "end_time":
                processed_results.sort(key=lambda x: x["n"]["end_time"])
            else:
                processed_results.sort(key=lambda x: x["n"]["score"], reverse=True)
        return processed_results

    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze question and determine retrieval strategy"""
        logger.info(f"Analyzing question: {question}")

        prompt = (
            self.prompt_config["QUESTION_ANALYSIS_PROMPT"].format(
                question=question, entity_types=await self.get_all_entity_types()
            )
            if (self.prompt_config and self.prompt_config["QUESTION_ANALYSIS_PROMPT"])
            else QUESTION_ANALYSIS_PROMPT.format(
                question=question, entity_types=await self.get_all_entity_types()
            )
        )
        response = await self.chat_llm.ainvoke(prompt)
        logger.info("Question analysis complete")
        # Parse LLM response to get retrieval strategy
        # This is a simplified version - you'd want proper JSON parsing
        return remove_think_tags(response.content)

    async def retrieve_relevant_context(self, question: str) -> List[Document]:
        """Main retrieval method that orchestrates different retrieval strategies"""
        with Metrics("AdvGraphRetrieval/retrieve_context", "blue") as tm:
            tm.input({"question": question})
            logger.info(f"Starting context retrieval for question: {question}")
            analysis_response = await self.analyze_question(question)
            json_start = analysis_response.find("{")
            json_end = analysis_response.rfind("}") + 1

            # Parse the JSON response from the LLM with retries
            logger.info(f"Analysis response: {analysis_response}")
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    analysis = json.loads(analysis_response[json_start:json_end])
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse JSON response: {e}")
                    retry_count += 1
                    if retry_count < self.max_retries:
                        # Retry getting analysis
                        analysis_response = await self.analyze_question(question)
                        json_start = analysis_response.find("{")
                        json_end = analysis_response.rfind("}") + 1
                        logger.info(
                            f"Retry {retry_count}: New analysis response: {analysis_response}"
                        )
                    else:
                        logger.error("Max retries reached, using default analysis")
                        analysis = {
                            "entity_types": [],
                            "relationships": [],
                            "time_references": {},
                            "location_references": [],
                            "sort_by": "score",
                            "retrieval_strategy": "similarity",
                        }

            # Collect context from multiple retrieval strategies
            contexts = []

            # Get retrieval strategy and parameters from analysis
            strategy = analysis.get("retrieval_strategy", "")
            logger.info(f"Using retrieval strategy: {strategy}")

            time_refs = analysis.get("time_references", {})
            start_time = None
            end_time = None

            # Temporal context retrieval
            if time_refs:
                start_time = time_refs.get("start")
                end_time = time_refs.get("end")

            if strategy == "temporal":
                temporal_data = await self.retrieve_temporal_context(
                    start_time, end_time
                )
                logger.info("Temporal Contexts...")
                logger.debug(f"Temporal Data: {temporal_data}")
                if temporal_data:
                    contexts.extend(temporal_data)
                    logger.info(f"Retrieved {len(temporal_data)} temporal records")
            else:  # semantic retrieval
                # Semantic similarity retrieval
                semantic_data = await self.retrieve_semantic_context(
                    question,
                    start_time=start_time,
                    end_time=end_time,
                    sort_by=analysis.get("sort_by", "score"),
                )
                logger.info("Semantic Contexts...")
                logger.debug(f"Semantic Data: {semantic_data}")
                if semantic_data:
                    contexts.extend(semantic_data)

            logger.info(f"Contexts: {contexts}")

            # Relationship-based retrieval
            relationships = analysis.get("relationships", [])
            for rel in relationships:
                if isinstance(rel, str):
                    # If relationship is specified without types, skip
                    continue
                start_type = rel.get("from")
                end_type = rel.get("to")
                rel_type = rel.get("type")
                if all([start_type, end_type, rel_type]):
                    rel_data = await self.retrieve_by_relationship(
                        start_type, rel_type, end_type
                    )
                    logger.debug(f"Relationship Data: {rel_data}")
                    if rel_data:
                        contexts.extend(rel_data)
                        logger.info(
                            f"Retrieved {len(rel_data)} records for "
                            f"relationship {rel_type}"
                        )

            # Convert to Documents
            documents = []
            for ctx in contexts:
                # Convert Neo4j results to Document format
                # Check if ctx has expected structure
                if isinstance(ctx, dict) and "n" in ctx:
                    if "text" in ctx["n"]:
                        doc = Document(
                            page_content=str(ctx["n"].get("text", "")),
                            metadata={
                                "start_time": ctx.get("n", {}).get("start_time", ""),
                                "end_time": ctx.get("n", {}).get("end_time", ""),
                                "asset_dirs": ctx.get("n", {}).get("asset_dirs", []),
                            },
                        )
                        documents.append(doc)

            logger.info(f"Returning {len(documents)} documents")

            tm.output({"documents": documents})
            return documents
