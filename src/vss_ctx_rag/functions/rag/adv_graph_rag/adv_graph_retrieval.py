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

"""adv_graph_retrieval.py: File contains AdvGraphRetrieval class"""

from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any
import json

from langchain_core.documents import Document
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.utils.utils import remove_lucene_chars, remove_think_tags
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
    QUESTION_TRANSFORM_TEMPLATE,
    VECTOR_SEARCH_TOP_K,
    CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
)
from langchain_community.vectorstores import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.runnables import RunnableBranch
from langchain_core.messages import HumanMessage


class AdvGraphRetrieval:
    def __init__(self, llm, graph: Neo4jGraphDB, top_k=None, max_retries=None):
        logger.info("Initializing AdvGraphRetrieval")
        self.chat_llm = llm
        self.graph_db = graph
        self.top_k = top_k
        self.max_retries = max_retries if max_retries else 3
        self.vector_retriever = Neo4jVector.from_existing_index(
            embedding=self.graph_db.embeddings,
            index_name="vector",
            graph=self.graph_db.graph_db,
        ).as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.top_k or VECTOR_SEARCH_TOP_K,
                "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
            },
        )
        self.doc_retriever = self.create_document_retriever_chain()
        logger.info(f"Initialized with top_k={top_k}")

    def _format_relative_time(self, timestamp: float) -> str:
        """Format timestamp relative to current time in seconds"""
        if timestamp is None:
            return "unknown time"

        now = datetime.now(timezone.utc).timestamp()
        diff = now - timestamp
        return f"{int(diff)} seconds ago"

    def create_document_retriever_chain(self):
        with TimeMeasure("GraphRetrieval/CreateDocRetChain", "blue"):
            try:
                logger.info("Starting to create document retriever chain")

                query_transform_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", QUESTION_TRANSFORM_TEMPLATE),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                output_parser = StrOutputParser()

                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.graph_db.embeddings,
                    similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
                )
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[embeddings_filter]
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=pipeline_compressor,
                    base_retriever=self.vector_retriever,
                )
                query_transforming_retriever_chain = RunnableBranch(
                    (
                        lambda x: len(x.get("messages", [])) == 1,
                        (lambda x: x["messages"][-1].content)
                        | output_parser
                        | remove_lucene_chars
                        | compression_retriever,
                    ),
                    query_transform_prompt
                    | self.chat_llm
                    | output_parser
                    | remove_think_tags
                    | remove_lucene_chars
                    | compression_retriever,
                ).with_config(run_name="chat_retriever_chain")

                logger.info("Successfully created document retriever chain")
                return query_transforming_retriever_chain

            except Exception as e:
                logger.error(
                    f"Error creating document retriever chain: {e}", exc_info=True
                )
                raise

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
            result = await self.graph_db.arun_cypher_query(query)
            entity_types = [record["label"] for record in result]
            logger.info(f"Found {len(entity_types)} entity types")
            return entity_types
        except Exception as e:
            logger.error(f"Error fetching entity types: {e}")
            return []

    async def get_all_stream_ids(self) -> List[str]:
        """Fetch all distinct stream_id values from the Neo4j database"""
        logger.info("Fetching all stream_ids from Neo4j")

        query = """
        MATCH (n:Chunk)
        WHERE n.stream_id IS NOT NULL
        RETURN DISTINCT n.stream_id as stream_id
        ORDER BY stream_id
        """

        try:
            result = await self.graph_db.arun_cypher_query(query)
            stream_ids = [record["stream_id"] for record in result]
            logger.info(f"Found {len(stream_ids)} stream_ids: {stream_ids}")
            return stream_ids
        except Exception as e:
            logger.error(f"Error fetching stream_ids: {e}")
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
        return await self.graph_db.arun_cypher_query(query)

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
        return await self.graph_db.arun_cypher_query(query)

    async def retrieve_temporal_context(
        self, start_time: float, end_time: float, stream_ids: List[str] = None
    ) -> List[Dict]:
        """Retrieve all events between start and end times, optionally filtered by stream_id"""
        logger.info(f"Retrieving temporal context between {start_time} and {end_time}")
        if stream_ids:
            logger.info(f"Filtering by stream_ids: {stream_ids}")

        result = []
        temporal_filter = ""
        stream_filter = ""

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

        if stream_ids:
            # Create filter for multiple stream_ids
            stream_conditions = [f"n.stream_id = '{stream_id}'" for stream_id in stream_ids]
            stream_filter = f"""
            AND ({' OR '.join(stream_conditions)})
            """

        query = f"""
        MATCH (n: Chunk)
        WHERE n.start_time IS NOT NULL AND n.end_time IS NOT NULL
        {temporal_filter}
        {stream_filter}
        RETURN n
            ORDER BY n.start_time
            LIMIT {self.top_k or 10}
            """
        result = await self.graph_db.arun_cypher_query(query)
        return result

    async def retrieve_semantic_context(
        self,
        question: str,
        start_time: float = None,
        end_time: float = None,
        sort_by: str = None,
        stream_ids: List[str] = None,
    ) -> List[Dict]:
        """Retrieve semantically similar content using vector similarity search"""
        logger.info(
            f"Retrieving semantic context for question: {question} "
            f"between {start_time} and {end_time}"
        )
        if stream_ids:
            logger.info(f"Filtering by stream_ids: {stream_ids}")

        try:
            result = await self.doc_retriever.ainvoke(
                {"messages": [HumanMessage(content=question)]}
            )
            # logger.info(f"Semantic search results raw: {result}")
            processed_results = []
            for doc in result:
                # Filter by stream_id if specified
                if stream_ids:
                    doc_stream_id = doc.metadata.get("stream_id")
                    if doc_stream_id not in stream_ids:
                        logger.debug(f"Skipping document with stream_id '{doc_stream_id}' (not in {stream_ids})")
                        continue

                processed_results.append(
                    {
                        "n": {
                            "text": doc.page_content,
                            "start_time": doc.metadata.get("start_time", ""),
                            "end_time": doc.metadata.get("end_time", ""),
                            "chunkIdx": doc.metadata.get("chunkIdx", ""),
                            "score": doc.state.get("query_similarity_score", 0),
                            "stream_id": doc.metadata.get("stream_id", ""),
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
            logger.debug(f"Semantic search results: {processed_results}")
            return processed_results
        except Exception as e:
            logger.error(f"Error during semantic search: {e}")
            return []

    async def analyze_question(self, question: str) -> Dict[str, Any]:
        """Use LLM to analyze question and determine basic retrieval elements"""
        logger.info(f"Analyzing question: {question}")
        prompt = f"""Analyze this question and identify key elements for graph database retrieval.
        Question: {question}

        Identify and return as JSON:
        1. Entity types mentioned. Available entity types: {await self.get_all_entity_types()}
        2. Relationships of interest
        3. Location references
        4. Stream IDs mentioned. Available stream_ids: {await self.get_all_stream_ids()}
        5. Retrieval strategy (similarity, temporal)
            a. similarity: If the question needs to find similar content, return the retrieval strategy as similarity
            b. temporal: If the question is about a specific time range or time-based filtering, return the strategy as temporal

        Example question: "Between 30 seconds and 5 minutes ago, has the dog found the ball?"
        Example response:
        {{\
            "entity_types": ["Dog", "Ball"],\
            "relationships": ["DROPPED", "PICKED_UP"],\
            "location_references": ["backyard"],\
            "stream_ids": [],\
            "retrieval_strategy": "temporal"\
        }}\

        Example question with stream: "Summarize channel 3 over the last 5 minutes."
        Example response:
        {{\
            "entity_types": [],\
            "relationships": [],\
            "location_references": [],\
            "stream_ids": ["fm-radio-ch3"],\
            "retrieval_strategy": "temporal"\
        }}\

        Example question without time filtering: "What topics were discussed about dogs?"
        Example response:
        {{\
            "entity_types": ["Dog"],\
            "relationships": [],\
            "location_references": [],\
            "stream_ids": [],\
            "retrieval_strategy": "similarity"\
        }}\

        Output only valid JSON. Do not include any other text.
        """

        response = await self.chat_llm.ainvoke(prompt)
        logger.info("Question analysis complete")
        return remove_think_tags(response.content)

    async def analyze_temporal_strategy(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine the type of temporal retrieval strategy"""
        logger.info(f"Analyzing temporal strategy for question: {question}")
        prompt = f"""Analyze this question to determine the type of temporal retrieval strategy needed.
        Question: {question}

        Determine which type of temporal filtering is needed and return as JSON:
        1. "only_recent" - Questions about recent events (e.g., "over the last 5 minutes...")
        2. "excluding_recent" - Questions excluding recent events (e.g., "excluding the previous hour...")
        3. "specific_start_stop" - Questions with specific time ranges (e.g., "between 5 and 20 minutes ago...")
        4. "relative_time_past" - Questions about a specific point in the past (e.g., "what was the topic half an hour ago...")
        5. "specific_time" - Questions about a specific clock time (e.g., "what was the topic at 10:15 PM?")
        6. "none" - No temporal filtering needed

        Example question: "Over the last 5 minutes, what topics were discussed?"
        Example response:
        {{\
            "temporal_strategy": "only_recent"\
        }}\

        Example question: "Between 10 and 20 minutes ago, what happened?"
        Example response:
        {{\
            "temporal_strategy": "specific_start_stop"\
        }}\

        Example question: "What was being discussed half an hour ago?"
        Example response:
        {{\
            "temporal_strategy": "relative_time_past"\
        }}\

        Example question: "What was the topic at 2:30 PM?"
        Example response:
        {{\
            "temporal_strategy": "specific_time"\
        }}\

        Output only valid JSON. Do not include any other text.
        """

        response = await self.chat_llm.ainvoke(prompt)
        logger.info("Temporal strategy analysis complete")
        return remove_think_tags(response.content)

    async def analyze_temporal_times(self, question: str, temporal_strategy: str) -> Dict[str, Any]:
        """Analyze question to determine specific start and stop times based on temporal strategy"""
        logger.info(f"Analyzing temporal times for question: {question}, strategy: {temporal_strategy}")

        if temporal_strategy == "none":
            return {"start_time": None, "end_time": None}

        # Strategy-specific prompt templates
        strategy_prompts = {
            "only_recent": {
                "instruction": "Return how many seconds back to look from now.",
                "format": '{"seconds_back": 300}  // for "last 5 minutes"',
                "examples": [
                    '"last 10 minutes" → {"seconds_back": 600}',
                    '"over the past hour" → {"seconds_back": 3600}',
                    '"in the previous 30 seconds" → {"seconds_back": 30}'
                ]
            },
            "excluding_recent": {
                "instruction": "Return how many seconds to exclude from recent time.",
                "format": '{"seconds_to_exclude": 3600}  // for "excluding the previous hour"',
                "examples": [
                    '"excluding the last 5 minutes" → {"seconds_to_exclude": 300}',
                    '"not including the past hour" → {"seconds_to_exclude": 3600}',
                    '"ignoring the previous 2 minutes" → {"seconds_to_exclude": 120}'
                ]
            },
            "specific_start_stop": {
                "instruction": "Return start and stop times in seconds from now (start_seconds_ago should be larger than end_seconds_ago).",
                "format": '{"start_seconds_ago": 1200, "end_seconds_ago": 300}  // 20 minutes ago to 5 minutes ago',
                "examples": [
                    '"between 5 and 15 minutes ago" → {"start_seconds_ago": 900, "end_seconds_ago": 300}',
                    '"from 1 hour to 30 minutes ago" → {"start_seconds_ago": 3600, "end_seconds_ago": 1800}',
                    '"between 2 and 10 minutes ago" → {"start_seconds_ago": 600, "end_seconds_ago": 120}'
                ]
            },
            "relative_time_past": {
                "instruction": "Return the past point in seconds and optional window size (default 300 seconds if not specified).",
                "format": '{"past_point_seconds_ago": 1800, "window_seconds": 300}  // 30 minutes ago with 5 minute window',
                "examples": [
                    '"half an hour ago" → {"past_point_seconds_ago": 1800, "window_seconds": 300}',
                    '"what happened 10 minutes ago" → {"past_point_seconds_ago": 600, "window_seconds": 300}',
                    '"around 2 hours ago, with a 10 minute window" → {"past_point_seconds_ago": 7200, "window_seconds": 600}'
                ]
            },
            "specific_time": {
                "instruction": "Return the specific time. If AM/PM was specified, convert to HH:MM:SS format (24-hour format). If AM/PM was NOT specified, preserve the original time format and indicate AM/PM was not specified. Also include window if specified (default 300 seconds if not specified).",
                "format": '{"specific_time": "14:30:00", "am_pm_specified": true, "window_seconds": 300}  // for "2:30 PM"',
                "examples": [
                    '"at 3:15 PM" → {"specific_time": "15:15:00", "am_pm_specified": true}',
                    '"around 10:30 AM, plus/minus 10 minutes" → {"specific_time": "10:30:00", "am_pm_specified": true, "window_seconds": 600}',
                    '"at 9 o\'clock" → {"specific_time": "09:00:00", "am_pm_specified": false}',
                    '"at 2:45" → {"specific_time": "02:45:00", "am_pm_specified": false}',
                    '"around 11:30" → {"specific_time": "11:30:00", "am_pm_specified": false}'
                ]
            }
        }

        if temporal_strategy not in strategy_prompts:
            logger.error(f"Unknown temporal strategy: {temporal_strategy}")
            return {"start_time": None, "end_time": None}

        strategy_config = strategy_prompts[temporal_strategy]

        prompt = f"""Analyze this question to extract specific time values for the "{temporal_strategy}" temporal strategy.
        Question: {question}
        Temporal Strategy: {temporal_strategy}

        Task: {strategy_config["instruction"]}

        Expected JSON format:
        {strategy_config["format"]}

        Examples:
        {chr(10).join(f"- {example}" for example in strategy_config["examples"])}

        Output only valid JSON. Do not include any other text.
        """

        response = await self.chat_llm.ainvoke(prompt)
        logger.info("Temporal times analysis complete")
        return remove_think_tags(response.content)

    def _convert_temporal_times_to_timestamps(self, temporal_times: Dict, temporal_strategy: str) -> Dict[str, float]:
        """Convert temporal analysis results to actual start/end timestamps"""
        tnow = datetime.now(timezone.utc).timestamp()

        try:
            if temporal_strategy == "none":
                return {"start_time": None, "end_time": None}

            elif temporal_strategy == "only_recent":
                return {
                    "start_time": tnow - temporal_times["seconds_back"],
                    "end_time": None
                }

            elif temporal_strategy == "excluding_recent":
                return {
                    "start_time": None,
                    "end_time": tnow - temporal_times["seconds_to_exclude"]
                }

            elif temporal_strategy == "specific_start_stop":
                return {
                    "start_time": tnow - temporal_times["start_seconds_ago"],
                    "end_time": tnow - temporal_times["end_seconds_ago"]
                }

            elif temporal_strategy == "relative_time_past":
                window_seconds = temporal_times.get("window_seconds", 300)  # Default 5 min window
                return {
                    "start_time": tnow - temporal_times["past_point_seconds_ago"] - (window_seconds / 2),
                    "end_time": tnow - temporal_times["past_point_seconds_ago"] + (window_seconds / 2)
                }

            elif temporal_strategy == "specific_time":
                # Parse the specific time and calculate seconds ago from current time
                specific_time = temporal_times["specific_time"]
                window_seconds = temporal_times.get("window_seconds", 300)  # Default 5 min window
                am_pm_specified = temporal_times.get("am_pm_specified", False)

                # Parse the time string (HH:MM:SS format)
                time_obj = datetime.strptime(specific_time, "%H:%M:%S").time()
                now = datetime.now(timezone.utc)

                if am_pm_specified:
                    # AM/PM was specified, so the time is already correct in 24-hour format
                    # Just need to determine if it's today or yesterday
                    target_datetime = datetime.combine(now.date(), time_obj, timezone.utc)

                    # If the target time is in the future today, use yesterday
                    if target_datetime > now:
                        target_datetime = datetime.combine(now.date() - timedelta(days=1), time_obj, timezone.utc)
                        logger.info(f"Specific time {specific_time} (AM/PM specified) was in the future today, using yesterday")
                else:
                    # AM/PM was NOT specified, use AM if PM is in the future (today) and PM otherwise
                    # Create PM version using timedelta
                    pm_time_today = datetime.combine(now.date(), time_obj, timezone.utc)
                    if time_obj.hour < 12:
                        pm_time_today += timedelta(hours=12)

                    if pm_time_today > now:
                        # PM is in the future today, so use AM
                        am_time_today = datetime.combine(now.date(), time_obj, timezone.utc)
                        if am_time_today > now:
                            # AM is also in future today, use PM yesterday
                            target_datetime = pm_time_today - timedelta(days=1)
                            logger.info(f"Specific time {specific_time} (no AM/PM): PM in future, using PM yesterday")
                        else:
                            # AM is in past today, use it
                            target_datetime = am_time_today
                            logger.info(f"Specific time {specific_time} (no AM/PM): PM in future, using AM today")
                    else:
                        # PM is in past today, so use PM
                        target_datetime = pm_time_today
                        logger.info(f"Specific time {specific_time} (no AM/PM): PM in past, using PM today")

                past_point_seconds_ago = (now - target_datetime).total_seconds()
                logger.info(f"Specific time {specific_time} was {past_point_seconds_ago} seconds ago")

                return {
                    "start_time": tnow - past_point_seconds_ago - (window_seconds / 2),
                    "end_time": tnow - past_point_seconds_ago + (window_seconds / 2)
                }

        except Exception as e:
            logger.error(f"Error converting temporal times to timestamps: {e}")
            return {"start_time": None, "end_time": None}

        # Fallback
        return {"start_time": None, "end_time": None}

    async def retrieve_relevant_context(self, question: str) -> List[Document]:
        """Main retrieval method that orchestrates different retrieval strategies using 3-step analysis"""
        with TimeMeasure("AdvGraphRetrieval/retrieve_context", "blue"):
            logger.info(f"Starting context retrieval for question: {question}")

            # Step 1: Basic question analysis
            analysis = await self._parse_json_with_retries(self.analyze_question, "basic analysis", question)

            if not analysis:
                logger.error("Failed to parse basic analysis, using defaults")
                analysis = {
                    "entity_types": [],
                    "relationships": [],
                    "location_references": [],
                    "stream_ids": [],
                    "retrieval_strategy": "similarity",
                }

            # Get basic parameters from analysis
            strategy = analysis.get("retrieval_strategy", "")
            stream_ids = analysis.get("stream_ids", [])
            logger.info(f"Using retrieval strategy: {strategy}")

            # Step 2 & 3: Temporal analysis
            start_time = None
            end_time = None
            temporal_strategy = "none"

            # Step 2: Determine temporal strategy type
            temporal_strategy_analysis = await self._parse_json_with_retries(self.analyze_temporal_strategy, "temporal strategy", question)

            if temporal_strategy_analysis:
                temporal_strategy = temporal_strategy_analysis.get("temporal_strategy", "none")
                logger.info(f"Using temporal strategy: {temporal_strategy}")

                # Step 3: Determine specific times if not "none"
                if temporal_strategy != "none":
                    temporal_times = await self._parse_json_with_retries(self.analyze_temporal_times, "temporal times", question, temporal_strategy)

                    if temporal_times:
                        # Convert to actual timestamps
                        timestamps = self._convert_temporal_times_to_timestamps(temporal_times, temporal_strategy)
                        start_time = timestamps.get("start_time")
                        end_time = timestamps.get("end_time")
                        logger.info(f"Temporal range: {start_time} to {end_time}")

            # Collect context from retrieval strategies
            contexts = []

            if strategy == "temporal":
                temporal_data = await self.retrieve_temporal_context(
                    start_time, end_time, stream_ids
                )
                logger.info(f"Temporal Contexts...")
                if temporal_data:
                    contexts.extend(temporal_data)
                    logger.info(f"Retrieved {len(temporal_data)} temporal records")
                else:
                    logger.info("No temporal data found in that time range")
                    return None, (
                        datetime.utcfromtimestamp(start_time).strftime("%D %H:%M:%S") if start_time else "now",
                        datetime.utcfromtimestamp(end_time).strftime("%D %H:%M:%S") if end_time else "max",
                        stream_ids
                    )
            else:  # semantic retrieval
                # Semantic similarity retrieval
                semantic_data = await self.retrieve_semantic_context(
                    question,
                    start_time=start_time,
                    end_time=end_time,
                    sort_by=analysis.get("sort_by", "score"),
                    stream_ids=stream_ids,
                )
                logger.info(f"Semantic Contexts...")
                if semantic_data:
                    contexts.extend(semantic_data)

            logger.debug(f"Contexts: {contexts}")

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
                    logger.info(f"Relationship Data: {rel_data}")
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
                                "stream_id": ctx.get("n", {}).get("stream_id", ""),
                            },
                        )
                        documents.append(doc)

            logger.info(f"Returning {len(documents)} documents")
            return documents, None

    async def _parse_json_with_retries(self, analysis_func, analysis_type: str, *args, **kwargs) -> Dict:
        """Helper method to retry analysis function calls and parse JSON responses"""
        retry_count = 0

        while retry_count < self.max_retries:
            try:
                # Call the analysis function
                response = await analysis_func(*args, **kwargs)

                # Parse JSON from response
                json_start = response.find("{")
                json_end = response.rfind("}") + 1

                logger.info(f"{analysis_type} response (attempt {retry_count + 1}): {response}")

                if json_start >= 0 and json_end > json_start:
                    result = json.loads(response[json_start:json_end])
                    logger.info(f"Successfully parsed {analysis_type} JSON on attempt {retry_count + 1}")
                    return result
                else:
                    raise json.JSONDecodeError(f"No JSON found in {analysis_type} response", response, 0)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse {analysis_type} JSON response (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count < self.max_retries:
                    logger.info(f"Retrying {analysis_type} analysis (attempt {retry_count + 1}/{self.max_retries})")
                else:
                    logger.error(f"Max retries ({self.max_retries}) reached for {analysis_type}")
                    return None
            except Exception as e:
                logger.error(f"Unexpected error in {analysis_type} analysis (attempt {retry_count + 1}): {e}")
                retry_count += 1
                if retry_count >= self.max_retries:
                    logger.error(f"Max retries ({self.max_retries}) reached for {analysis_type}")
                    return None

        return None
