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

"""Structured inference function for extracting structured events from text.

This module provides the StructuredInference function class that processes documents
in batches to extract structured events using LLM-based inference with configurable
schemas and prompts.
"""

import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from pydantic import Field
from schema import Schema

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher, Batch
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_SUMM_RECURSION_LIMIT,
    DEFAULT_SUMM_TIMEOUT_SEC,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.utils.utils import call_token_safe
from vss_ctx_rag.functions.summarization.config import SummarizationConfig
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from jsonschema import validate, ValidationError
import json_repair


@register_function_config("structured_inference")
class StructuredInferenceConfig(SummarizationConfig):
    class StructuredInferenceParams(SummarizationConfig.SummarizationParams):
        # Make prompts optional for structured inference since it uses its own prompt system
        prompts: Optional["SummarizationConfig.Prompts"] = None
        scenario: str = Field(default="warehouse")
        events: List[str] = Field(default_factory=list)
        schema: Optional[str] = Field(
            default="""
                                     {
                                        "title": "EventExtraction",
                                        "description": "Extract structured events from video captions",
                                        "type": "object",
                                        "properties": {
                                            "events": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "start_time": { "type": "number" },
                                                        "end_time": { "type": "number" },
                                                        "description": { "type": "string" },
                                                        "type": { "type": "string" }
                                                    },
                                                    "required": ["start_time", "end_time", "description", "type"]
                                                }
                                            }
                                        },
                                        "required": ["events"]
                                    }"""
        )
        auto_generate_prompt: bool = Field(default=False)
        prompt: Optional[str] = Field(default=None)
        batch_response_method: str = Field(default="json_mode")
        time_metadata_keys: List[str] = Field(
            default_factory=lambda: ["start_pts", "end_pts"]
        )
        time_threshold: float = Field(
            default=10.0,
            description="Time threshold in seconds for merging temporally close events of the same type",
        )

    params: StructuredInferenceParams


@register_function(config=StructuredInferenceConfig)
class StructuredInference(Function):
    """Structured inference function for extracting events from documents.

    This function processes documents in batches, applies structured inference using
    an LLM with a specified schema, and aggregates results. It supports custom prompts
    and automatic prompt generation based on scenarios and event types.

    Attributes:
        config: Function configuration dictionary
        structured_prompt: Prompt template for structured inference
        aggregation_prompt: Prompt template for aggregating batch results
        output_parser: Parser for LLM output
        batch_size: Number of documents per batch
        batch_pipeline: LangChain pipeline for batch processing
        aggregation_pipeline: LangChain pipeline for result aggregation
        db: Storage tool for database operations
        batcher: Batcher instance for managing document batches
        llm: Language model instance
        metrics: Summary metrics tracker
        uuid: Unique identifier for the processing session
        scenario: Processing scenario (e.g., "warehouse")
        schema: JSON schema for structured output
        events: List of event types to extract
        batch_response_method: Method for structured output ("json_mode", etc.)
        time_metadata_keys: Keys for time metadata in documents
        time_threshold: Threshold for merging temporally close events (seconds)
        summary_start_time: Timestamp when summary processing started
        recursion_limit: Maximum recursion depth for LLM calls
        log_dir: Directory for logging metrics
        timeout: Timeout for processing operations (seconds)
        call_schema: Schema validator for function calls
    """

    # Core attributes - initialized in setup()
    config: Dict[str, Any]
    structured_prompt: str
    aggregation_prompt: ChatPromptTemplate
    output_parser: StrOutputParser
    batch_size: int
    batch_pipeline: RunnableSequence
    aggregation_pipeline: RunnableSequence
    db: StorageTool
    batcher: Batcher
    llm: BaseChatModel
    metrics: SummaryMetrics
    uuid: str

    # Configuration parameters
    scenario: str
    schema: str
    events: List[str]
    batch_response_method: str
    time_metadata_keys: List[str]
    time_threshold: float
    summary_start_time: Optional[float]
    recursion_limit: int
    log_dir: Optional[str]

    # Constants
    timeout: int = DEFAULT_SUMM_TIMEOUT_SEC
    call_schema: Schema = Schema(
        {"start_index": int, "end_index": int}, ignore_extra_keys=True
    )

    def setup(self) -> None:
        """Initialize the structured inference function with configuration parameters."""
        # Initialize core parameters
        self.batch_size = self.get_param("batch_size", default=6)
        self.batcher = Batcher(self.batch_size)
        self.scenario = self.get_param("scenario", default="warehouse")
        self.schema = self.get_param("schema")
        self.events = self.get_param("events", default=[])
        self.llm = self.get_tool(LLM_TOOL_NAME)
        self.batch_response_method = self.get_param(
            "batch_response_method", default="json_mode"
        )
        self.time_metadata_keys = self.get_param(
            "time_metadata_keys", default=["start_pts", "end_pts"]
        )
        self.time_threshold = self.get_param("time_threshold", default=10.0)
        self.db = self.get_tool("db")
        self.summary_start_time = None
        self.uuid = self.get_param("uuid", default="default")
        self.recursion_limit = self.get_param(
            "summ_rec_lim", default=DEFAULT_SUMM_RECURSION_LIMIT
        )
        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.metrics = SummaryMetrics()

        # Setup prompts and pipelines
        self._setup_structured_prompt()
        self._setup_pipelines()

    def _escape_schema_for_prompt(self, schema: str) -> str:
        """Escape curly braces in schema for use in ChatPromptTemplate.

        Args:
            schema: JSON schema string

        Returns:
            Schema string with escaped braces
        """
        return schema.replace("{", "{{").replace("}", "}}")

    def _generate_prompt_with_llm(self, escaped_schema: str) -> str:
        """Generate a structured inference prompt using the LLM.

        Args:
            escaped_schema: Schema string with escaped braces

        Returns:
            Generated prompt string with escaped braces
        """
        logger.info(
            f"Generating structured prompt for scenario: {self.scenario} and events: {self.events}"
        )

        system_message = (
            "You are a helpful assistant that generates a prompt for structured inference. "
            "\n\nBackground: The generated prompt will be used to extract any events that are relevant to the scenario. "
            "If no events are provided then the generated prompt should be able to extract interesting events and not "
            "mundane regular events. They should contain high level information. For example, in case of a warehouse, "
            "fire, theft, accident, etc. are interesting events whereas workers working in orderly manner is not an "
            "interesting event. In case of a traffic monitoring video, collision, unsafe maneuver, traffic violation, "
            "obstructed traffic flow etc. are interesting events and normal traffic flow is not an interesting event."
        )

        user_message = (
            f"Given a scenario and events, generate a prompt that will be used to generate structured output.\n"
            f"Scenario: {self.scenario}\n"
            f"Events: {self.events}\n"
            f"Required schema for the structured output: {escaped_schema}\n"
            f"Prompt: "
        )

        generated_prompt = self.llm.invoke(
            [("system", system_message), ("user", user_message)]
        )
        # Escape curly braces in the generated prompt for ChatPromptTemplate
        return str(generated_prompt.content).replace("{", "{{").replace("}", "}}")

    def _get_default_prompt(self, escaped_schema: str) -> str:
        """Get the default structured inference prompt.

        Args:
            escaped_schema: Schema string with escaped braces

        Returns:
            Default prompt string
        """
        return (
            f"Extract {self.scenario} related events from the list of event types: {self.events} from the captions of "
            f"a {self.scenario} monitoring video. Return a structured json output as per the "
            f"schema provided. Schema: {escaped_schema}. Ensure the event type is one of the event types in the list."
        )

    def _setup_structured_prompt(self) -> None:
        """Setup the structured inference prompt based on configuration."""
        escaped_schema = self._escape_schema_for_prompt(self.schema)

        if self.get_param("auto_generate_prompt", default=False):
            self.structured_prompt = self._generate_prompt_with_llm(escaped_schema)
        else:
            prompt_value = self.get_param("prompts", {}).get("event_extraction_prompt")
            if prompt_value is None:
                prompt_value = self._get_default_prompt(escaped_schema)
            logger.info(f"Using prompt: {prompt_value}")
            self.structured_prompt = prompt_value

        logger.info(f"Structured prompt: {self.structured_prompt}")

    def _setup_pipelines(self) -> None:
        """Setup LangChain pipelines for batch processing and aggregation."""
        # Setup batch processing pipeline
        structured_batch_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.structured_prompt),
                ("user", "{input}"),
            ]
        )
        logger.info(f"Structured batch prompt: {structured_batch_prompt}")

        self.output_parser = StrOutputParser()
        self.batch_pipeline = structured_batch_prompt | self.llm.with_structured_output(
            method=self.batch_response_method,
            schema=json.loads(self.schema),
        )

        # Setup aggregation pipeline
        self.aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant that aggregates structured events from multiple batches.",
                ),
                (
                    "user",
                    "Aggregate the following batch summaries and provide an overall summary:\n{input} Summary:",
                ),
            ]
        )
        self.aggregation_pipeline = (
            self.aggregation_prompt | self.llm | self.output_parser
        )

    def _compute_timestamp_bounds(self, batch: Batch) -> Dict[str, Optional[float]]:
        """Compute min/max timestamp bounds for all timestamp fields in the batch.

        Args:
            batch: Batch containing documents with metadata

        Returns:
            Dictionary mapping timestamp field names to their min/max values
        """
        timestamp_fields = [
            "start_pts",
            "end_pts",
            "start_ntp",
            "end_ntp",
            "start_ntp_float",
            "end_ntp_float",
        ]
        bounds = {field: None for field in timestamp_fields}

        for _, _, meta in batch.as_list():
            for field in timestamp_fields:
                value = meta.get(field)
                if value is not None:
                    if bounds[field] is None:
                        bounds[field] = value
                    elif field.startswith("start_"):
                        bounds[field] = min(bounds[field], value)
                    else:  # end_ fields
                        bounds[field] = max(bounds[field], value)

        logger.info(
            f"Timestamp bounds - pts: [{bounds['start_pts']}, {bounds['end_pts']}], "
            f"ntp: [{bounds['start_ntp']}, {bounds['end_ntp']}], "
            f"ntp_float: [{bounds['start_ntp_float']}, {bounds['end_ntp_float']}]"
        )

        return bounds

    async def _process_full_batch(self, batch: Batch) -> None:
        """Process a full batch of documents and store the summary.

        This method:
        1. Combines documents in the batch into text
        2. Generates a structured summary using the LLM pipeline
        3. Extracts metadata and timestamp bounds from batch documents
        4. Stores the summary with metadata in the database

        Args:
            batch: Full batch ready for processing
        """
        batch_index = batch.get_batch_index()
        is_last = batch.as_list()[-1][2].get("is_last", False)

        with Metrics(f"Batch {batch_index} Summary IS LAST {is_last}", "pink"):
            batch_summary = "{}"  # Default empty JSON

            logger.info("Batch %d is full. Processing ...", batch_index)
            logger.info(f"Batch: {batch}")

            # Generate batch summary using LLM
            try:
                with get_openai_callback() as cb:
                    batch_text = " ".join([doc for doc, _, _ in batch.as_list()])
                    logger.info(
                        f"Batch Text before calling batch pipeline: {batch_text}"
                    )
                    logger.debug(f"{self.batch_pipeline.__dict__}")
                    batch_summary = await call_token_safe(
                        {"input": batch_text},
                        self.batch_pipeline,
                        self.recursion_limit,
                    )
                    logger.info(f"Batch pipeline output: {batch_summary}")
                    # Convert dict to JSON string if needed
                    batch_summary = json.dumps(batch_summary, ensure_ascii=False)
                    logger.info(
                        f"Batch Summary: {batch_summary}, type: {type(batch_summary)}"
                    )
            except Exception as e:
                traceback.print_exc()
                logger.error(f"Error summarizing batch {batch_index}: {e}")

            # Update metrics
            self.metrics.summary_tokens += cb.total_tokens
            self.metrics.summary_requests += cb.successful_requests

            # Extract metadata and chunk indices from batch
            chunk_indices = []
            doc_meta_sample = None
            for _, _, doc_meta in batch.as_list():
                if doc_meta_sample is None:
                    doc_meta_sample = doc_meta
                if "chunkIdx" in doc_meta and doc_meta["chunkIdx"] is not None:
                    chunk_indices.append(doc_meta["chunkIdx"])

            # Remove duplicate chunk indices
            if chunk_indices:
                chunk_indices = list(set(chunk_indices))

            # Compute timestamp bounds across all documents in batch
            timestamp_bounds = self._compute_timestamp_bounds(batch)

            # Log summary and token usage
            logger.info("Batch %d summary: %s", batch_index, batch_summary)
            logger.info(
                "Total Tokens: %s, Prompt Tokens: %s, Completion Tokens: %s, "
                "Successful Requests: %s, Total Cost (USD): $%s",
                cb.total_tokens,
                cb.prompt_tokens,
                cb.completion_tokens,
                cb.successful_requests,
                cb.total_cost,
            )
        # Store the batch summary in database
        try:
            batch_meta = self._build_batch_metadata(
                doc_meta_sample, batch_index, timestamp_bounds, chunk_indices
            )

            # TODO: Use async method once https://github.com/langchain-ai/langchain-milvus/pull/29 is released
            # await self.db.aadd_summary(summary=batch_summary, metadata=batch_meta)
            if batch_summary is not None:
                logger.debug(f"Metadata being added: {batch_meta}")
                logger.info(f"Batch Summary: {batch_summary}")
            self.db.add_summary(summary=str(batch_summary), metadata=batch_meta)
        except Exception as e:
            traceback.print_exc()
            logger.error(f"Error adding summary to database: {e}")

    def _build_batch_metadata(
        self,
        doc_meta_sample: Optional[Dict[str, Any]],
        batch_index: int,
        timestamp_bounds: Dict[str, Optional[float]],
        chunk_indices: List[int],
    ) -> Dict[str, Any]:
        """Build metadata dictionary for a batch summary.

        Args:
            doc_meta_sample: Sample document metadata to extract schema
            batch_index: Index of the batch being processed
            timestamp_bounds: Computed timestamp bounds for the batch
            chunk_indices: List of chunk indices in the batch

        Returns:
            Complete metadata dictionary for the batch summary
        """
        # Create empty metadata template from sample document
        empty_doc_meta = {}
        if doc_meta_sample:
            empty_doc_meta = {
                key: type(value)() for key, value in doc_meta_sample.items()
            }

        # Build base metadata
        batch_meta = {
            **empty_doc_meta,
            "chunkIdx": -1,
            "batch_i": batch_index,
            "doc_type": "caption_summary",
            "uuid": self.uuid,
            "camera_id": "default",
        }

        # Add available timestamp fields
        batch_meta.update({k: v for k, v in timestamp_bounds.items() if v is not None})

        # Add linked chunk indices if any exist
        if chunk_indices:
            batch_meta["linked_summary_chunks"] = chunk_indices

        return batch_meta

    async def aprocess_doc(
        self, doc: str, doc_i: int, doc_meta: Dict[str, Any]
    ) -> None:
        """Process a document by adding it to a batch and processing when full.

        Args:
            doc: Document text to process
            doc_i: Document index
            doc_meta: Document metadata dictionary
        """
        try:
            logger.info("Adding doc %d", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            with Metrics("summ/aprocess_doc", "red") as bs:
                doc_meta["batch_i"] = doc_i // self.batch_size
                batch = self.batcher.add_doc(doc, doc_i, doc_meta)
                if batch.is_full():
                    # Process the batch immediately when full
                    logger.info(f"Batch: {batch}")
                    await asyncio.create_task(self._process_full_batch(batch))

            # Track timing metrics
            if self.summary_start_time is None:
                self.summary_start_time = bs.start_time
            self.metrics.summary_latency = bs.end_time - self.summary_start_time
        except Exception as e:
            logger.error(f"Error processing document {doc_i}: {e}")

    async def acall(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate batch summaries into a final summary with extracted events.

        This method retrieves processed batch summaries from the database,
        aggregates them using the LLM, and extracts structured events.

        Args:
            state: Dictionary containing:
                - start_index (int): Starting document index
                - end_index (int): Ending document index

        Returns:
            dict: State dictionary with added fields:
                - result (str): JSON string with "video_summary" and "events"
                - error_code (str): Error message if any (optional)

        Raises:
            Exception: If batch summarization fails
        """
        try:
            logger.info(f"Batch Summarization Acall: {state}")
            with Metrics("OffBatchSumm/Acall", "blue"):
                # Validate input state
                self.call_schema.validate(state)

                # Retrieve batch summaries from database
                batches = await self._retrieve_batch_summaries(
                    state["start_index"], state["end_index"]
                )

                # Process batches and generate final result
                if not batches:
                    state["result"] = ""
                    state["error_code"] = "No batch summaries found"
                    logger.error("No batch summaries found")
                else:
                    # Aggregate batches and extract events
                    await self._aggregate_batches(state, batches)
                state["metadata"] = self.metrics.dump_dict()

            # Save metrics if log directory is configured
            if self.log_dir:
                log_path = Path(self.log_dir) / "summary_metrics.json"
                self.metrics.dump_json(log_path.absolute())
        except Exception as e:
            logger.error(f"Error in batch summarization: {e}")
            state["error_code"] = f"{e}"
            raise e
        return state

    async def _retrieve_batch_summaries(
        self, start_index: int, end_index: int
    ) -> List[Dict[str, Any]]:
        """Retrieve batch summaries from database with retry logic.

        Args:
            start_index: Starting document index
            end_index: Ending document index

        Returns:
            List of batch summary dictionaries
        """
        target_start_batch_index = self.batcher.get_batch_index(start_index)
        target_end_batch_index = self.batcher.get_batch_index(end_index)

        logger.info(f"Target Batch Start: {target_start_batch_index}")
        logger.info(f"Target Batch End: {target_end_batch_index}")

        # Handle -1 end index (retrieve all batches)
        if target_end_batch_index == -1:
            max_batch_index = await self.db.aget_max_batch_index(self.uuid)
            target_end_batch_index = max_batch_index
            logger.debug(f"Updated target_end_batch_index to {target_end_batch_index}")

        expected_batch_count = target_end_batch_index - target_start_batch_index + 1
        stop_time = time.time() + self.timeout

        # Retry logic to wait for all batches to be available
        while time.time() < stop_time:
            batches = await self.db.aget_text_data(
                target_start_batch_index, target_end_batch_index, self.uuid
            )
            batches.sort(key=lambda x: x["batch_i"])

            logger.debug(
                f"Batches Fetched: {[{k: v for k, v in batch.items() if k != 'vector'} for batch in batches]}"
            )
            logger.info(f"Number of Batches Fetched: {len(batches)}")

            if len(batches) == expected_batch_count:
                logger.info(f"Need {expected_batch_count} batches. Moving forward.")
                break
            elif len(batches) >= expected_batch_count:
                logger.info(
                    f"Found {len(batches)} batches. Taking first {expected_batch_count} batches."
                )
                batches = batches[:expected_batch_count]
                break
            else:
                logger.info(f"Need {expected_batch_count} batches. Waiting ...")
                await asyncio.sleep(1)

        # Extract only text field for aggregation
        batches = [
            {k: v for k, v in batch.items() if k in ["text"]} for batch in batches
        ]
        logger.debug(f"Batches for aggregation: {batches}")

        return batches

    async def _aggregate_batches(
        self, state: Dict[str, Any], batches: List[Dict[str, Any]]
    ) -> None:
        """Aggregate batch summaries and extract events.

        Args:
            state: State dictionary to update with results
            batches: List of batch summary dictionaries
        """
        with Metrics("summ/acall/batch-aggregation-summary", "pink") as bas:
            with get_openai_callback() as cb:
                # Generate aggregated summary
                aggregation_result = await call_token_safe(
                    batches, self.aggregation_pipeline, self.recursion_limit
                )

                # Extract structured events from batches
                events = self.extract_events_from_batches(batches)

                # Combine results
                state["result"] = json.dumps(
                    {"video_summary": aggregation_result, "events": events},
                    ensure_ascii=False,
                )

            logger.info("Summary Aggregation Done")
            self.metrics.aggregation_tokens = cb.total_tokens
            logger.info(
                "Total Tokens: %s, Prompt Tokens: %s, Completion Tokens: %s, "
                "Successful Requests: %s, Total Cost (USD): $%s",
                cb.total_tokens,
                cb.prompt_tokens,
                cb.completion_tokens,
                cb.successful_requests,
                cb.total_cost,
            )
        self.metrics.aggregation_latency = bas.execution_time

    def extract_events_from_batches(
        self, batches: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Extract and merge events from batch summaries.

        Parses JSON batch summaries to extract events, then merges temporally close
        events of the same type based on the configured time threshold.

        Args:
            batches: List of batch dictionaries containing "text" field with JSON

        Returns:
            List of merged event dictionaries
        """
        result = []
        logger.info("Starting to extract events from batches.")

        # Extract events from all batches
        for batch in batches:
            text = batch.get("text")
            if text and text != "None":
                logger.info(f"Batch Text: {text}")
                try:
                    with Metrics("StructuredBatchSumm/ParseJSONDocument", "green"):
                        events_object = json_repair.loads(text)
                    logger.debug(f"Parsed and repaired JSON data: {events_object}")

                    validate(instance=events_object, schema=json.loads(self.schema))

                    events = events_object.get("events", [])
                    result.extend(event for event in events if event)
                except ValidationError as e:
                    logger.warning(
                        f"Validation error: {e.message}. Schema: {self.schema}. Skipping."
                    )
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse events from batch: {e}")

        # Merge temporally close events of the same type
        if result:
            result = self._merge_temporal_events(result)

        return result

    def _merge_temporal_events(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Merge temporally close events of the same type.

        Args:
            events: List of event dictionaries with type, start_time, end_time, description

        Returns:
            List of merged event dictionaries
        """
        # Sort by type then by start_time
        events.sort(key=lambda e: (e.get("type", ""), e.get("start_time", 0)))

        combined = []
        for event in events:
            # Check if this event should be merged with the previous one
            should_merge = (
                combined
                and event.get("type") == combined[-1].get("type")
                and (event.get("start_time", 0) - combined[-1].get("end_time", 0))
                <= self.time_threshold
            )

            if should_merge:
                logger.info(f"Merging events: {combined[-1]} and {event}")
                # Expand the time range
                combined[-1]["end_time"] = max(
                    combined[-1].get("end_time", 0), event.get("end_time", 0)
                )
                # Combine descriptions if different
                if event.get("description") != combined[-1].get("description"):
                    logger.info(
                        f"Combining descriptions: {combined[-1]['description']} and {event['description']}"
                    )
                    combined[-1]["description"] = (
                        combined[-1]["description"] + " " + event["description"]
                    )
            else:
                combined.append(event.copy())

        return combined

    async def areset(self, state: Dict[str, Any]) -> None:
        """Reset the function state and clear all accumulated data.

        Args:
            state: State dictionary (passed to database reset)
        """
        # TODO: use async method for drop data
        self.db.reset(state)
        self.summary_start_time = None
        self.batcher.flush()
        self.metrics.reset()
        await asyncio.sleep(0.001)
