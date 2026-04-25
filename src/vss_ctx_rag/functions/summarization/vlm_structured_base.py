# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""vlm_structured_base.py: Shared base class and Event model for VLM structured summarization.

Contains all logic common to both the online (DB-backed) and offline (in-memory)
VLM structured summarization functions: Event parsing, merging, LLM aggregation,
batch storage, and result building.
"""

import json
import os
import re
from pathlib import Path
from typing import List, Optional

import json_repair

from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from pydantic import BaseModel, Field
from schema import Schema

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_SUMM_RECURSION_LIMIT,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.utils.utils import call_token_safe, remove_think_tags


# ── Shared Pydantic params base ────────────────────────────────────────


class VlmStructuredParamsBase(BaseModel):
    """Parameter schema shared by both DB-backed and in-memory configs."""

    uuid: Optional[str] = Field(
        default=None,
        description="For single-uuid processing. Falls back to ``['default']``.",
    )
    uuids: Optional[List[str]] = Field(
        default=None,
        description="One or more UUIDs to process. Falls back to *uuid* then ``['default']``.",
    )
    time_overlap_threshold: float = Field(
        default=0.1,
        ge=0.0,
        description="Minimum overlap duration in seconds to merge overlapping events",
    )
    time_adjacent_threshold: float = Field(
        default=4,
        ge=0.0,
        description="Maximum gap in seconds between events to merge adjacent events",
    )
    max_events_per_batch: int = Field(default=50, ge=1)
    enable_llm_merging: bool = Field(
        default=False,
        description="Enable LLM-based merging of descriptions for adjacent same-type events.",
    )
    kafka_enabled: bool = Field(
        default=False,
        description="When enabled, ES storage is handled externally by the kafka-consumer-service.",
    )
    start_time: Optional[float] = Field(
        default=None,
        description="If set, only events whose end_time >= this value are included.",
    )
    end_time: Optional[float] = Field(
        default=None,
        description="If set, only events whose start_time <= this value are included.",
    )


# ── Event model ─────────────────────────────────────────────────────────


class Event(BaseModel):
    """Represents a single event with time boundaries, type, and description."""

    start_time: float
    end_time: float
    type: str
    description: str
    uuid: Optional[str] = None

    def overlaps_with(
        self,
        other: "Event",
        overlap_threshold: float = 0.1,
        adjacent_threshold: float = 4,
    ) -> bool:
        """Check if this event overlaps or is adjacent to *other* and shares its type."""
        if self.type != other.type:
            return False

        overlap_start = max(self.start_time, other.start_time)
        overlap_end = min(self.end_time, other.end_time)
        overlap_duration = max(0, overlap_end - overlap_start)

        if overlap_duration > 0:
            return overlap_duration >= overlap_threshold

        return (
            abs(self.end_time - other.start_time) <= adjacent_threshold
            or abs(other.end_time - self.start_time) <= adjacent_threshold
        )

    def merge_with(self, other: "Event") -> "Event":
        """Simple concatenation merge (fallback when LLM merging is off)."""
        return Event(
            start_time=min(self.start_time, other.start_time),
            end_time=max(self.end_time, other.end_time),
            type=self.type,
            description=f"{self.description} | {other.description}",
            uuid=self.uuid,
        )


# ── Base function ───────────────────────────────────────────────────────


class VlmStructuredBase(Function):
    """Abstract base containing all shared VLM structured summarization logic.

    Subclasses must implement ``acall``, ``aprocess_doc``, and ``areset``.
    """

    config: dict
    db: StorageTool
    call_schema: Schema = Schema({}, ignore_extra_keys=True)
    metrics = SummaryMetrics()
    uuids: List[str]

    time_overlap_threshold: float
    time_adjacent_threshold: float
    max_events_per_batch: int
    enable_llm_merging: bool
    kafka_enabled: bool
    filter_start_time: Optional[float]
    filter_end_time: Optional[float]

    llm: BaseChatModel
    aggregation_pipeline: RunnableSequence
    description_merge_pipeline: RunnableSequence
    output_parser: StrOutputParser
    recursion_limit: int

    # ── setup ────────────────────────────────────────────────────────

    def setup(self):
        self.db = self.get_tool("db")

        self.time_overlap_threshold = self.get_param(
            "time_overlap_threshold", default=0.1
        )
        self.time_adjacent_threshold = self.get_param(
            "time_adjacent_threshold", default=4
        )
        self.max_events_per_batch = self.get_param("max_events_per_batch", default=50)
        self.enable_llm_merging = self.get_param("enable_llm_merging", default=False)
        self.kafka_enabled = self.get_param("kafka_enabled", default=False)
        self.filter_start_time = self.get_param("start_time", default=None)
        self.filter_end_time = self.get_param("end_time", default=None)

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.summary_start_time = None

        uuids_param = self.get_param("uuids", default=None)
        uuid_param = self.get_param("uuid", default=None)
        if uuids_param is not None:
            self.uuids = uuids_param if isinstance(uuids_param, list) else [uuids_param]
        elif uuid_param is not None:
            self.uuids = [uuid_param] if isinstance(uuid_param, str) else uuid_param
        else:
            self.uuids = ["default"]

        self.llm = self.get_tool(LLM_TOOL_NAME)
        self.recursion_limit = self.get_param(
            "summ_rec_lim", default=DEFAULT_SUMM_RECURSION_LIMIT
        )
        self._setup_aggregation_pipeline()

    def _setup_aggregation_pipeline(self) -> None:
        """Setup LangChain pipelines for event aggregation and description merging."""
        aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a professional analyst preparing an observational report. Your task is to synthesize "
                    "timestamped events into a formal, cohesive narrative. Follow these guidelines:\n"
                    "- Write in a neutral, objective tone appropriate for official documentation.\n"
                    "- Organize the narrative in chronological order, maintaining logical flow between events.\n"
                    "- Consolidate events occurring within fractions of a second into single, coherent statements.\n"
                    "- Omit raw timestamps from the final output; focus on the sequence and nature of observed activities.\n"
                    "- Use precise, descriptive language avoiding colloquialisms or informal expressions.\n"
                    "- Structure the summary with clear transitions to convey the progression of events.",
                ),
                (
                    "user",
                    "The following events have been recorded:\n\n{input}\n\n"
                    "Please synthesize these observations into a formal summary report:",
                ),
            ]
        )
        self.output_parser = StrOutputParser()
        self.aggregation_pipeline = (
            aggregation_prompt | self.llm | self.output_parser | remove_think_tags
        )

        description_merge_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an expert at combining related event descriptions into a single, coherent description. "
                    "Your task is to merge multiple descriptions of the same type of event into one unified description.\n"
                    "Guidelines:\n"
                    "- Preserve all important details from each description\n"
                    "- Remove redundant or duplicate information\n"
                    "- Maintain a consistent tone and style\n"
                    "- Keep the description concise but comprehensive\n"
                    "- Output ONLY the merged description, no additional text or explanation",
                ),
                (
                    "user",
                    "Event type: {event_type}\n\n"
                    "Descriptions to merge:\n{descriptions}\n\n"
                    "Merged description:",
                ),
            ]
        )
        self.description_merge_pipeline = (
            description_merge_prompt | self.llm | self.output_parser | remove_think_tags
        )

    # ── JSON parsing ────────────────────────────────────────────────────

    @staticmethod
    def _find_json_boundaries(content: str) -> tuple[int, int] | None:
        """Find the best JSON boundaries (array or object) in content."""
        candidates = [
            (content.find("["), content.rfind("]")),
            (content.find("{"), content.rfind("}")),
        ]
        valid = [(s, e) for s, e in candidates if s != -1 and e > s]
        return min(valid, key=lambda x: x[0]) if valid else None

    @classmethod
    def _extract_json_from_vlm_response(cls, vlm_response: str) -> str:
        """Extract and clean JSON from a VLM response string."""
        content = re.sub(
            r"```(?:json)?\s*\n(.*?)\n```", r"\1", vlm_response, flags=re.DOTALL
        )

        if not content.strip().startswith(("{", "[")):
            if bounds := cls._find_json_boundaries(content):
                content = content[bounds[0] : bounds[1] + 1]

        return content.strip().replace("\\n", "\n").replace('\\"', '"')

    @classmethod
    def _parse_json_document(cls, doc: str) -> List[Event]:
        """Parse a JSON document and return a list of Event objects."""
        try:
            json_content = cls._extract_json_from_vlm_response(doc)
            with Metrics("VlmStructured/ParseJSONDocument", "green"):
                data = json_repair.loads(json_content)

            logger.debug(f"Parsed and repaired JSON data: {data}")

            events_data = data.get("events", []) if isinstance(data, dict) else data
            if not isinstance(events_data, list):
                logger.warning(f"Expected events list, got {type(events_data)}")
                return []

            events = []
            for event_data in events_data:
                try:
                    event = Event(**event_data)
                    if event.start_time >= event.end_time:
                        logger.warning(
                            "Dropping zero/negative-duration event "
                            "(start_time=%.3f, end_time=%.3f): %s",
                            event.start_time,
                            event.end_time,
                            event.description,
                        )
                        continue
                    events.append(event)
                except Exception as e:
                    logger.warning(f"Skipping invalid event {event_data}: {e}")

            logger.info(f"Parsed {len(events)} events from JSON document")
            return events

        except Exception as e:
            logger.warning(f"Failed to parse JSON document: {e}")
            return []

    # ── Merging ─────────────────────────────────────────────────────────

    async def _merge_descriptions_with_llm(
        self, event_type: str, descriptions: List[str]
    ) -> str:
        """Use LLM to merge multiple event descriptions into one coherent description."""
        if len(descriptions) == 1:
            return descriptions[0]

        formatted_descriptions = "\n".join(
            f"{i + 1}. {desc}" for i, desc in enumerate(descriptions)
        )

        try:
            with Metrics("VlmStructured/MergeDescriptions", "yellow"):
                with get_openai_callback() as cb:
                    merged_description = await call_token_safe(
                        {
                            "event_type": event_type,
                            "descriptions": formatted_descriptions,
                        },
                        self.description_merge_pipeline,
                        self.recursion_limit,
                    )
                    logger.info(
                        f"LLM merged {len(descriptions)} descriptions for '{event_type}' event"
                    )
                    logger.info(
                        f"MergeDescriptions - Total Tokens: {cb.total_tokens}, "
                        f"Prompt Tokens: {cb.prompt_tokens}, "
                        f"Completion Tokens: {cb.completion_tokens}, "
                        f"Total Cost (USD): ${cb.total_cost}"
                    )
                return (
                    merged_description.strip()
                    if isinstance(merged_description, str)
                    else str(merged_description)
                )
        except Exception as e:
            logger.warning(
                f"Failed to merge descriptions with LLM: {e}. Falling back to simple concatenation."
            )
            return " | ".join(descriptions)

    async def _merge_similar_events(self, events: List[Event]) -> List[Event]:
        """Merge events based on time overlap/adjacency and same event type.

        Uses chain merging (A->B->C where B overlaps A, C overlaps B).
        """
        if not events:
            return events

        merged_events = []
        processed_indices = set()

        for i, event1 in enumerate(events):
            if i in processed_indices:
                continue

            events_to_merge = [event1]
            processed_indices.add(i)

            current_start = event1.start_time
            current_end = event1.end_time
            current_type = event1.type

            found_merge = True
            while found_merge:
                found_merge = False
                for j, event2 in enumerate(events):
                    if j in processed_indices:
                        continue

                    current_merged = Event(
                        start_time=current_start,
                        end_time=current_end,
                        type=current_type,
                        description="",
                    )

                    if current_merged.overlaps_with(
                        event2,
                        self.time_overlap_threshold,
                        self.time_adjacent_threshold,
                    ):
                        logger.info(
                            f"Will merge time-overlapping events of type '{current_type}': "
                            f"current [{current_start:.1f}-{current_end:.1f}] and "
                            f"'{event2.description[:50]}...' [{event2.start_time:.1f}-{event2.end_time:.1f}]"
                        )
                        events_to_merge.append(event2)
                        processed_indices.add(j)
                        current_start = min(current_start, event2.start_time)
                        current_end = max(current_end, event2.end_time)
                        found_merge = True

            if len(events_to_merge) == 1:
                merged_events.append(event1)
            else:
                descriptions = [e.description for e in events_to_merge]
                if self.enable_llm_merging:
                    merged_description = await self._merge_descriptions_with_llm(
                        current_type, descriptions
                    )
                else:
                    merged_description = " | ".join(descriptions)
                    logger.info(
                        f"LLM merging disabled - using simple concatenation for '{current_type}' event"
                    )

                merged_event = Event(
                    start_time=current_start,
                    end_time=current_end,
                    type=current_type,
                    description=merged_description,
                    uuid=events_to_merge[0].uuid,
                )
                merged_events.append(merged_event)
                logger.info(
                    f"Merged {len(events_to_merge)} events of type '{current_type}' "
                    f"into single event: {merged_description[:100]}..."
                )

        logger.info(f"Merged {len(events)} events into {len(merged_events)} events")
        merged_events.sort(key=lambda event: event.start_time)
        return merged_events

    # ── Time filtering ───────────────────────────────────────────────────

    @staticmethod
    def _filter_events_by_time(
        events: List[Event],
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> List[Event]:
        """Return only events that overlap the ``[start_time, end_time]`` window.

        An event is kept when its time span intersects the filter window, i.e.
        ``event.end_time >= start_time`` and ``event.start_time <= end_time``.
        If both bounds are *None* the original list is returned unchanged.
        """
        if start_time is None and end_time is None:
            return events

        filtered = []
        for event in events:
            if start_time is not None and event.end_time < start_time:
                continue
            if end_time is not None and event.start_time > end_time:
                continue
            filtered.append(event)

        logger.info(
            f"Time filter [{start_time}, {end_time}]: "
            f"{len(events)} events -> {len(filtered)} events"
        )
        return filtered

    # ── UUID resolution ────────────────────────────────────────────────

    def _resolve_uuids(self, state: dict) -> List[str]:
        """Return the effective UUID list from *state* (with config fallback).

        Accepts ``uuids`` (list) **or** the legacy ``uuid`` (str) key in
        *state*.  Falls back to ``self.uuids`` when neither is present.
        """
        uuids = state.get("uuids", None)
        if uuids is None:
            uuid = state.get("uuid", None)
            if uuid is not None:
                uuids = [uuid] if isinstance(uuid, str) else uuid
            else:
                uuids = self.uuids
        elif isinstance(uuids, str):
            uuids = [uuids]
        return uuids

    # ── Batch storage ───────────────────────────────────────────────────

    async def _store_merged_events(self, events: List[Event]) -> None:
        """Merge the given events and persist each batch to the database."""
        if not events:
            logger.info("No events to process")
            return

        logger.info(f"Processing {len(events)} events for storage")

        if self.kafka_enabled:
            logger.info(
                "Ready to merge %d events (merge deferred to caller, "
                "ES storage handled by kafka-consumer-service)",
                len(events),
            )
            return

        with Metrics("VlmStructured/StoreMergedEvents", "green"):
            merged_events = await self._merge_similar_events(events)

            multi = len(self.uuids) > 1

            grouped: dict[str, list[Event]] = {}
            for event in merged_events:
                key = event.uuid if multi and event.uuid else self.uuids[0]
                grouped.setdefault(key, []).append(event)

            for uuid_key, uuid_events in grouped.items():
                for i in range(0, len(uuid_events), self.max_events_per_batch):
                    batch_events = uuid_events[i : i + self.max_events_per_batch]
                    batch_json = {
                        "events": [
                            {
                                "start_time": event.start_time,
                                "end_time": event.end_time,
                                "type": event.type,
                                "description": event.description,
                                **(
                                    {"uuid": event.uuid} if multi and event.uuid else {}
                                ),
                            }
                            for event in batch_events
                        ]
                    }

                    batch_meta = {
                        "chunkIdx": -1,
                        "batch_i": i // self.max_events_per_batch,
                        "doc_type": "structured_events",
                        "uuid": uuid_key,
                        "camera_id": "default",
                        "event_count": len(batch_events),
                    }

                    batch_doc = json.dumps(batch_json, indent=2, ensure_ascii=False)
                    self.db.add_summary(summary=batch_doc, metadata=batch_meta)
                    logger.info(
                        f"Stored batch {i // self.max_events_per_batch} with "
                        f"{len(batch_events)} events for uuid '{uuid_key}'"
                    )

    # ── LLM aggregation ─────────────────────────────────────────────────

    async def _aggregate_events_with_llm(self, merged_events: List[Event]) -> str:
        """Aggregate merged events into a cohesive summary using the LLM."""
        events_text = []
        for event in merged_events:
            event_str = (
                f"- Time: {event.start_time}s to {event.end_time}s\n"
                f"  Type: {event.type}\n"
                f"  Description: {event.description}"
            )
            events_text.append(event_str)

        input_text = "\n\n".join(events_text)
        logger.info(f"Aggregating {len(merged_events)} events with LLM")

        with Metrics("VlmStructured/LLMAggregation", "cyan"):
            with get_openai_callback() as cb:
                aggregated_summary = await call_token_safe(
                    input_text,
                    self.aggregation_pipeline,
                    self.recursion_limit,
                )

                self.metrics.aggregation_tokens = cb.total_tokens
                logger.info(
                    f"Aggregation - Total Tokens: {cb.total_tokens}, "
                    f"Prompt Tokens: {cb.prompt_tokens}, "
                    f"Completion Tokens: {cb.completion_tokens}, "
                    f"Successful Requests: {cb.successful_requests}, "
                    f"Total Cost (USD): ${cb.total_cost}"
                )

        return aggregated_summary

    # ── Result building ─────────────────────────────────────────────────

    async def _build_result(
        self,
        state: dict,
        events: List[Event],
        uuids: List[str],
        log_filename: str = "structured_events_metrics.json",
    ) -> dict:
        """Merge *events*, aggregate via LLM, and populate *state* with the result JSON.

        This is the shared tail of ``acall`` for both DB-backed and in-memory variants.
        """
        multi = len(uuids) > 1

        if events:
            merged_events = await self._merge_similar_events(events)

            aggregated_summary = await self._aggregate_events_with_llm(merged_events)
            if not aggregated_summary.strip():
                aggregated_summary = "No events detected"

            events_list = [
                {
                    "id": idx + 1,
                    "start_time": event.start_time,
                    "end_time": event.end_time,
                    "type": event.type,
                    "description": event.description,
                    **({"uuid": event.uuid} if multi and event.uuid else {}),
                }
                for idx, event in enumerate(merged_events)
            ]

            result_json = {
                "events": events_list,
                "total_events": len(merged_events),
                "video_summary": aggregated_summary,
                "uuids": uuids,
            }
            state["result"] = json.dumps(result_json, indent=2, ensure_ascii=False)
            logger.info(
                f"Processed {len(events)} events into {len(merged_events)} merged events"
            )
            logger.info(f"Aggregated summary: {aggregated_summary}")
        else:
            state["result"] = json.dumps(
                {
                    "events": [],
                    "total_events": 0,
                    "video_summary": "",
                    "uuids": uuids,
                },
                indent=2,
                ensure_ascii=False,
            )
            logger.info("No events to process")

        state["metadata"] = self.metrics.dump_dict()

        if self.log_dir:
            log_path = Path(self.log_dir).joinpath(log_filename)
            self.metrics.dump_json(log_path.absolute())

        return state

    # ── Doc ingestion helper ────────────────────────────────────────────

    def _store_raw_events(
        self, events: List[Event], doc_i: int, doc_meta: dict
    ) -> None:
        """Serialize *events* as raw JSON and persist to the database.

        When multiple UUIDs are configured, each event dict includes its
        ``uuid`` field so the source stream is preserved in the stored JSON.
        """
        multi = len(self.uuids) > 1
        event_dicts = []
        for e in events:
            d = e.model_dump(exclude_none=True)
            if not multi:
                d.pop("uuid", None)
            event_dicts.append(d)

        raw_events_json = json.dumps(
            {"events": event_dicts},
            indent=2,
            ensure_ascii=False,
        )
        raw_meta = {
            "chunkIdx": doc_meta.get("chunkIdx", doc_i),
            "doc_type": "raw_events",
            "uuid": doc_meta.get("uuid", self.uuids[0]),
            "camera_id": doc_meta.get("camera_id", "default"),
            "event_count": len(events),
            "batch_i": doc_meta.get("batch_i", doc_i),
        }
        self.db.add_summary(summary=raw_events_json, metadata=raw_meta)
        logger.info(f"Stored {len(events)} raw events for chunk {raw_meta['chunkIdx']}")
