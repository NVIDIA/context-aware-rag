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

import asyncio
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple, Union

import json_repair

from langchain_community.callbacks import get_openai_callback
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from pydantic import BaseModel, Field, field_validator
from schema import Schema

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_SUMM_RECURSION_LIMIT,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.utils.utils import (
    call_token_safe,
    remove_think_tags,
    split_top_level_json_values,
)


# ── Shared Pydantic params base ────────────────────────────────────────


DEFAULT_AGGREGATION_PROMPT = (
    "You are a professional analyst preparing an observational report. Your task is to synthesize "
    "timestamped events into a formal, cohesive narrative. Follow these guidelines:\n"
    "- Write in a neutral, objective tone appropriate for official documentation.\n"
    "- Organize the narrative in chronological order, maintaining logical flow between events.\n"
    "- Consolidate events occurring within fractions of a second into single, coherent statements.\n"
    "- Omit raw timestamps from the final output; focus on the sequence and nature of observed activities.\n"
    "- Use precise, descriptive language avoiding colloquialisms or informal expressions.\n"
    "- Structure the summary with clear transitions to convey the progression of events."
)

DEFAULT_DESCRIPTION_MERGE_PROMPT = (
    "You are an expert at combining related event descriptions into a single, coherent description. "
    "Your task is to merge multiple descriptions of the same type of event into one unified description.\n"
    "Guidelines:\n"
    "- Preserve all important details from each description\n"
    "- Remove redundant or duplicate information\n"
    "- Maintain a consistent tone and style\n"
    "- Keep the description concise but comprehensive\n"
    "- Output ONLY the merged description, no additional text or explanation"
)


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
        description=(
            "Enable LLM-based merging of descriptions for adjacent same-type events. "
            "Also enabled when the LVS_ENABLE_LLM_MERGING environment variable is true/1/yes."
        ),
    )
    aggregation_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional system prompt for final event aggregation. "
            "When unset or empty, the built-in observational-report prompt is used. "
            "The user message always provides events via the {input} placeholder."
        ),
    )
    description_merge_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Optional system prompt for LLM description merging. Used only when "
            "enable_llm_merging is true or LVS_ENABLE_LLM_MERGING is enabled. "
            "When unset or empty, the built-in merge prompt is used. The user message "
            "always provides {event_type} and {descriptions}."
        ),
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


# ── Timestamp parsing ───────────────────────────────────────────────────


def _parse_timestamp(value: Union[int, float, str]) -> float:
    """Coerce a timestamp to float seconds.

    Accepts:
      - Numeric values (int / float) — returned as-is.
      - Numeric strings (e.g. ``"123.45"``) — converted via ``float()``.
      - Video-relative timestamps in ``MM:SS`` or ``HH:MM:SS`` form —
        converted to elapsed seconds. These measure elapsed time rather than
        wall-clock time, so hours are not capped at 24 (the two-digit fields
        allow up to ``99:59:59``) and cannot be negative. A malformed
        component (e.g. ``"00:60"``) raises ``ValueError`` rather than
        falling back to ISO parsing.
      - ISO 8601 strings (e.g. ``"2025-01-15T10:30:00Z"``) — parsed and
        converted to a POSIX / epoch-seconds float.

    Raises:
        ValueError: If *value* is a string that matches none of the accepted
            forms, or is a duration with an out-of-range component.
        TypeError: If *value* is neither numeric nor a string (e.g. ``None``).
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass

        duration_match = re.fullmatch(
            r"(?:(?P<hours>\d{2}):)?"
            r"(?P<minutes>\d{2}):"
            r"(?P<seconds>\d{2}(?:\.\d+)?)",
            value,
        )
        if duration_match:
            hours_group = duration_match.group("hours")
            hours = int(hours_group) if hours_group is not None else 0
            minutes = int(duration_match.group("minutes"))
            seconds = float(duration_match.group("seconds"))
            # In MM:SS form the leading field is a total minute count, so it may
            # exceed 59; once hours are given, minutes must wrap normally.
            if seconds >= 60 or (hours_group is not None and minutes >= 60):
                raise ValueError(
                    f"Cannot parse timestamp: {value!r}; seconds must be < 60 and "
                    "minutes must be < 60 when hours are present"
                )
            return hours * 3600 + minutes * 60 + seconds

        normalized = value.replace("Z", "+00:00") if value.endswith("Z") else value
        try:
            dt = datetime.fromisoformat(normalized)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except ValueError:
            raise ValueError(f"Cannot parse timestamp: {value!r}")
    raise TypeError(
        "Expected numeric, MM:SS, HH:MM:SS, or ISO timestamp, "
        f"got {type(value).__name__}"
    )


# ── Event model ─────────────────────────────────────────────────────────


class Event(BaseModel):
    """Represents a single event with time boundaries, type, and description."""

    start_time: float
    end_time: float
    type: str
    description: str
    uuid: Optional[str] = None

    @field_validator("start_time", "end_time", mode="before")
    @classmethod
    def _coerce_timestamp(cls, v: Union[int, float, str]) -> float:
        return _parse_timestamp(v)

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
    metrics: SummaryMetrics
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
    description_merge_pipeline: Optional[RunnableSequence]
    output_parser: StrOutputParser
    recursion_limit: int

    # ── setup ────────────────────────────────────────────────────────

    @staticmethod
    def _env_flag_enabled(name: str) -> bool:
        return os.environ.get(name, "false").lower() in ("true", "1", "yes")

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        return str(value).strip().lower() in ("true", "1", "yes")

    def setup(self):
        self.db = self.get_tool("db")
        self.metrics = SummaryMetrics()

        self.time_overlap_threshold = self.get_param(
            "time_overlap_threshold", default=0.1
        )
        self.time_adjacent_threshold = self.get_param(
            "time_adjacent_threshold", default=4
        )
        self.max_events_per_batch = self.get_param("max_events_per_batch", default=50)
        # Param or LVS_ENABLE_LLM_MERGING env may enable LLM description merging.
        self.enable_llm_merging = self._as_bool(
            self.get_param("enable_llm_merging", default=False)
        ) or self._env_flag_enabled("LVS_ENABLE_LLM_MERGING")
        self.kafka_enabled = self.get_param("kafka_enabled", default=False)
        _raw_start = self.get_param("start_time", default=None)
        _raw_end = self.get_param("end_time", default=None)
        self.filter_start_time = (
            _parse_timestamp(_raw_start) if _raw_start is not None else None
        )
        self.filter_end_time = (
            _parse_timestamp(_raw_end) if _raw_end is not None else None
        )

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
        aggregation_system_prompt = (
            self.get_param("aggregation_prompt", default=None) or ""
        ).strip() or DEFAULT_AGGREGATION_PROMPT
        aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", aggregation_system_prompt),
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

        # Description-merge pipeline is only built when LLM merging is enabled
        # (enable_llm_merging param or LVS_ENABLE_LLM_MERGING env).
        self.description_merge_pipeline = None
        if self.enable_llm_merging:
            description_merge_system_prompt = (
                self.get_param("description_merge_prompt", default=None) or ""
            ).strip() or DEFAULT_DESCRIPTION_MERGE_PROMPT
            description_merge_prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", description_merge_system_prompt),
                    (
                        "user",
                        "Event type: {event_type}\n\n"
                        "Descriptions to merge:\n{descriptions}\n\n"
                        "Merged description:",
                    ),
                ]
            )
            self.description_merge_pipeline = (
                description_merge_prompt
                | self.llm
                | self.output_parser
                | remove_think_tags
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
    def _parse_json_document(
        cls,
        doc: str,
        doc_meta: Optional[dict] = None,
    ) -> Tuple[List[Event], List[Event]]:
        """Parse a JSON document and return validated events plus untyped events.

        Returns
        -------
        (events, needs_type_inference)
            *events* are ready to use.  *needs_type_inference* contains
            events that have a valid description but an empty/missing type
            (only populated when ``LVS_DROP_EMPTY_EVENT_FIELDS`` is
            disabled/false).  Callers should pass the second list through
            ``_infer_event_types`` to attempt LLM-based classification.

            When ``LVS_DROP_EMPTY_EVENT_FIELDS`` is enabled (the default),
            events with an empty type are dropped outright.  Events with
            an empty description are always dropped regardless of the flag.

        Parameters
        ----------
        doc:
            Raw JSON (or VLM-wrapped JSON) text.
        doc_meta:
            Optional chunk metadata dict (from ``ChunkInfo``).  When
            provided, ``_chunk_boundaries`` extracts the chunk's time
            window.  Events with ``start_time >= end_time`` are rescued
            using those boundaries when they form a valid (positive-
            duration) window; otherwise the event is dropped.
        """
        chunk_start_time, chunk_end_time = (
            cls._chunk_boundaries(doc_meta) if doc_meta else (None, None)
        )
        try:
            json_content = cls._extract_json_from_vlm_response(doc)
            with Metrics("VlmStructured/ParseJSONDocument", "green"):
                # The VLM sometimes emits multiple concatenated top-level
                # arrays/objects; parse each so no events are dropped.
                events_data: List = []
                for segment in split_top_level_json_values(json_content):
                    data = json_repair.loads(segment)
                    segment_events = (
                        data.get("events", []) if isinstance(data, dict) else data
                    )
                    if isinstance(segment_events, list):
                        events_data.extend(segment_events)
                    else:
                        logger.warning(
                            f"Expected events list, got {type(segment_events)}"
                        )

            logger.debug(f"Parsed and repaired JSON data: {events_data}")

            drop_empty = os.environ.get(
                "LVS_DROP_EMPTY_EVENT_FIELDS", "true"
            ).lower() in ("true", "1", "yes", "on")

            valid_events: List[Event] = []
            deferred_events: List[Event] = []
            needs_type_inference: List[Event] = []

            for event_data in events_data:
                candidate_events = (
                    event_data if isinstance(event_data, list) else [event_data]
                )
                for candidate_event in candidate_events:
                    if not isinstance(candidate_event, dict):
                        logger.warning(
                            f"Skipping invalid event {candidate_event}: expected object"
                        )
                        continue

                    try:
                        ev_type = (candidate_event.get("type") or "").strip()
                        ev_desc = (candidate_event.get("description") or "").strip()

                        if not ev_desc:
                            logger.warning(
                                "Dropping event with empty description "
                                "(type=%r, description=%r): %s",
                                candidate_event.get("type"),
                                candidate_event.get("description"),
                                candidate_event,
                            )
                            continue

                        if not ev_type:
                            if drop_empty:
                                logger.warning(
                                    "Dropping event with empty type "
                                    "(type=%r, description=%r): %s",
                                    candidate_event.get("type"),
                                    candidate_event.get("description"),
                                    candidate_event,
                                )
                                continue

                            candidate_event = {**candidate_event, "type": ev_type}

                        event = Event(**candidate_event)

                        # Check duration first so zero/negative-duration events
                        # take the deferred (chunk-boundary rescue/drop) path
                        # regardless of type — this keeps invalid-timestamp
                        # events out of _infer_event_types.
                        if event.start_time >= event.end_time:
                            deferred_events.append(event)
                        elif not ev_type:
                            needs_type_inference.append(event)
                        else:
                            valid_events.append(event)
                    except Exception as e:
                        logger.warning(f"Skipping invalid event {candidate_event}: {e}")

            if deferred_events:
                for event in deferred_events:
                    # Replace VLM-emitted zero-duration timestamps with the
                    # authoritative chunk boundaries from content_metadata:
                    #   - live stream → start_ntp_float / end_ntp_float
                    #   - file        → start_pts / end_pts
                    # (selected by ``_chunk_boundaries``).  Events are dropped
                    # only when no chunk boundaries are available at all (e.g.
                    # legacy in-memory path with empty ``doc_meta``).
                    if (
                        chunk_start_time is not None
                        and chunk_end_time is not None
                        and chunk_end_time > chunk_start_time
                    ):
                        logger.warning(
                            "Adjusting zero/negative-duration event "
                            "(start_time=%.3f, end_time=%.3f) to chunk "
                            "boundaries (%.3f, %.3f): %s",
                            event.start_time,
                            event.end_time,
                            chunk_start_time,
                            chunk_end_time,
                            event.description,
                        )
                        event.start_time = chunk_start_time
                        event.end_time = chunk_end_time
                        valid_events.append(event)
                    else:
                        logger.warning(
                            "Dropping zero/negative-duration event — no chunk "
                            "boundaries in content_metadata "
                            "(start_time=%.3f, end_time=%.3f): %s",
                            event.start_time,
                            event.end_time,
                            event.description,
                        )

            logger.info(
                f"Parsed {len(valid_events)} events from JSON document "
                f"({len(needs_type_inference)} need type inference)"
            )
            return valid_events, needs_type_inference

        except Exception as e:
            logger.warning(f"Failed to parse JSON document: {e}")
            return [], []

    # ── Event-list persistence ─────────────────────────────────────────

    def _store_event_list(self, doc: str, doc_i: int, doc_meta: dict) -> None:
        """Persist an ``event_list`` document to the DB.

        This must be called during ``aprocess_doc`` so that
        ``_get_known_event_types`` can retrieve the list later when
        performing LLM-based type inference.
        """
        meta = {
            "chunkIdx": doc_meta.get("chunkIdx", doc_i),
            "batch_i": doc_meta.get("batch_i", doc_i),
            "doc_type": "event_list",
            "uuid": doc_meta.get("uuid", self.uuids[0]),
            "camera_id": doc_meta.get("camera_id", "default"),
        }
        with Metrics("VlmStructured/StoreEventList", "green"):
            self.db.add_summary(summary=doc, metadata=meta)
        logger.info("Stored event_list document for uuid=%s", meta["uuid"])

    # ── LLM type inference ──────────────────────────────────────────────

    def _get_known_event_types(self, uuid: str) -> List[str]:
        """Retrieve known event types from the event_list document in the DB."""
        try:
            docs = self.db.retrieve_docs(uuid=uuid, doc_type="event_list")
            if not docs:
                logger.debug("No event_list document found for uuid=%s", uuid)
                return []
            text = docs[0].get("text", "")
            if not text:
                return []
            data = json.loads(text)
            events_list = data.get("events", [])
            known_types = sorted(
                {
                    (e["type"] if isinstance(e, dict) else e)
                    for e in events_list
                    if (isinstance(e, dict) and e.get("type"))
                    or (isinstance(e, str) and e.strip())
                }
            )
            if known_types:
                logger.info(
                    "Retrieved %d known event types for uuid=%s: %s",
                    len(known_types),
                    uuid,
                    known_types,
                )
            return known_types
        except Exception as e:
            logger.warning(
                "Failed to retrieve known event types for uuid=%s: %s",
                uuid,
                e,
            )
            return []

    async def _infer_event_types(
        self,
        events: List[Event],
        uuid: str = "",
    ) -> List[Event]:
        """Use LLM to assign a type to events that have a description but no type.
        If *uuid* is provided, the ``event_list`` document is fetched from
        the DB to supply the LLM with known event types for that stream.
        All events are sent to the LLM concurrently via ``asyncio.gather``.
        Returns only the events for which a non-empty type was successfully
        inferred. Events that cannot be typed are logged and discarded.
        """
        if not events:
            return []

        known_types = self._get_known_event_types(uuid) if uuid else []

        if known_types:
            types_str = ", ".join(f"'{t}'" for t in known_types)
            type_guidance = (
                f"The following event types are known for this stream: "
                f"{types_str}.\n"
                "Pick the most appropriate type from this list and only from this list\n\n"
            )
        else:
            type_guidance = ""

        async def _infer_single(event: Event) -> Optional[Event]:
            prompt = (
                "You are classifying a video event. Given the description below, "
                "reply with ONLY a short event type label (1-3 words). "
                "Do not include any other text.\n\n"
                f"{type_guidance}"
                f"Description: {event.description}"
            )
            try:
                with Metrics("VlmStructured/InferEventType", "yellow"):
                    response = await self.llm.ainvoke(prompt)
                t = remove_think_tags(response.content).strip().strip("\"'")
                if t:
                    event.type = t
                    logger.info("Inferred type %r for event: %s", t, event.description)
                    return event
                logger.warning(
                    "LLM returned empty type for event, dropping: %s",
                    event.description,
                )
            except Exception as e:
                logger.warning(
                    "LLM type inference failed for event, dropping: %s — %s",
                    event.description,
                    e,
                )
            return None

        _infer_start = time.time()
        with Metrics("VlmStructured/InferEventTypes", "yellow"):
            with get_openai_callback() as cb:
                results = await asyncio.gather(*(_infer_single(e) for e in events))
                logger.info(
                    f"InferEventTypes - Total Tokens: {cb.total_tokens}, "
                    f"Prompt Tokens: {cb.prompt_tokens}, "
                    f"Completion Tokens: {cb.completion_tokens}, "
                    f"Total Cost (USD): ${cb.total_cost}"
                )
                # summary_tokens keeps the grand total; event_type_infer_tokens is the
                # per-call breakdown. calls/requests count actual LLM requests (one per
                # event, fanned out via asyncio.gather).
                self.metrics.summary_tokens += cb.total_tokens
                self.metrics.event_type_infer_tokens += cb.total_tokens
                self.metrics.event_type_infer_calls += cb.successful_requests
                self.metrics.summary_requests += cb.successful_requests
        self.metrics.event_type_infer_latency += time.time() - _infer_start
        return [e for e in results if e is not None]

    # ── Merging ─────────────────────────────────────────────────────────

    async def _merge_descriptions_with_llm(
        self, event_type: str, descriptions: List[str]
    ) -> str:
        """Use LLM to merge multiple event descriptions into one coherent description."""
        if len(descriptions) == 1:
            return descriptions[0]
        if self.description_merge_pipeline is None:
            return " | ".join(descriptions)

        formatted_descriptions = "\n".join(
            f"{i + 1}. {desc}" for i, desc in enumerate(descriptions)
        )

        try:
            with Metrics("VlmStructured/MergeDescriptions", "yellow"):
                with get_openai_callback() as cb:
                    _merge_start = time.time()
                    merged_description = await call_token_safe(
                        {
                            "event_type": event_type,
                            "descriptions": formatted_descriptions,
                        },
                        self.description_merge_pipeline,
                        self.recursion_limit,
                    )
                    _merge_elapsed = time.time() - _merge_start
                    logger.info(
                        f"LLM merged {len(descriptions)} descriptions for '{event_type}' event"
                    )
                    logger.info(
                        f"MergeDescriptions - Total Tokens: {cb.total_tokens}, "
                        f"Prompt Tokens: {cb.prompt_tokens}, "
                        f"Completion Tokens: {cb.completion_tokens}, "
                        f"Total Cost (USD): ${cb.total_cost}"
                    )
                    # Accumulate merge metrics (None -> 0 on first call). Only reached
                    # when enable_llm_merging=True, so a disabled run leaves these None.
                    # summary_tokens keeps the grand total; llm_merge_tokens is the breakdown.
                    self.metrics.summary_tokens += cb.total_tokens
                    self.metrics.llm_merge_latency = (
                        self.metrics.llm_merge_latency or 0
                    ) + _merge_elapsed
                    self.metrics.llm_merge_tokens = (
                        self.metrics.llm_merge_tokens or 0
                    ) + cb.total_tokens
                    self.metrics.llm_merge_calls = (
                        self.metrics.llm_merge_calls or 0
                    ) + cb.successful_requests
                    self.metrics.summary_requests += cb.successful_requests
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
        start_time: Optional[Union[float, str]] = None,
        end_time: Optional[Union[float, str]] = None,
    ) -> List[Event]:
        """Return only events that overlap the ``[start_time, end_time]`` window.

        An event is kept when its time span intersects the filter window, i.e.
        ``event.end_time >= start_time`` and ``event.start_time <= end_time``.
        If both bounds are *None* the original list is returned unchanged.

        *start_time* and *end_time* accept numeric values, ``MM:SS`` /
        ``HH:MM:SS`` elapsed-time strings, or ISO 8601 strings; they are
        coerced to ``float`` seconds before comparison.
        """
        if start_time is None and end_time is None:
            return events

        start_time = _parse_timestamp(start_time) if start_time is not None else None
        end_time = _parse_timestamp(end_time) if end_time is not None else None

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

    # ── DB retrieval ────────────────────────────────────────────────────

    async def _fetch_events_from_db(
        self,
        uuids: List[str],
        start_time: Optional[Union[float, str]] = None,
        end_time: Optional[Union[float, str]] = None,
    ) -> List[Event]:
        """Retrieve raw event documents from the DB, parse them into Events, and
        narrow to the ``[start_time, end_time]`` window.

        Caption-source-agnostic: shared by the online (live) path and the
        file-path ``LVS_CAPTION_SOURCE=db`` branch. The
        ``dense_captions_retrieval_latency`` metric accumulates only the ES/DB
        pull (``retrieve_docs``) plus the time-narrowing filter; JSON parsing
        and the ``_infer_event_types`` LLM call in between are excluded (the
        latter has its own ``event_type_infer_latency`` metric).

        When *uuids* contains more than one entry, each parsed ``Event`` is
        tagged with the UUID it originated from so downstream storage and
        result building can preserve provenance.
        """
        multi = len(uuids) > 1
        all_events: List[Event] = []
        retrieval_latency = 0.0

        for uuid in uuids:
            _rd_start = time.time()
            raw_docs = self.db.retrieve_docs(uuid=uuid, doc_type="raw_events")
            retrieval_latency += time.time() - _rd_start
            logger.info(
                f"Fetched {len(raw_docs)} raw_event documents from DB for uuid '{uuid}'"
            )
            for doc in raw_docs:
                text = doc.get("text", "")
                if not text:
                    continue
                doc_meta = {k: v for k, v in doc.items() if k != "text"}
                events, needs_type = self._parse_json_document(text, doc_meta)
                if needs_type:
                    inferred = await self._infer_event_types(needs_type, uuid=uuid)
                    events.extend(inferred)
                if multi:
                    for event in events:
                        event.uuid = uuid
                all_events.extend(events)

        _filter_start = time.time()
        all_events = self._filter_events_by_time(all_events, start_time, end_time)
        retrieval_latency += time.time() - _filter_start

        # Dense-caption retrieval latency = ES pull + time-narrowing only.
        self.metrics.dense_captions_retrieval_latency = retrieval_latency
        logger.info(
            f"Parsed {len(all_events)} total events from DB documents "
            f"across {len(uuids)} UUID(s)"
        )
        return all_events

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

        agg_start_time = time.time()
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
        self.metrics.aggregation_latency = time.time() - agg_start_time

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

        # Enabled-vs-disabled reporting: when LLM event-merging is ENABLED but no
        # events needed merging, report 0 (not None) so downstream can distinguish
        # "enabled, nothing to merge" (0) from "disabled" (None -> N/A / '-'). When
        # merging actually ran, llm_merge_latency is already set and is left as-is.
        if self.enable_llm_merging and self.metrics.llm_merge_latency is None:
            self.metrics.llm_merge_latency = 0.0
            self.metrics.llm_merge_tokens = 0
            self.metrics.llm_merge_calls = 0

        state["metadata"] = self.metrics.dump_dict()

        if self.log_dir:
            log_path = Path(self.log_dir).joinpath(log_filename)
            self.metrics.dump_json(log_path.absolute())

        return state

    # ── Doc ingestion helpers ───────────────────────────────────────────

    @staticmethod
    def _chunk_boundaries(doc_meta: dict) -> tuple[float, float]:
        """
        Extract ``(chunk_start_time, chunk_end_time)`` in seconds from *doc_meta*.

        Two metadata shapes are supported, depending on stream type:

        - **Live stream summarization** — RTVI populates absolute NTP wall
          clock fields on the proto message; Logstash mirrors them as
          ``start_ntp_float`` / ``end_ntp_float`` (float epoch-seconds).
          Used for live because ``start_pts`` is unreliable for RTSP streams
          (PTS rolls over per GStreamer pipeline restart).

        - **File summarization** — chunk boundaries come from the video
          container's PTS (``start_pts`` / ``end_pts`` in nanoseconds, relative
          to the asset's ``creation_time``). Returned as seconds-from-creation.
        """
        if "start_ntp_float" in doc_meta and "end_ntp_float" in doc_meta:
            return float(doc_meta["start_ntp_float"]), float(doc_meta["end_ntp_float"])
        if "start_pts" in doc_meta and "end_pts" in doc_meta:
            return doc_meta["start_pts"] / 1e9, doc_meta["end_pts"] / 1e9
        return None, None

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
