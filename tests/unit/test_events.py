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

"""Unit tests for Event validation (duration, empty fields, type inference)."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from vss_ctx_rag.functions.summarization.vlm_structured_base import (
    Event,
    VlmStructuredBase,
)
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics


class TestEventDurationValidation:
    """Verify that _parse_json_document drops events with start_time >= end_time."""

    def _make_doc(self, events: list[dict]) -> str:
        return json.dumps({"events": events})

    def test_valid_event_is_kept(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 1.0,
                    "end_time": 5.0,
                    "type": "motion",
                    "description": "ok",
                }
            ]
        )
        events, _ = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 1
        assert events[0].start_time == 1.0
        assert events[0].end_time == 5.0

    def test_zero_duration_event_is_dropped(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 3.0,
                    "end_time": 3.0,
                    "type": "motion",
                    "description": "zero",
                }
            ]
        )
        events, _ = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0

    def test_negative_duration_event_is_dropped(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 5.0,
                    "end_time": 4.0,
                    "type": "motion",
                    "description": "neg",
                }
            ]
        )
        events, _ = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0

    def test_mixed_valid_and_invalid_events(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 2.0,
                    "type": "a",
                    "description": "good",
                },
                {"start_time": 3.0, "end_time": 3.0, "type": "b", "description": "bad"},
                {"start_time": 5.0, "end_time": 4.0, "type": "c", "description": "bad"},
                {
                    "start_time": 6.0,
                    "end_time": 7.0,
                    "type": "d",
                    "description": "good",
                },
            ]
        )
        events, _ = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 2
        assert events[0].description == "good"
        assert events[1].description == "good"

    def test_barely_positive_duration_is_kept(self):
        doc = self._make_doc(
            [{"start_time": 1.0, "end_time": 1.001, "type": "x", "description": "tiny"}]
        )
        events, _ = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 1

    def test_event_model_still_allows_construction(self):
        ev = Event(start_time=0.0, end_time=0.0, type="x", description="allowed")
        assert ev.start_time == ev.end_time


class TestEventEmptyFieldValidation:
    """Verify that LVS_DROP_EMPTY_EVENT_FIELDS controls whether events with
    empty type/description are handled by _parse_json_document."""

    def _make_doc(self, events: list[dict]) -> str:
        return json.dumps({"events": events})

    # ── Model allows empty strings (no validator) ───────────────────

    def test_event_model_accepts_empty_type(self):
        ev = Event(start_time=0.0, end_time=1.0, type="", description="ok")
        assert ev.type == ""

    def test_event_model_accepts_empty_description(self):
        ev = Event(start_time=0.0, end_time=1.0, type="motion", description="")
        assert ev.description == ""

    # ── Parser drops empty-type events when LVS_DROP_EMPTY_EVENT_FIELDS=true (default) ─

    def test_parse_drops_empty_type_when_enabled(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "",
                    "description": "ok",
                },
                {
                    "start_time": 2.0,
                    "end_time": 3.0,
                    "type": "motion",
                    "description": "good",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 1
        assert events[0].type == "motion"
        assert len(needs_type) == 0

    def test_parse_drops_whitespace_only_type_when_enabled(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "   ",
                    "description": "ok",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    def test_parse_skips_empty_description(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "motion",
                    "description": "",
                },
                {
                    "start_time": 2.0,
                    "end_time": 3.0,
                    "type": "alert",
                    "description": "fire",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 1
        assert events[0].type == "alert"
        assert len(needs_type) == 0

    def test_parse_skips_whitespace_only_description(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "motion",
                    "description": "  \n ",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    def test_parse_skips_missing_type_key(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "description": "no type key",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    def test_parse_skips_missing_description_key(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "motion",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    # ── Parser routes empty-type to inference when LVS_DROP_EMPTY_EVENT_FIELDS=false ─

    def test_parse_routes_empty_type_to_inference_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LVS_DROP_EMPTY_EVENT_FIELDS", "false")
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "",
                    "description": "ok",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 1
        assert needs_type[0].description == "ok"

    def test_parse_drops_empty_description_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LVS_DROP_EMPTY_EVENT_FIELDS", "false")
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "motion",
                    "description": "",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0


# ── Helpers for _infer_event_types tests ─────────────────────────────

_EVENT_LIST_TEXT = json.dumps(
    {
        "events": [
            {"id": 1, "type": "vehicle movement", "description": "car drives"},
            {"id": 2, "type": "person detected", "description": "person walks"},
            {"id": 3, "type": "safety violation", "description": "no helmet"},
        ]
    }
)

_EVENT_LIST_TEXT_STRINGS = json.dumps(
    {
        "events": [
            "box dropping",
            "not wearing PPE",
            "unsafe forklift operations",
            "normal activity",
        ]
    }
)


def _make_mock_instance(
    return_text: str = "vehicle movement",
    event_list_docs: list | None = None,
):
    """Create a mock VlmStructuredBase-like object with an async LLM and DB."""
    instance = MagicMock(spec=VlmStructuredBase)
    instance.llm = AsyncMock()
    instance.llm.ainvoke.return_value = SimpleNamespace(content=return_text)

    # Real metrics object so new counters written by production code do not
    # require updating a hand-rolled stub.
    instance.metrics = SummaryMetrics()

    instance.db = MagicMock()
    if event_list_docs is None:
        instance.db.retrieve_docs.return_value = [{"text": _EVENT_LIST_TEXT}]
    else:
        instance.db.retrieve_docs.return_value = event_list_docs

    instance._infer_event_types = VlmStructuredBase._infer_event_types.__get__(
        instance, VlmStructuredBase
    )
    instance._get_known_event_types = VlmStructuredBase._get_known_event_types.__get__(
        instance, VlmStructuredBase
    )
    return instance


@pytest.fixture()
def _es_backend(monkeypatch):
    """Set LVS_DATABASE_BACKEND=elasticsearch_db for inference tests."""
    monkeypatch.setenv("LVS_DATABASE_BACKEND", "elasticsearch_db")


@pytest.mark.usefixtures("_es_backend")
class TestInferEventTypes:
    """Tests for VlmStructuredBase._infer_event_types (async, instance method)."""

    def _make_event(self, **overrides):
        defaults = dict(
            start_time=0.0, end_time=1.0, type="", description="a car drives by"
        )
        defaults.update(overrides)
        return Event(**defaults)

    # ── edge cases ──────────────────────────────────────────────────

    def test_empty_list_returns_empty(self):
        inst = _make_mock_instance()
        result = asyncio.run(inst._infer_event_types([]))
        assert result == []
        inst.llm.ainvoke.assert_not_called()

    # ── successful inference ────────────────────────────────────────

    def test_single_event_type_inferred(self):
        inst = _make_mock_instance("vehicle movement")
        events = [self._make_event(description="a car drives by")]
        result = asyncio.run(inst._infer_event_types(events, uuid="test-uuid"))
        assert len(result) == 1
        assert result[0].type == "vehicle movement"
        assert result[0].description == "a car drives by"
        inst.llm.ainvoke.assert_called_once()

    def test_multiple_events_run_concurrently(self):
        inst = _make_mock_instance()
        inst.llm.ainvoke.side_effect = [
            SimpleNamespace(content="person detected"),
            SimpleNamespace(content="vehicle movement"),
        ]
        events = [
            self._make_event(description="someone walks"),
            self._make_event(description="a truck passes"),
        ]
        result = asyncio.run(inst._infer_event_types(events, uuid="test-uuid"))
        assert len(result) == 2
        assert result[0].type == "person detected"
        assert result[1].type == "vehicle movement"
        assert inst.llm.ainvoke.call_count == 2

    def test_known_types_included_in_prompt(self):
        inst = _make_mock_instance("vehicle movement")
        events = [self._make_event(description="a car drives by")]
        asyncio.run(inst._infer_event_types(events, uuid="test-uuid"))
        prompt = inst.llm.ainvoke.call_args[0][0]
        assert "vehicle movement" in prompt
        assert "person detected" in prompt
        assert "safety violation" in prompt

    def test_no_known_types_still_infers(self):
        inst = _make_mock_instance("vehicle movement", event_list_docs=[])
        events = [self._make_event(description="a car drives by")]
        result = asyncio.run(inst._infer_event_types(events, uuid="test-uuid"))
        assert len(result) == 1
        assert result[0].type == "vehicle movement"

    def test_strips_quotes_from_response(self):
        inst = _make_mock_instance("'safety violation'")
        result = asyncio.run(inst._infer_event_types([self._make_event()]))
        assert result[0].type == "safety violation"

    def test_strips_think_tags_from_response(self):
        inst = _make_mock_instance("<think>hmm</think>person detected")
        result = asyncio.run(inst._infer_event_types([self._make_event()]))
        assert result[0].type == "person detected"

    # ── failure modes ───────────────────────────────────────────────

    def test_empty_response_drops_event(self):
        inst = _make_mock_instance("   ")
        result = asyncio.run(inst._infer_event_types([self._make_event()]))
        assert result == []

    def test_llm_exception_drops_event(self):
        inst = _make_mock_instance()
        inst.llm.ainvoke.side_effect = RuntimeError("LLM unavailable")
        result = asyncio.run(inst._infer_event_types([self._make_event()]))
        assert result == []

    def test_partial_failure_keeps_successful(self):
        inst = _make_mock_instance()
        inst.llm.ainvoke.side_effect = [
            SimpleNamespace(content="person detected"),
            RuntimeError("timeout"),
            SimpleNamespace(content="safety violation"),
        ]
        events = [
            self._make_event(description="person walks"),
            self._make_event(description="unknown"),
            self._make_event(description="fire spotted"),
        ]
        result = asyncio.run(inst._infer_event_types(events))
        assert len(result) == 2
        assert result[0].type == "person detected"
        assert result[1].type == "safety violation"


@pytest.mark.usefixtures("_es_backend")
class TestParseAndInferIntegration:
    """Integration: _parse_json_document + _infer_event_types two-step flow."""

    def _make_doc(self, events: list[dict]) -> str:
        return json.dumps({"events": events})

    def test_empty_type_inferred_via_llm(self, monkeypatch):
        monkeypatch.setenv("LVS_DROP_EMPTY_EVENT_FIELDS", "false")
        inst = _make_mock_instance("vehicle movement")
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "",
                    "description": "a car",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 1
        inferred = asyncio.run(inst._infer_event_types(needs_type, uuid="test-uuid"))
        events.extend(inferred)
        assert len(events) == 1
        assert events[0].type == "vehicle movement"

    def test_empty_type_dropped_when_enabled(self):
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "",
                    "description": "a car",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    def test_mixed_typed_and_untyped_events(self, monkeypatch):
        monkeypatch.setenv("LVS_DROP_EMPTY_EVENT_FIELDS", "false")
        inst = _make_mock_instance("person detected")
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "motion",
                    "description": "good",
                },
                {
                    "start_time": 2.0,
                    "end_time": 3.0,
                    "type": "",
                    "description": "someone walks",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 1
        assert len(needs_type) == 1
        inferred = asyncio.run(inst._infer_event_types(needs_type, uuid="test-uuid"))
        events.extend(inferred)
        assert len(events) == 2
        assert events[0].type == "motion"
        assert events[1].type == "person detected"

    def test_empty_description_not_routed_to_inference(self):
        doc = self._make_doc(
            [
                {"start_time": 0.0, "end_time": 1.0, "type": "", "description": ""},
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 0

    def test_inference_used_when_drop_disabled(self, monkeypatch):
        monkeypatch.setenv("LVS_DROP_EMPTY_EVENT_FIELDS", "false")
        inst = _make_mock_instance("vehicle movement")
        doc = self._make_doc(
            [
                {
                    "start_time": 0.0,
                    "end_time": 1.0,
                    "type": "",
                    "description": "a car",
                },
            ]
        )
        events, needs_type = VlmStructuredBase._parse_json_document(doc)
        assert len(events) == 0
        assert len(needs_type) == 1
        inferred = asyncio.run(inst._infer_event_types(needs_type, uuid="test-uuid"))
        events.extend(inferred)
        assert len(events) == 1
        assert events[0].type == "vehicle movement"


@pytest.mark.usefixtures("_es_backend")
class TestGetKnownEventTypes:
    """Tests for _get_known_event_types handling both dict and string event formats."""

    def test_dict_format_events(self):
        inst = _make_mock_instance()
        types = inst._get_known_event_types("test-uuid")
        assert types == ["person detected", "safety violation", "vehicle movement"]

    def test_string_format_events(self):
        inst = _make_mock_instance(event_list_docs=[{"text": _EVENT_LIST_TEXT_STRINGS}])
        types = inst._get_known_event_types("test-uuid")
        assert types == [
            "box dropping",
            "normal activity",
            "not wearing PPE",
            "unsafe forklift operations",
        ]

    def test_empty_db_returns_empty(self):
        inst = _make_mock_instance(event_list_docs=[])
        types = inst._get_known_event_types("test-uuid")
        assert types == []

    def test_empty_text_returns_empty(self):
        inst = _make_mock_instance(event_list_docs=[{"text": ""}])
        types = inst._get_known_event_types("test-uuid")
        assert types == []

    def test_whitespace_strings_excluded(self):
        text = json.dumps({"events": ["valid type", "  ", ""]})
        inst = _make_mock_instance(event_list_docs=[{"text": text}])
        types = inst._get_known_event_types("test-uuid")
        assert types == ["valid type"]


class TestStoreEventList:
    """Tests for _store_event_list persisting event_list documents to DB."""

    def test_stores_event_list_with_correct_metadata(self):
        inst = _make_mock_instance()
        inst.uuids = ["default"]
        inst._store_event_list = VlmStructuredBase._store_event_list.__get__(
            inst, VlmStructuredBase
        )

        doc = '{"events": ["box dropping", "fire"]}'
        doc_meta = {"uuid": "abc-123", "doc_type": "event_list", "camera_id": "cam1"}

        inst._store_event_list(doc, -1, doc_meta)

        inst.db.add_summary.assert_called_once()
        call_kwargs = inst.db.add_summary.call_args
        assert call_kwargs[1]["summary"] == doc
        meta = call_kwargs[1]["metadata"]
        assert meta["doc_type"] == "event_list"
        assert meta["uuid"] == "abc-123"
        assert meta["camera_id"] == "cam1"

    def test_falls_back_to_first_uuid(self):
        inst = _make_mock_instance()
        inst.uuids = ["fallback-uuid"]
        inst._store_event_list = VlmStructuredBase._store_event_list.__get__(
            inst, VlmStructuredBase
        )

        doc = '{"events": ["fire"]}'
        doc_meta = {"doc_type": "event_list"}

        inst._store_event_list(doc, -1, doc_meta)

        meta = inst.db.add_summary.call_args[1]["metadata"]
        assert meta["uuid"] == "fallback-uuid"
        assert meta["camera_id"] == "default"
