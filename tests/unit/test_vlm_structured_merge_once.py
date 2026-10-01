# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from pyaml_env import parse_config

from vss_ctx_rag.functions.summarization.vlm_structured import (
    VlmStructuredSummarization,
)
from vss_ctx_rag.functions.summarization.vlm_structured_base import (
    Event,
    VlmStructuredParamsBase,
)
from vss_ctx_rag.functions.summarization.vlm_structured_online import (
    VlmStructuredOnlineSummarization,
)


class MergeOnceTests(unittest.IsolatedAsyncioTestCase):
    def make_function(self, cls, kafka=False, narrative=False, empty=False):
        obj = cls.__new__(cls)
        raw = (
            []
            if empty
            else [
                Event(start_time=0, end_time=5, type="action", description="walks"),
                Event(
                    start_time=5,
                    end_time=10,
                    type="action",
                    description="continues walking",
                ),
            ]
        )
        merged = (
            []
            if empty
            else [
                Event(
                    start_time=0,
                    end_time=10,
                    type="action",
                    description="walks across the room",
                )
            ]
        )
        obj.accumulated_events = raw
        obj.caption_source = "sse"
        obj.kafka_enabled = kafka
        obj.generate_video_summary = narrative
        obj.enable_llm_merging = True
        obj.uuids = ["video-1"]
        obj.max_events_per_batch = 100
        obj.filter_start_time = obj.filter_end_time = None
        obj.call_schema = Mock()
        obj.db = Mock()
        obj.metrics = Mock(llm_merge_latency=0.0)
        obj.metrics.dump_dict.return_value = {}
        obj.log_dir = None
        obj._fetch_events_from_db = AsyncMock(return_value=raw)
        obj._merge_similar_events = AsyncMock(return_value=merged)
        obj._aggregate_events_with_llm = AsyncMock(return_value="A person walks.")
        return obj, raw, merged

    async def test_file_and_online_store_and_return_same_single_merge(self):
        for cls in (VlmStructuredSummarization, VlmStructuredOnlineSummarization):
            for kafka in (False, True):
                for narrative in (False, True):
                    with self.subTest(
                        cls=cls.__name__, kafka=kafka, narrative=narrative
                    ):
                        obj, raw, merged = self.make_function(cls, kafka, narrative)
                        result = json.loads(
                            (await obj.acall({"uuid": "video-1"}))["result"]
                        )
                        obj._merge_similar_events.assert_awaited_once_with(raw)
                        self.assertEqual(result["total_events"], 1)
                        self.assertEqual(
                            result["events"][0]["description"], merged[0].description
                        )
                        self.assertEqual(
                            result["video_summary"],
                            "A person walks." if narrative else "",
                        )
                        if narrative:
                            obj._aggregate_events_with_llm.assert_awaited_once_with(
                                merged
                            )
                        else:
                            obj._aggregate_events_with_llm.assert_not_awaited()
                        if kafka:
                            obj.db.add_summary.assert_not_called()
                        else:
                            obj.db.add_summary.assert_called_once()
                            stored = json.loads(
                                obj.db.add_summary.call_args.kwargs["summary"]
                            )["events"]
                            returned = [
                                {k: v for k, v in e.items() if k != "id"}
                                for e in result["events"]
                            ]
                            self.assertEqual(stored, returned)

    async def test_empty_input_skips_llm_and_storage(self):
        for cls in (VlmStructuredSummarization, VlmStructuredOnlineSummarization):
            obj, _, _ = self.make_function(cls, empty=True)
            result = json.loads((await obj.acall({"uuid": "video-1"}))["result"])
            self.assertEqual(result["events"], [])
            obj._merge_similar_events.assert_not_awaited()
            obj._aggregate_events_with_llm.assert_not_awaited()
            obj.db.add_summary.assert_not_called()

    def test_narrative_setting_defaults_and_config(self):
        self.assertTrue(VlmStructuredParamsBase().generate_video_summary)
        self.assertFalse(
            VlmStructuredParamsBase(generate_video_summary=False).generate_video_summary
        )
        for configured in (True, False):
            with self.subTest(configured=configured):
                obj = VlmStructuredSummarization.__new__(VlmStructuredSummarization)
                obj.get_param = lambda name, default=None, configured=configured: (
                    configured if name == "generate_video_summary" else default
                )
                obj.get_tool = Mock()
                obj._setup_aggregation_pipeline = Mock()
                with patch.dict(
                    os.environ, {"LVS_GENERATE_VIDEO_SUMMARY": str(not configured)}
                ):
                    obj.setup()
                self.assertEqual(obj.generate_video_summary, configured)

    def test_lvs_config_reads_summary_setting_from_env(self):
        config_path = Path(__file__).parents[2] / "config" / "config_lvs.yaml"
        for env, expected in [(None, True), ("false", False), ("OFF", False)]:
            with self.subTest(env=env):
                environ = dict(os.environ)
                environ.pop("LVS_GENERATE_VIDEO_SUMMARY", None)
                if env is not None:
                    environ["LVS_GENERATE_VIDEO_SUMMARY"] = env
                with patch.dict(os.environ, environ, clear=True):
                    config = parse_config(str(config_path))
                params = config["functions"]["vlm_structured_summarization"]["params"]
                self.assertIs(
                    VlmStructuredParamsBase(**params).generate_video_summary, expected
                )

    async def test_merge_failure_does_not_store_raw_events(self):
        obj, _, _ = self.make_function(VlmStructuredSummarization)
        obj._merge_similar_events.side_effect = RuntimeError("merge failed")
        with self.assertRaisesRegex(RuntimeError, "merge failed"):
            await obj.acall({"uuid": "video-1"})
        obj.db.add_summary.assert_not_called()
        obj._aggregate_events_with_llm.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
