# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError

from vss_ctx_rag.functions.summarization.vlm_structured import (
    VlmStructuredSummarization,
)
from vss_ctx_rag.functions.summarization.vlm_structured_base import (
    Event,
    VlmStructuredParamsBase,
)


class ParallelMergeTests(unittest.IsolatedAsyncioTestCase):
    def make_function(self, concurrency=4, enabled=True):
        obj = VlmStructuredSummarization.__new__(VlmStructuredSummarization)
        obj.time_overlap_threshold = 0.1
        obj.time_adjacent_threshold = 4
        obj.enable_llm_merging = enabled
        obj.llm_merge_concurrency = concurrency
        return obj

    def events(self):
        return [
            Event(
                start_time=30,
                end_time=35,
                type="action",
                description="late A",
                uuid="v",
            ),
            Event(
                start_time=0, end_time=5, type="action", description="early A", uuid="v"
            ),
            Event(
                start_time=14,
                end_time=19,
                type="action",
                description="early C",
                uuid="v",
            ),
            Event(
                start_time=7,
                end_time=12,
                type="action",
                description="early B",
                uuid="v",
            ),
            Event(
                start_time=30,
                end_time=35,
                type="object",
                description="object A",
                uuid="v",
            ),
            Event(
                start_time=37,
                end_time=42,
                type="action",
                description="late B",
                uuid="v",
            ),
            Event(
                start_time=37,
                end_time=42,
                type="object",
                description="object B",
                uuid="v",
            ),
            Event(
                start_time=60,
                end_time=65,
                type="action",
                description="singleton",
                uuid="v",
            ),
        ]

    async def test_parallel_matches_sequential_groups_and_stable_order(self):
        results = []
        calls = []
        for concurrency in (1, 4):
            obj = self.make_function(concurrency)

            async def merge(event_type, descriptions):
                await asyncio.sleep(0.01 if descriptions[0] == "late A" else 0)
                return " | ".join(descriptions)

            obj._merge_descriptions_with_llm = AsyncMock(side_effect=merge)
            results.append(await obj._merge_similar_events(self.events()))
            calls.append(obj._merge_descriptions_with_llm.call_args_list)
        self.assertEqual(results[0], results[1])
        self.assertEqual(calls[0], calls[1])
        self.assertEqual(
            [(e.start_time, e.end_time, e.type, e.description) for e in results[1]],
            [
                (0, 19, "action", "early A | early B | early C"),
                (30, 42, "action", "late A | late B"),
                (30, 42, "object", "object A | object B"),
                (60, 65, "action", "singleton"),
            ],
        )
        self.assertTrue(all(e.uuid == "v" for e in results[1]))

    async def test_concurrency_limit_and_actual_overlap(self):
        obj = self.make_function(2)
        active = peak = 0
        both_started = asyncio.Event()

        async def merge(event_type, descriptions):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)
            await asyncio.sleep(0)
            active -= 1
            return " | ".join(descriptions)

        obj._merge_descriptions_with_llm = AsyncMock(side_effect=merge)
        result = await obj._merge_similar_events(self.events())
        self.assertEqual(peak, 2)
        self.assertEqual(len(result), 4)
        self.assertEqual(obj._merge_descriptions_with_llm.await_count, 3)

    async def test_disabled_merging_and_empty_input_skip_llm(self):
        obj = self.make_function(enabled=False)
        obj._merge_descriptions_with_llm = AsyncMock()
        self.assertEqual(await obj._merge_similar_events([]), [])
        with patch.object(
            asyncio, "create_task", wraps=asyncio.create_task
        ) as create_task:
            result = await obj._merge_similar_events(self.events())
        create_task.assert_not_called()
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0].description, "early A | early B | early C")
        obj._merge_descriptions_with_llm.assert_not_awaited()

    async def test_failure_cancels_other_merges(self):
        obj = self.make_function(2)
        started = asyncio.Event()
        cancelled = asyncio.Event()

        # The real _merge_descriptions_with_llm falls back to concatenation on
        # Exception, so this stands in for errors that escape it (for example,
        # cancellation of the enclosing request).
        async def merge(event_type, descriptions):
            if descriptions[0] == "late A":
                await started.wait()
                raise RuntimeError("unexpected failure")
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise

        obj._merge_descriptions_with_llm = AsyncMock(side_effect=merge)
        with self.assertRaisesRegex(RuntimeError, "unexpected failure"):
            await asyncio.wait_for(obj._merge_similar_events(self.events()), timeout=1)
        self.assertTrue(cancelled.is_set())

    def test_concurrency_configuration(self):
        self.assertEqual(VlmStructuredParamsBase().llm_merge_concurrency, 4)
        self.assertEqual(
            VlmStructuredParamsBase(llm_merge_concurrency=1).llm_merge_concurrency, 1
        )
        for invalid in (0, -1, 33):
            with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                VlmStructuredParamsBase(llm_merge_concurrency=invalid)


if __name__ == "__main__":
    unittest.main()
