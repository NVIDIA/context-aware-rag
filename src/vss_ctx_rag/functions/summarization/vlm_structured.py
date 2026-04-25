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

"""vlm_structured.py: VLM structured summarization (in-memory event accumulation).

This version accumulates parsed events in memory during ``aprocess_doc`` and
processes them all at ``acall`` time.  For the DB-backed variant see
``vlm_structured_online.py``.
"""

import asyncio
from typing import List

from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
    FunctionModel,
)

from vss_ctx_rag.functions.summarization.vlm_structured_base import (
    Event,
    VlmStructuredBase,
    VlmStructuredParamsBase,
)


@register_function_config("vlm_structured_summarization")
class VlmStructuredSummarizationConfig(FunctionModel):
    class VlmStructuredSummarizationParams(VlmStructuredParamsBase):
        pass

    params: VlmStructuredSummarizationParams


@register_function(config=VlmStructuredSummarizationConfig)
class VlmStructuredSummarization(VlmStructuredBase):
    """VLM Structured Summarization - accumulates events in memory."""

    accumulated_events: List[Event]

    def setup(self):
        super().setup()
        self.accumulated_events = []
        self.kafka_enabled = self.get_param("kafka_enabled", default=False)

    async def acall(self, state: dict):
        """Process and merge accumulated events, then aggregate with LLM.

        Optional ``start_time`` / ``end_time`` keys in *state* (or the
        function config) restrict processing to events overlapping that
        time window.  ``uuids`` (or legacy ``uuid``) in *state* override
        the configured UUIDs.

        Returns:
            dict: state with ``result`` (JSON) and ``metadata`` keys.
        """
        with Metrics("StructuredBatchSumm/Acall", "blue"):
            self.call_schema.validate(state)

            uuids = self._resolve_uuids(state)

            start_time = state.get("start_time", self.filter_start_time)
            end_time = state.get("end_time", self.filter_end_time)
            events = self._filter_events_by_time(
                self.accumulated_events, start_time, end_time
            )

            await self._store_merged_events(events)

            state = await self._build_result(
                state,
                events,
                uuids,
                log_filename="structured_events_metrics.json",
            )

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        """Parse events from *doc*, accumulate in memory, and persist raw events to DB."""
        try:
            logger.info("Processing structured doc %d for processing", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            with Metrics("StructuredBatchSumm/aprocess_doc", "red") as bs:
                events = self._parse_json_document(doc)

                if events:
                    logger.info(f"Extracted {len(events)} events from document {doc_i}")
                    self.accumulated_events.extend(events)
                    if not self.kafka_enabled:
                        self._store_raw_events(events, doc_i, doc_meta)
                else:
                    logger.warning(f"No events found in document {doc_i}")

            if self.summary_start_time is None:
                self.summary_start_time = bs.start_time
            self.metrics.summary_latency = bs.end_time - self.summary_start_time
        except Exception as e:
            logger.error(f"Error processing document {doc_i}: {e}")

    async def areset(self, state: dict):
        """Reset function state including accumulated events."""
        self.db.reset(state)
        self.summary_start_time = None
        self.metrics.reset()
        self.accumulated_events.clear()
        await asyncio.sleep(0.001)
