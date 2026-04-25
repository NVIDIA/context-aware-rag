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

"""vlm_structured_online.py: Online variant of VLM structured summarization.

Instead of accumulating events in memory, this function fetches raw event
documents from the database by UUID at ``acall`` time.  This makes it suitable
for online / batch-replay workflows where events were already persisted
(e.g. via the VlmStructuredSummarization or an external ingest
pipeline like ``elasticpull``).
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


@register_function_config("vlm_structured_summarization_online")
class VlmStructuredOnlineSummarizationConfig(FunctionModel):
    class VlmStructuredOnlineSummarizationParams(VlmStructuredParamsBase):
        pass

    params: VlmStructuredOnlineSummarizationParams


@register_function(config=VlmStructuredOnlineSummarizationConfig)
class VlmStructuredOnlineSummarization(VlmStructuredBase):
    """Online VLM Structured Summarization - fetches events from DB by UUID."""

    # ── DB retrieval ────────────────────────────────────────────────────

    def _fetch_events_from_db(self, uuids: List[str]) -> List[Event]:
        """Retrieve raw event documents from the database and parse into Event objects.

        When *uuids* contains more than one entry, each parsed ``Event``
        is tagged with the UUID it originated from so that downstream
        storage and result building can preserve this provenance.
        """
        multi = len(uuids) > 1
        all_events: List[Event] = []

        for uuid in uuids:
            raw_docs = self.db.retrieve_docs(uuid=uuid, doc_type="raw_events")
            logger.info(
                f"Fetched {len(raw_docs)} raw_event documents from DB for uuid '{uuid}'"
            )
            for doc in raw_docs:
                text = doc.get("text", "")
                if not text:
                    continue
                events = self._parse_json_document(text)
                if multi:
                    for event in events:
                        event.uuid = uuid
                all_events.extend(events)

        logger.info(
            f"Parsed {len(all_events)} total events from DB documents "
            f"across {len(uuids)} UUID(s)"
        )
        return all_events

    # ── Function interface ──────────────────────────────────────────────

    async def acall(self, state: dict):
        """Fetch raw events from DB by UUID(s), merge, aggregate, and return.

        ``uuids`` (or legacy ``uuid``) can be overridden in *state*;
        ``start_time`` and ``end_time`` restrict processing to events
        overlapping that time window (falls back to function config values).
        """
        with Metrics("StructuredOnlineSumm/Acall", "blue"):
            self.call_schema.validate(state)

            uuids = self._resolve_uuids(state)
            events = self._fetch_events_from_db(uuids)

            start_time = state.get("start_time", self.filter_start_time)
            end_time = state.get("end_time", self.filter_end_time)
            events = self._filter_events_by_time(events, start_time, end_time)

            await self._store_merged_events(events)

            state = await self._build_result(
                state,
                events,
                uuids,
                log_filename="structured_online_events_metrics.json",
            )

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        """Parse and store raw events to DB (no in-memory accumulation)."""
        try:
            logger.info("Processing structured doc %d for online storage", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            with Metrics("StructuredOnlineSumm/aprocess_doc", "red") as bs:
                events = self._parse_json_document(doc)

                if events:
                    logger.info(f"Extracted {len(events)} events from document {doc_i}")
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
        self.db.reset(state)
        self.summary_start_time = None
        self.metrics.reset()
        await asyncio.sleep(0.001)
