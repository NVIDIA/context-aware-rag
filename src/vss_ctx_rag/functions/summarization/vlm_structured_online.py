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

from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
    FunctionModel,
)

from vss_ctx_rag.functions.summarization.vlm_structured_base import (
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

    async def acall(self, state: dict):
        """Fetch raw events from DB by UUID(s), merge, aggregate, and return.

        ``uuids`` (or legacy ``uuid``) can be overridden in *state*;
        ``start_time`` and ``end_time`` restrict processing to events
        overlapping that time window (falls back to function config values).
        """
        with Metrics("StructuredOnlineSumm/Acall", "blue"):
            self.call_schema.validate(state)

            uuids = self._resolve_uuids(state)

            start_time = state.get("start_time", self.filter_start_time)
            end_time = state.get("end_time", self.filter_end_time)
            events = await self._fetch_events_from_db(uuids, start_time, end_time)

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
            if doc_meta.get("doc_type") == "event_list":
                self._store_event_list(doc, doc_i, doc_meta)
                return
            logger.info("Processing structured doc %d for online storage", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            with Metrics("StructuredOnlineSumm/aprocess_doc", "red") as bs:
                events, needs_type = self._parse_json_document(doc, doc_meta)
                if needs_type:
                    uuid = doc_meta.get("uuid", self.uuids[0] if self.uuids else "")
                    inferred = await self._infer_event_types(needs_type, uuid=uuid)
                    events.extend(inferred)

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
