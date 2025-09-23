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

"""offline_batch.py: File contains OfflineBatchSummarization Function class"""

import asyncio
import os
from pathlib import Path
from typing import Dict

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from schema import Schema

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_SUMM_RECURSION_LIMIT,
    DEFAULT_SUMM_TIMEOUT_SEC,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.utils.utils import (
    remove_think_tags,
    call_token_safe,
    add_timestamps_to_doc,
)
from vss_ctx_rag.functions.summarization.config import SummarizationConfig
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)


@register_function_config("offline_summarization")
class OfflineBatchSummarizationConfig(SummarizationConfig):
    pass


@register_function(config=OfflineBatchSummarizationConfig)
class OfflineBatchSummarization(Function):
    """Offline Batch Summarization Function - processes all batches in acall instead of during document processing"""

    config: dict
    batch_prompt: str
    aggregation_prompt: str
    output_parser = StrOutputParser()
    batch_size: int
    curr_batch: str
    curr_batch_size: int
    batch_pipeline: RunnableSequence
    aggregation_pipeline: RunnableSequence
    db: StorageTool
    timeout: int = DEFAULT_SUMM_TIMEOUT_SEC  # seconds
    call_schema: Schema = Schema(
        {"start_index": int, "end_index": int}, ignore_extra_keys=True
    )
    metrics = SummaryMetrics()
    uuid: str
    batch_summaries: Dict[int, str]

    def setup(self):
        # fixed params
        prompts = self.get_param("prompts")
        self.batch_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.get("caption_summarization")),
                ("user", "{input}"),
            ]
        )
        self.aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompts.get("summary_aggregation")),
                ("user", "{input}"),
            ]
        )
        self.output_parser = StrOutputParser()
        self.batch_pipeline = (
            self.batch_prompt
            | self.get_tool(LLM_TOOL_NAME)
            | self.output_parser
            | remove_think_tags
        )
        self.aggregation_pipeline = (
            self.aggregation_prompt
            | self.get_tool(LLM_TOOL_NAME)
            | self.output_parser
            | remove_think_tags
        )
        self.batch_size = self.get_param("batch_size")
        self.db = self.get_tool("db")
        self.timeout = self.get_param("timeout_sec", default=DEFAULT_SUMM_TIMEOUT_SEC)
        self.batcher = Batcher(self.batch_size)
        self.recursion_limit = self.get_param(
            "summ_rec_lim", default=DEFAULT_SUMM_RECURSION_LIMIT
        )

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.summary_start_time = None
        self.enable_summary = True

        self.uuid = self.get_param("uuid", default="default")

        self.batch_summaries = {}

    async def _process_accumulated_batches(self):
        """Process all batches accumulated in the batcher and generate summaries"""
        all_batches = self.batcher.get_all_full_batches()
        logger.info(f"Processing {len(all_batches)} batches")

        with Metrics("OfflineBatchSumm/ProcessAccumulatedBatches", "green"):
            tasks = [self._process_single_batch(batch) for batch in all_batches]

            if tasks:
                batch_summaries = await asyncio.gather(*tasks)
                for batch, batch_summary in batch_summaries:
                    if batch_summary is not None:
                        self.batch_summaries[batch.get_batch_index()] = batch_summary

    async def _process_single_batch(self, batch):
        """Process a single batch and store its summary"""
        with Metrics(
            f"OfflineBatchSumm/ProcessBatch_{batch.get_batch_index()}",
            "pink",
        ):
            logger.info(
                f"Processing batch {batch.get_batch_index()} with {len(batch.as_list())} documents"
            )

            try:
                with get_openai_callback() as cb:
                    batch_text = " ".join([doc for doc, _, _ in batch.as_list()])
                    batch_summary = await call_token_safe(
                        batch_text, self.batch_pipeline, self.recursion_limit
                    )
            except Exception as e:
                logger.error(f"Error summarizing batch {batch.get_batch_index()}: {e}")
                batch_summary = "."

            self.metrics.summary_tokens += cb.total_tokens
            self.metrics.summary_requests += cb.successful_requests

            # Collect chunk indices
            chunk_indices = []
            doc_meta_sample = None
            for _, _, doc_meta in batch.as_list():
                if doc_meta_sample is None:
                    doc_meta_sample = doc_meta
                if "chunkIdx" in doc_meta and doc_meta["chunkIdx"] is not None:
                    chunk_indices.append(doc_meta["chunkIdx"])

            # Remove duplicates
            if chunk_indices:
                chunk_indices = list(set(chunk_indices))

            logger.info(f"Batch {batch.get_batch_index()} summary: {batch_summary}")
            logger.info(
                "Total Tokens: %s, "
                "Prompt Tokens: %s, "
                "Completion Tokens: %s, "
                "Successful Requests: %s, "
                "Total Cost (USD): $%s"
                % (
                    cb.total_tokens,
                    cb.prompt_tokens,
                    cb.completion_tokens,
                    cb.successful_requests,
                    cb.total_cost,
                ),
            )

            try:
                if doc_meta_sample:
                    empty_doc_meta = {
                        key: type(value)() for key, value in doc_meta_sample.items()
                    }
                else:
                    empty_doc_meta = {}

                batch_meta = {
                    **empty_doc_meta,
                    "chunkIdx": -1,
                    "batch_i": batch.get_batch_index(),
                    "doc_type": "caption_summary",
                    "uuid": self.uuid,
                    "camera_id": "default",
                }

                if chunk_indices:
                    batch_meta["linked_summary_chunks"] = chunk_indices

                logger.debug(f"Metadata being added: {batch_meta}")
                self.db.add_summary(summary=batch_summary, metadata=batch_meta)

            except Exception as e:
                logger.error(f"Error adding summary to database: {e}")

            return batch, batch_summary

    async def acall(self, state: dict):
        """offline batch summarization function call - processes all accumulated docs first, then aggregates

        Args:
            state (dict): should validate against call_schema
        Returns:
            dict: the state dict will contain result:
            {
                # ...
                # The following key is overwritten or added
                "result" : "summary",
                "error_code": "Error String" # Optional
            }
        """
        with Metrics("OfflineBatchSumm/Acall", "blue"):
            self.call_schema.validate(state)

            await self._process_accumulated_batches()

            target_start_batch_index = self.batcher.get_batch_index(
                state["start_index"]
            )
            target_end_batch_index = self.batcher.get_batch_index(state["end_index"])
            logger.info(f"Target Batch Start: {target_start_batch_index}")
            logger.info(f"Target Batch End: {target_end_batch_index}")
            if target_end_batch_index == -1:
                max_batch_index = await self.db.aget_max_batch_index(self.uuid)
                target_end_batch_index = max_batch_index
                logger.debug(
                    f"Updated target_end_batch_index to {target_end_batch_index}"
                )

            batches = []
            for batch_index in range(
                target_start_batch_index, target_end_batch_index + 1
            ):
                if batch_index in self.batch_summaries:
                    batches.append(self.batch_summaries[batch_index])

            logger.info(f"Number of Batches Retrieved from memory: {len(batches)}")

            if len(batches) == 0:
                state["result"] = ""
                state["error_code"] = "No batch summaries found"
                logger.error("No batch summaries found")
            elif len(batches) > 0:
                with Metrics(
                    "OfflineBatchSumm/acall/batch-aggregation-summary", "pink"
                ) as bas:
                    with get_openai_callback() as cb:
                        result = await call_token_safe(
                            batches, self.aggregation_pipeline, self.recursion_limit
                        )
                        state["result"] = result
                    logger.info("Summary Aggregation Done")
                    self.metrics.aggregation_tokens = cb.total_tokens
                    logger.info(
                        "Total Tokens: %s, "
                        "Prompt Tokens: %s, "
                        "Completion Tokens: %s, "
                        "Successful Requests: %s, "
                        "Total Cost (USD): $%s"
                        % (
                            cb.total_tokens,
                            cb.prompt_tokens,
                            cb.completion_tokens,
                            cb.successful_requests,
                            cb.total_cost,
                        ),
                    )
                self.metrics.aggregation_latency = bas.execution_time
        if self.log_dir:
            log_path = Path(self.log_dir).joinpath("summary_metrics.json")
            self.metrics.dump_json(log_path.absolute())
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        """Process document by accumulating it for later batch processing instead of immediate processing"""
        try:
            logger.info("Accumulating doc %d for offline processing", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            with Metrics("OfflineBatchSumm/aprocess_doc", "red") as bs:
                # Add timestamps to document using utility function
                doc = add_timestamps_to_doc(doc, doc_meta)

                # Store document in batcher for later processing instead of processing immediately
                batch = self.batcher.add_doc(doc, doc_i, doc_meta)

                logger.debug(f"Added document {doc_i} to batch {batch._batch_index}")

            if self.summary_start_time is None:
                self.summary_start_time = bs.start_time
            self.metrics.summary_latency = bs.end_time - self.summary_start_time
        except Exception as e:
            logger.error(e)

    async def areset(self, state: dict):
        """Reset the function state including accumulated documents"""
        # TODO: use async method for drop data
        self.db.reset(state)
        self.summary_start_time = None
        self.batcher.flush()
        self.metrics.reset()
        # Clear in-memory batch summaries
        self.batch_summaries.clear()
        await asyncio.sleep(0.001)
