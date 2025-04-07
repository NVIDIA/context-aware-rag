# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""summarization.py: File contains Function class"""

import asyncio
import os
from pathlib import Path
import time
from langchain_community.callbacks import get_openai_callback
from schema import Schema

from vss_ctx_rag.base import Function
from vss_ctx_rag.utils.utils import remove_think_tags
from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.globals import DEFAULT_SUMM_RECURSION_LIMIT, LLM_TOOL_NAME
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.base import RunnableSequence
from langchain_text_splitters import RecursiveCharacterTextSplitter


class BatchSummarization(Function):
    """Batch Summarization Function"""

    config: dict
    batch_prompt: str
    aggregation_prompt: str
    output_parser = StrOutputParser()
    batch_size: int
    curr_batch: str
    curr_batch_size: int
    curr_batch_i: int
    batch_pipeline: RunnableSequence
    aggregation_pipeline: RunnableSequence
    vector_db: StorageTool
    timeout: int = 120  # seconds
    call_schema: Schema = Schema(
        {"start_index": int, "end_index": int}, ignore_extra_keys=True
    )
    metrics = SummaryMetrics()

    def setup(self):
        # fixed params
        self.batch_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_param("prompts", "caption_summarization")),
                ("user", "{input}"),
            ]
        )
        self.aggregation_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.get_param("prompts", "summary_aggregation")),
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
        self.batch_size = self.get_param("params", "batch_size")
        self.vector_db = self.get_tool("vector_db")
        self.timeout = (
            self.get_param("timeout_sec", required=False)
            if self.get_param("timeout_sec", required=False)
            else self.timeout
        )

        # working params
        self.curr_batch_i = 0
        self.batcher = Batcher(self.batch_size)
        self.recursion_limit = (
            self.get_param("summ_rec_lim", required=False)
            if self.get_param("summ_rec_lim", required=False)
            else DEFAULT_SUMM_RECURSION_LIMIT
        )

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.summary_start_time = None
        self.enable_summary = True

    async def acall(self, state: dict):
        """batch summarization function call

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
        with TimeMeasure("OffBatchSumm/Acall", "blue"):
            batches = []
            self.call_schema.validate(state)
            stop_time = time.time() + self.timeout
            target_start_batch_index = self.batcher.get_batch_index(
                state["start_index"]
            )
            target_end_batch_index = self.batcher.get_batch_index(state["end_index"])
            logger.info(f"Target Batch Start: {target_start_batch_index}")
            logger.info(f"Target Batch End: {target_end_batch_index}")
            if target_end_batch_index == -1:
                logger.info(f"Current batch index: {self.curr_batch_i}")
                target_end_batch_index = self.curr_batch_i
            while time.time() < stop_time:
                batches = await self.vector_db.aget_text_data(
                    fields=["text", "batch_i"],
                    filter=f"doc_type == 'caption_summary' and "
                    f"{target_start_batch_index}<=batch_i<={target_end_batch_index}",
                )
                # Sort batches by batch_i field
                batches.sort(key=lambda x: x["batch_i"])
                logger.debug(f"Batches Fetched: {batches}")
                logger.info(f"Number of Batches Fetched: {len(batches)}")

                # Need ceiling of results/batch_size for correct batch size target end
                if (
                    len(batches)
                    == target_end_batch_index - target_start_batch_index + 1
                ):
                    logger.info(
                        f"Need {target_end_batch_index - target_start_batch_index + 1} batches. Moving forward."
                    )
                    break
                else:
                    logger.info(
                        f"Need {target_end_batch_index - target_start_batch_index + 1} batches. Waiting ..."
                    )
                    await asyncio.sleep(1)
                    continue

            if len(batches) == 0:
                state["result"] = ""
                state["error_code"] = "No batch summaries found"
                logger.error("No batch summaries found")
            elif len(batches) > 0:
                with TimeMeasure("summ/acall/batch-aggregation-summary", "pink") as bas:
                    with get_openai_callback() as cb:

                        async def aggregate_token_safe(batch, retries_left):
                            try:
                                with TimeMeasure("OffBatSumm/AggPipeline", "blue"):
                                    results = await self.aggregation_pipeline.ainvoke(
                                        batch
                                    )
                                    return results
                            except Exception as e:
                                if "400" not in str(e):
                                    raise e
                                logger.warning(
                                    f"Received 400 error from LLM endpoint {e}. "
                                    "If this is token length exceeded, resolving now..."
                                )

                                if retries_left <= 0:
                                    logger.debug(
                                        "Maximum recursion depth exceeded. Returning batch as is."
                                    )
                                    return batch

                                if len(batch) == 1:
                                    with TimeMeasure("OffBatSumm/BaseCase", "yellow"):
                                        logger.debug("Base Case, batch size = 1")
                                        text = batch[0]
                                        text_splitter = RecursiveCharacterTextSplitter(
                                            chunk_size=len(text) // 2,
                                            chunk_overlap=50,
                                            length_function=len,
                                            is_separator_regex=False,
                                        )

                                        chunks = text_splitter.split_text(text)
                                        first_half, second_half = chunks[0], chunks[1]

                                        logger.debug(
                                            f"Text exceeds token length. Splitting into "
                                            f"two parts of lengths {len(first_half)} and {len(second_half)}."
                                        )

                                        tasks = [
                                            aggregate_token_safe(
                                                [first_half], retries_left - 1
                                            ),
                                            aggregate_token_safe(
                                                [second_half], retries_left - 1
                                            ),
                                        ]
                                        summaries = await asyncio.gather(*tasks)
                                        combined_summary = "\n".join(summaries)

                                        try:
                                            aggregated = (
                                                await self.aggregation_pipeline.ainvoke(
                                                    [combined_summary]
                                                )
                                            )
                                            return aggregated
                                        except Exception:
                                            logger.debug(
                                                "Error after combining summaries, retrying with combined summary."
                                            )
                                            return await aggregate_token_safe(
                                                [combined_summary], retries_left - 1
                                            )
                                else:
                                    midpoint = len(batch) // 2
                                    first_batch = batch[:midpoint]
                                    second_batch = batch[midpoint:]

                                    logger.debug(
                                        f"Batch size {len(batch)} exceeds token length. "
                                        f"Splitting into two batches of sizes {len(first_batch)} and {len(second_batch)}."
                                    )

                                    tasks = [
                                        aggregate_token_safe(
                                            first_batch, retries_left - 1
                                        ),
                                        aggregate_token_safe(
                                            second_batch, retries_left - 1
                                        ),
                                    ]
                                    results = await asyncio.gather(*tasks)

                                    combined_results = []
                                    for result in results:
                                        if isinstance(result, list):
                                            combined_results.extend(result)
                                        else:
                                            combined_results.append(result)

                                    try:
                                        with TimeMeasure(
                                            "OffBatSumm/CombindAgg", "red"
                                        ):
                                            aggregated = (
                                                await self.aggregation_pipeline.ainvoke(
                                                    combined_results
                                                )
                                            )
                                            return aggregated
                                    except Exception:
                                        logger.debug(
                                            "Error after combining batch summaries, retrying with combined summaries."
                                        )
                                        return await aggregate_token_safe(
                                            combined_results, retries_left - 1
                                        )

                        result = await aggregate_token_safe(
                            batches, self.recursion_limit
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
        try:
            logger.info("Adding doc %d", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            self.vector_db.add_summary(
                summary=doc,
                metadata={**doc_meta, "doc_type": "caption", "batch_i": -1},
            )

            with TimeMeasure("summ/aprocess_doc", "red") as bs:
                if not doc_meta["is_last"] and "file" in doc_meta:
                    if doc_meta["file"].startswith("rtsp://"):
                        # if live stream summarization
                        if "start_ntp" in doc_meta and "end_ntp" in doc_meta:
                            doc = (
                                f"<{doc_meta['start_ntp']}> <{doc_meta['end_ntp']}> "
                                + doc
                            )
                        else:
                            logger.info(
                                "start_ntp or end_ntp not found in doc_meta. "
                                "No timestamp will be added."
                            )
                    else:
                        # if file summmarization
                        if "start_pts" in doc_meta and "end_pts" in doc_meta:
                            doc = (
                                f"<{doc_meta['start_pts'] / 1e9:.2f}> <{doc_meta['end_pts'] / 1e9:.2f}> "
                                + doc
                            )
                        else:
                            logger.info(
                                "start_pts or end_pts not found in doc_meta. "
                                "No timestamp will be added."
                            )
                doc_meta["batch_i"] = doc_i // self.batch_size
                batch = self.batcher.add_doc(doc, doc_i, doc_meta)
                if batch.is_full():
                    with TimeMeasure(
                        "Batch "
                        + str(self.batcher.get_batch_index(doc_i))
                        + " Summary IS LAST "
                        + str(doc_meta["is_last"]),
                        "pink",
                    ):
                        logger.info(
                            "Batch %d is full. Processing ...", batch._batch_index
                        )
                        try:
                            with get_openai_callback() as cb:
                                batch_summary = await self.batch_pipeline.ainvoke(
                                    " ".join([doc for doc, _, _ in batch.as_list()])
                                )
                        except Exception as e:
                            logger.error(
                                f"Error summarizing batch {batch._batch_index}: {e}"
                            )
                            batch_summary = "."
                        self.metrics.summary_tokens += cb.total_tokens
                        self.metrics.summary_requests += cb.successful_requests
                        logger.info(
                            "Batch %d summary: %s", batch._batch_index, batch_summary
                        )
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
                        batch_meta = {
                            **doc_meta,
                            "batch_i": batch._batch_index,
                            "doc_type": "caption_summary",
                        }
                        # TODO: Use the async method once https://github.com/langchain-ai/langchain-milvus/pull/29 is released
                        # await self.vector_db.aadd_summary(summary=batch_summary, metadata=batch_meta)
                        self.vector_db.add_summary(
                            summary=batch_summary, metadata=batch_meta
                        )
                    except Exception as e:
                        logger.error(e)
            if self.summary_start_time is None:
                self.summary_start_time = bs.start_time
            self.metrics.summary_latency = bs.end_time - self.summary_start_time
        except Exception as e:
            logger.error(e)

    async def areset(self, state: dict):
        # TODO: use async method for drop data
        self.vector_db.drop_data(state["expr"])
        self.summary_start_time = None
        self.batcher.flush()
        self.metrics.reset()
        await asyncio.sleep(0.001)
