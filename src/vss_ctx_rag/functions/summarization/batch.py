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
import aiohttp
from typing import Dict, Any
import os
import time
from pathlib import Path
from typing import Optional
import traceback
import json
import re

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from schema import Schema

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.health.rag_health import SummaryMetrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger, TimeMeasure
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


DEFAULT_BATCH_ENRICHMENT_PROMPT = (
    "You are tasked with enriching a video summary with additional context that provides relevant information. "
    "The video summary has specific structure with timestamps and categories - maintain this structure. "
    "Integrate the additional context naturally where relevant throughout the summary, enhancing descriptions and explanations. "
    "Do not just append it as a separate section - weave it in contextually where it makes sense.\n\n"
    "Video Summary:\n{video_summary}\n\n"
    "Additional Context:\n{external_context}\n\n"
    "Instructions: Enhance the video summary by incorporating the additional context information where it's relevant and useful."
    "Do not include any introductory phrases, notes, explanations, or comments about how the inputs were combined. Do not reference the video summary or external context. Only provide the enriched summary itself."
)


class BatchSummarization(Function):
    """Batch Summarization Function"""

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
    external_rag_query: Optional[str] = None

    def _extract_external_rag_query(self, text: str) -> tuple[str, str]:
        """Extract external RAG query from text marked with <e> tags."""
        logger.info(f"=== DEBUG: _extract_external_rag_query ===")
        logger.info(f"Input text: {repr(text)}")
        logger.info(f"Input text length: {len(text)}")
        
        # Check for exact character sequences
        logger.info(f"Contains '<e>': {'<e>' in text}")
        
        # Look for the exact sequence at the end
        text_end = text[-100:] if len(text) > 100 else text
        logger.info(f"Last 100 chars: {repr(text_end)}")
        
        # Look for <e>content<e> pattern - match the last occurrence
        pattern = r'<e>(.*?)<e>'
        logger.info(f"Regex pattern: {pattern}")
        
        # Use DOTALL flag to match across newlines
        matches = re.findall(pattern, text, re.DOTALL)
        logger.info(f"Regex matches: {matches}")
        
        clean_text = re.sub(pattern, '', text, flags=re.DOTALL).strip()
        logger.info(f"Clean text: {repr(clean_text)}")
        
        external_rag_query = matches[0] if matches else ''
        logger.info(f"External RAG query: {repr(external_rag_query)}")
        logger.info(f"=== END DEBUG: _extract_external_rag_query ===")
        
        return clean_text, external_rag_query

    async def _get_external_rag_context(self, query: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Get context from external RAG service."""
        custom_server = os.getenv("EXTERNAL_RAG_CUSTOM_SERVER")
        self.external_rag_enabled = os.environ.get("EXTERNAL_RAG_ENABLED", "false").lower() == "true"
        self.external_rag_timeout = int(os.environ.get("EXTERNAL_RAG_TIMEOUT", "30"))
        self.external_rag_collection = [item.strip() for item in os.environ.get("EXTERNAL_RAG_COLLECTION", "").split(',') if item.strip()]
        self.reranker_endpoint = os.environ.get("RERANKER_ENDPOINT")
        self.llm_endpoint = os.environ.get("LLM_ENDPOINT")
        self.embedding_endpoint = os.environ.get("EMBEDDING_ENDPOINT")

        if not custom_server:
            logger.error("EXTERNAL_RAG_ENABLED is true, but EXTERNAL_RAG_CUSTOM_SERVER is not set.")
            return ""

        try:
            headers = {"Content-Type": "application/json"}
            endpoint = f"{custom_server.rstrip('/')}/v1/generate"
            
            payload = {
            "messages": [{"role": "user", "content": query}],
            "use_knowledge_base": True,
            "temperature": 0.2,
            "top_p": 0.7,
            "max_tokens": 1024,
            "reranker_top_k": 2,
            "vdb_top_k": 10,
            "vdb_endpoint": "http://milvus:19530",
            "collection_names": self.external_rag_collection,
            "enable_query_rewriting": True,
            "enable_reranker": True,
            "enable_citations": True,
            "model": "nvidia/llama-3.3-nemotron-super-49b-v1",
            "reranker_model": "nvidia/llama-3.2-nv-rerankqa-1b-v2",
            "embedding_model": "nvidia/llama-3.2-nv-embedqa-1b-v2",
            "stop": [],
            "filter_expr": ''
            }

            if metadata:
                payload["metadata"] = metadata
            if self.llm_endpoint:
                payload["llm_endpoint"] = self.llm_endpoint
            if self.embedding_endpoint:
                payload["embedding_endpoint"] = self.embedding_endpoint
            if self.reranker_endpoint:
                payload["reranker_endpoint"] = self.reranker_endpoint

            logger.info(f"Sending request to: {endpoint}")
            logger.debug(f"Payload: {json.dumps(payload, indent=2)}")

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=30
                ) as response:
                    logger.info(f"Received response status: {response.status}")
                    if response.status != 200:
                        logger.warning(f"Service returned status {response.status}: {await response.text()}")
                        return ""
                    
                    content_type = response.headers.get('Content-Type', '')
                    logger.info(f"Response Content-Type: {content_type}")
                    context = ""  # Initialize

                    if 'text/event-stream' in content_type:
                        logger.info("Processing text/event-stream response...")
                        full_text = ""
                        citations_list = []  # Still collect to see if they are sent
                        async for line_bytes in response.content:
                            line_decoded = line_bytes.decode('utf-8').strip()
                            logger.debug(f"Stream line: {line_decoded}")
                            if line_decoded.startswith('data: '):
                                data_str = line_decoded[6:]
                                if data_str == '[DONE]':
                                    logger.debug("Stream [DONE]")
                                    break
                                try:
                                    event_data = json.loads(data_str)
                                    logger.debug(f"Stream event_data: {event_data}")
                                    if 'choices' in event_data and len(event_data['choices']) > 0:
                                        delta = event_data['choices'][0].get('delta', {})
                                        if 'content' in delta and delta['content'] is not None:
                                            full_text += delta['content']
                                    # Parse citations if present
                                    if 'citations' in event_data:
                                        results = event_data['citations'].get('results', [])
                                        for citation_item in results:
                                            if 'content' in citation_item:
                                                citations_list.append(citation_item['content'])
                                except json.JSONDecodeError:
                                    logger.warning(f"Failed to decode JSON from stream: {data_str}")
                        context = full_text  # Only the main answer
                        if citations_list:  # Log that citations were received, but not appended
                            logger.debug(f"Citations received (but NOT appended to final context): {list(set(citations_list))}")
                    else:
                        logger.info("Processing standard JSON response...")
                        json_response = await response.json()
                        logger.debug(f"JSON response data: {json_response}")
                        context = json_response.get("answer", "")  # Only the main answer
                        # Parse and log citations if present, but don't append
                        citations_data = json_response.get("citations", {})
                        if isinstance(citations_data, dict):
                            citation_results = citations_data.get('results', [])
                            if citation_results:
                                logger.debug(f"Citations received (but NOT appended to final context): {[cite.get('content', '') for cite in citation_results if cite.get('content')]}")
                        elif isinstance(citations_data, list):
                            if citations_data:
                                logger.debug(f"Citations received (but NOT appended to final context): {citations_data}")
                    
                    logger.info("--- Final Parsed Context (Answer Only) ---")
                    logger.info(context[:500] + "..." if len(context) > 500 else context)
                    logger.info("-----------------------------------------")
                    return context

        except aiohttp.ClientConnectorError as e:
            logger.error(f"Connection Error: Could not connect to {custom_server}. Is the server running and accessible? Error: {e}")
            return ""
        except asyncio.TimeoutError:
            logger.error(f"Request timed out after 30 seconds.")
            return ""
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            return ""

    def setup(self):
        # fixed params
        caption_summarization_prompt = self.get_param("prompts", "caption_summarization")
        
        # DEBUG: Log what prompt is received
        logger.info(f"=== DEBUG: BatchSummarization.setup() ===")
        logger.info(f"caption_summarization_prompt received: {caption_summarization_prompt}")
        logger.info(f"EXTERNAL_RAG_ENABLED env var: {os.environ.get('EXTERNAL_RAG_ENABLED', 'NOT_SET')}")
        logger.info(f"EXTERNAL_RAG_CUSTOM_SERVER env var: {os.environ.get('EXTERNAL_RAG_CUSTOM_SERVER', 'NOT_SET')}")
        
        # Store external RAG query for later use, but use clean prompt for batch processing
        self.external_rag_query = None
        if os.environ.get("EXTERNAL_RAG_ENABLED", "false").lower() == "true":
            clean_prompt, external_rag_query = self._extract_external_rag_query(caption_summarization_prompt)
            logger.info(f"=== DEBUG: After extraction ===")
            logger.info(f"clean_prompt: {clean_prompt}")
            logger.info(f"external_rag_query: {external_rag_query}")
            if external_rag_query:
                self.external_rag_query = external_rag_query
                logger.info(f"External RAG query stored for final aggregation: {external_rag_query}")
                # Use clean prompt for batch processing (without <e> tags)
                caption_summarization_prompt = clean_prompt
                logger.info("Using clean caption prompt for batch processing")
            else:
                logger.info("No external RAG query found in caption summarization prompt")
        else:
            logger.info("EXTERNAL_RAG_ENABLED is not set to 'true'")
        
        logger.info(f"=== END DEBUG ===")
        
        self.batch_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", caption_summarization_prompt),
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

        # working params
        self.batcher = Batcher(self.batch_size)
        self.recursion_limit = self.get_param(
            "summ_rec_lim", default=DEFAULT_SUMM_RECURSION_LIMIT
        )

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        self.summary_start_time = None
        self.enable_summary = True

        self.uuid = self.get_param("uuid", default="default")

    async def _process_full_batch(self, batch):
        """Process a full batch immediately"""
        with Metrics(
            "Batch "
            + str(batch._batch_index)
            + " Summary IS LAST "
            + str(batch.as_list()[-1][2]["is_last"]),
            "pink",
        ):
            batch_summary = "."

            logger.info("Batch %d is full. Processing ...", batch._batch_index)
            try:
                with get_openai_callback() as cb:
                    batch_text = " ".join([doc for doc, _, _ in batch.as_list()])
                    batch_summary = await call_token_safe(
                        batch_text,
                        self.batch_pipeline,
                        self.recursion_limit,
                    )
            except Exception as e:
                logger.error(f"Error summarizing batch {batch._batch_index}: {e}")

            self.metrics.summary_tokens += cb.total_tokens
            self.metrics.summary_requests += cb.successful_requests

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

            # get start and end times for the batch's caption_summary document
            min_start_pts = None
            max_end_pts = None
            min_start_ntp = None
            max_end_ntp = None
            min_start_ntp_float = None
            max_end_ntp_float = None
            for _, _, meta in batch.as_list():
                if "start_pts" in meta:
                    if min_start_pts is None:
                        min_start_pts = meta["start_pts"]
                    else:
                        min_start_pts = min(min_start_pts, meta["start_pts"])
                if "end_pts" in meta:
                    if max_end_pts is None:
                        max_end_pts = meta["end_pts"]
                    else:
                        max_end_pts = max(max_end_pts, meta["end_pts"])
                if "start_ntp" in meta:
                    if min_start_ntp is None:
                        min_start_ntp = meta["start_ntp"]
                    else:
                        min_start_ntp = min(min_start_ntp, meta["start_ntp"])
                if "end_ntp" in meta:
                    if max_end_ntp is None:
                        max_end_ntp = meta["end_ntp"]
                    else:
                        max_end_ntp = max(max_end_ntp, meta["end_ntp"])
                if "start_ntp_float" in meta:
                    if min_start_ntp_float is None:
                        min_start_ntp_float = meta["start_ntp_float"]
                    else:
                        min_start_ntp_float = min(
                            min_start_ntp_float, meta["start_ntp_float"]
                        )
                if "end_ntp_float" in meta:
                    if max_end_ntp_float is None:
                        max_end_ntp_float = meta["end_ntp_float"]
                    else:
                        max_end_ntp_float = max(
                            max_end_ntp_float, meta["end_ntp_float"]
                        )

            logger.info(f"Min start pts: {min_start_pts}, Max end pts: {max_end_pts}")
            logger.info(f"Min start ntp: {min_start_ntp}, Max end ntp: {max_end_ntp}")
            logger.info(
                f"Min start ntp float: {min_start_ntp_float}, Max end ntp float: {max_end_ntp_float}"
            )

            logger.info("Batch %d summary: %s", batch._batch_index, batch_summary)
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
            empty_doc_meta = {}
            if doc_meta_sample:
                empty_doc_meta = {
                    key: type(value)() for key, value in doc_meta_sample.items()
                }

            batch_meta = {
                **empty_doc_meta,
                "chunkIdx": -1,
                "batch_i": batch._batch_index,
                "doc_type": "caption_summary",
                "uuid": self.uuid,
                "camera_id": "default",
            }
            if min_start_ntp:
                batch_meta["start_ntp"] = min_start_ntp
            if max_end_ntp:
                batch_meta["end_ntp"] = max_end_ntp
            if min_start_ntp_float:
                batch_meta["start_ntp_float"] = min_start_ntp_float
            if max_end_ntp_float:
                batch_meta["end_ntp_float"] = max_end_ntp_float
            if min_start_pts:
                batch_meta["start_pts"] = min_start_pts
            if max_end_pts:
                batch_meta["end_pts"] = max_end_pts

            # Add the chunk indices if any exist
            if chunk_indices:
                batch_meta["linked_summary_chunks"] = chunk_indices
            # TODO: Use the async method once https://github.com/langchain-ai/langchain-milvus/pull/29 is released
            # await self.db.aadd_summary(summary=batch_summary, metadata=batch_meta)
            logger.debug(f"Metadata being added: {batch_meta}")
            self.db.add_summary(summary=batch_summary, metadata=batch_meta)

        except Exception as e:
            logger.error(f"Error adding summary to database: {e}")

    async def acall(self, state: dict):
        """batch summarization function call"""
        logger.info("Starting batch summarization")
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
                                    logger.info(f"BatchSummarization.acall: Input to aggregation_pipeline (list of batch summaries): {batch}")
                                    results = await self.aggregation_pipeline.ainvoke(
                                        batch
                                    )
                                    logger.info(f"BatchSummarization.acall: Result from aggregation_pipeline: {results[:500]}...")
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

                        # Get initial aggregated summary
                        result = await aggregate_token_safe(batches, self.recursion_limit)
                        
                        # Process external RAG at final aggregation if query was stored
                        if self.external_rag_query:
                            logger.info(f"Processing external RAG at final aggregation with query: {self.external_rag_query}")
                            try:
                                external_context = await self._get_external_rag_context(self.external_rag_query)
                                if external_context:
                                    logger.info("=== BATCH_SUMMARIZATION_ENRICHMENT_PROMPT Logic Start ===")
                                    custom_prompt = os.getenv("BATCH_SUMMARIZATION_ENRICHMENT_PROMPT", "").strip()
                                    logger.info(f"BATCH_SUMMARIZATION_ENRICHMENT_PROMPT env var length: {len(custom_prompt)}")
                                    logger.info(f"BATCH_SUMMARIZATION_ENRICHMENT_PROMPT first 200 chars: {custom_prompt[:200]}...")
                                    prompt_template = custom_prompt if custom_prompt else DEFAULT_BATCH_ENRICHMENT_PROMPT
                                    logger.info(f"Using custom prompt: {bool(custom_prompt)}")
                                    logger.info(f"Final prompt template length: {len(prompt_template)}")
                                    logger.info(f"Final prompt template first 200 chars: {prompt_template[:200]}...")
                                    enrichment_prompt = ChatPromptTemplate.from_template(prompt_template)
                                    logger.info("ChatPromptTemplate created successfully")
                                    enriched_pipeline = enrichment_prompt | self.get_tool(LLM_TOOL_NAME) | self.output_parser
                                    logger.info("Enriched pipeline created successfully")
                                    logger.info(f"Video summary length for enrichment: {len(result)}")
                                    logger.info(f"External context length for enrichment: {len(external_context)}")
                                    result = await enriched_pipeline.ainvoke({
                                        "video_summary": result,
                                        "external_context": external_context
                                    })
                                    logger.info("Video summary enriched with external RAG context successfully")
                                    logger.info("=== BATCH_SUMMARIZATION_ENRICHMENT_PROMPT Logic End ===")
                                else:
                                    logger.info("External RAG returned no context, using original summary")
                            except Exception as e:
                                logger.error(f"External RAG enrichment failed: {e}", exc_info=True)
                                logger.info("Continuing with original summary without enrichment")
                        
                        state["result"] = result
                    logger.info("Summary Aggregation Done")
                    self.metrics.aggregation_tokens = cb.total_tokens
                    logger.info(
                        "Total Tokens: %s, Prompt Tokens: %s, Completion Tokens: %s, "
                        "Successful Requests: %s, Total Cost (USD): $%s"
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
            logger.info(f"Batch Summarization Acall: {state}")
            with Metrics("OffBatchSumm/Acall", "blue"):
                batches = []
                self.call_schema.validate(state)
                stop_time = time.time() + self.timeout
                target_start_batch_index = self.batcher.get_batch_index(
                    state["start_index"]
                )
                target_end_batch_index = self.batcher.get_batch_index(
                    state["end_index"]
                )
                logger.info(f"Target Batch Start: {target_start_batch_index}")
                logger.info(f"Target Batch End: {target_end_batch_index}")
                if target_end_batch_index == -1:
                    max_batch_index = await self.db.aget_max_batch_index(self.uuid)
                    target_end_batch_index = max_batch_index
                    logger.debug(
                        f"Updated target_end_batch_index to {target_end_batch_index}"
                    )

                while time.time() < stop_time:
                    batches = await self.db.aget_text_data(
                        target_start_batch_index, target_end_batch_index, self.uuid
                    )
                    # Sort batches by batch_i field
                    batches.sort(key=lambda x: x["batch_i"])
                    logger.debug(
                        f"Batches Fetched: {[{k: v for k, v in batch.items() if k != 'vector'} for batch in batches]}"
                    )
                    logger.info(f"Number of Batches Fetched: {len(batches)}")
                    # Need ceiling of results/batch_size for correct batch size target end
                    if (
                        len(batches)
                        == target_end_batch_index - target_start_batch_index + 1
                    ):
                        logger.info(
                            f"Need {target_end_batch_index - target_start_batch_index + 1} batches. Moving forward."
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
                            cb = type('obj', (object,), {'total_tokens': 0, 'successful_requests': 0})()
                        self.metrics.summary_tokens += cb.total_tokens
                        self.metrics.summary_requests += cb.successful_requests

                        chunk_indices = list(
                            set(
                                doc_meta.get("chunkIdx")
                                for _, _, doc_meta in batch.as_list()
                            )
                        )

                        if self.graph_db:
                            logger.debug(
                                f"Adding batch summary {batch._batch_index} to Graph DB"
                            )
                            self.graph_db.add_summary(
                                summary=batch_summary,
                                metadata={
                                    "chunkIdx": chunk_indices,
                                    "batch_i": batch._batch_index,
                                    "uuid": doc_meta["uuid"],
                                },
                            )
                        logger.info(
                            f"Found {len(batches)} batches. Taking first {target_end_batch_index - target_start_batch_index + 1} batches."
                        )
                        batches = batches[
                            : target_end_batch_index - target_start_batch_index + 1
                        ]
                        break
                    else:
                        logger.info(
                            f"Need {target_end_batch_index - target_start_batch_index + 1} batches. Waiting ..."
                        )
                        await asyncio.sleep(1)
                        continue

                # Sort batches by batch_i field
                batches.sort(key=lambda x: x["batch_i"])
                logger.info(f"Number of Batches Fetched: {len(batches)}")
                batches = [
                    {k: v for k, v in batch.items() if k == "text"} for batch in batches
                ]

                if len(batches) == 0:
                    state["result"] = ""
                    state["error_code"] = "No batch summaries found"
                    logger.error("No batch summaries found")
                elif len(batches) > 0:
                    with Metrics("summ/acall/batch-aggregation-summary", "pink") as bas:
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
                    try:
                        batch_meta = {
                            **doc_meta,
                            "batch_i": batch._batch_index,
                            "doc_type": "caption_summary",
                        }
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
        self.db.reset(state)
        self.summary_start_time = None
        self.batcher.flush()
        self.metrics.reset()
        await asyncio.sleep(0.001)
