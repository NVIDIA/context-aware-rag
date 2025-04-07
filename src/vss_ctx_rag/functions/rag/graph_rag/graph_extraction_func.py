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

"""graph_rag.py: File contains Function class"""

import asyncio
import os
from pathlib import Path
import traceback
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_community.callbacks import get_openai_callback

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.graph_extraction import GraphExtraction
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_EMBEDDING_PARALLEL_COUNT,
)
from vss_ctx_rag.functions.rag.graph_rag.constants import QUERY_TO_DELETE_UUID_GRAPH


class GraphExtractionFunc(Function):
    """GraphExtractionFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: Neo4jGraphDB
    metrics = GraphMetrics()

    def setup(self):
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.rag = self.get_param("rag")
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.batch_size = self.get_param("params", "batch_size")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.batcher = Batcher(self.batch_size)
        uuid = (
            self.get_param("params", "uuid", required=False)
            if self.get_param("params", "uuid", required=False)
            else "default"
        )
        self.embedding_parallel_count = (
            self.get_param("params", "embedding_parallel_count", required=False)
            if self.get_param("params", "embedding_parallel_count", required=False)
            else DEFAULT_EMBEDDING_PARALLEL_COUNT
        )
        logger.info(f"Embedding parallel count: {self.embedding_parallel_count}")
        self.graph_extraction = GraphExtraction(
            batcher=self.batcher,
            uuid=uuid,
            llm=self.chat_llm,
            graph=self.graph_db,
            embedding_parallel_count=self.embedding_parallel_count,
        )
        self.graph_create_start = None

    async def acall(self, state: dict):
        logger.debug(f"Graph Extraction Acall {state}")
        with TimeMeasure(
            "GraphRAG/Acall/graph-extraction/postprocessing", "green"
        ) as tm:
            await self.graph_extraction.apost_process()
        self.metrics.graph_post_process_latency = tm.execution_time

        # Dump Graph RAG Metrics after all the add_doc and create_graph calls
        # When acall happens, all the aprocess_docs are complete and we want to publish the
        # total time taken in aprocess_doc which we can't do in aprocess_doc itself.
        if self.log_dir:
            log_path = Path(self.log_dir).joinpath("graph_rag_metrics.json")
            self.metrics.dump_json(log_path.absolute())
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        """QnA process doc call"""
        with TimeMeasure("GraphRAG/aprocess-doc:", "blue") as tm:
            if not doc_meta["is_last"]:
                if doc_meta["file"].startswith("rtsp://"):
                    # if live stream summarization
                    doc = f"<{doc_meta['start_ntp']}> <{doc_meta['end_ntp']}> " + doc
                else:
                    # if file summmarization
                    doc = (
                        f"<{doc_meta['start_pts'] / 1e9:.2f}> <{doc_meta['end_pts'] / 1e9:.2f}> "
                        + doc
                    )
            batch = self.batcher.add_doc(doc, doc_i=doc_i, doc_meta=doc_meta)
            if batch.is_full():
                with TimeMeasure(
                    "GraphRAG/aprocess-doc/graph-create: "
                    + str(self.batcher.get_batch_index(doc_i)),
                    "green",
                ):
                    try:
                        with get_openai_callback() as cb:
                            await self.graph_extraction.acreate_graph(batch)
                        logger.info(
                            "GraphRAG Creation for %d docs\n"
                            "Total Tokens: %s, "
                            "Prompt Tokens: %s, "
                            "Completion Tokens: %s, "
                            "Successful Requests: %s, "
                            "Total Cost (USD): $%s"
                            % (
                                batch._batch_size,
                                cb.total_tokens,
                                cb.prompt_tokens,
                                cb.completion_tokens,
                                cb.successful_requests,
                                cb.total_cost,
                            ),
                        )
                        self.metrics.graph_create_tokens += cb.total_tokens
                        self.metrics.graph_create_requests += cb.successful_requests
                    except Exception as e:
                        logger.error(traceback.format_exc())
                        logger.error(
                            "GraphRAG/aprocess-doc Failed with error %s\n Skipping...",
                            e,
                        )
                        return "Failed"
        if self.graph_create_start is None:
            self.graph_create_start = tm.start_time
        self.metrics.graph_create_latency = tm.end_time - self.graph_create_start
        return "Success"

    async def areset(self, state: dict):
        self.batcher.flush()
        self.graph_create_start = None
        self.graph_extraction.reset()
        self.metrics.reset()
        if "uuid" in state and state["uuid"] is not None:
            self.graph_db.run_cypher_query(
                QUERY_TO_DELETE_UUID_GRAPH, params={"uuid": state["uuid"]}
            )
        else:
            self.graph_db.run_cypher_query("MATCH (n) DETACH DELETE n")

        await asyncio.sleep(0.01)
