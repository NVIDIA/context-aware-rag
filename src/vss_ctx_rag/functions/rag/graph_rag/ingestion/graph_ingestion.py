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
import traceback
from pathlib import Path
from typing import Optional, ClassVar, Dict, List

from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import StrOutputParser
from pydantic import Field

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.functions.rag.graph_rag.ingestion.base import GraphIngestion
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_batcher import Batcher
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_EMBEDDING_PARALLEL_COUNT,
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
)
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    DUPLICATE_SCORE_VALUE,
)

from vss_ctx_rag.functions.rag.config import IngestionConfig


@register_function_config("graph_ingestion")
class GraphIngestionConfig(IngestionConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "neo4j": ["db"],
        "arango": ["db"],
        "llm": ["llm"],
    }

    class GraphIngestionParams(IngestionConfig.IngestionParams):
        embedding_parallel_count: Optional[int] = Field(
            default=DEFAULT_EMBEDDING_PARALLEL_COUNT, ge=1
        )
        duplicate_score_value: Optional[float] = Field(default=DUPLICATE_SCORE_VALUE)
        node_types: Optional[List[str]] = Field(default=[])
        relationship_types: Optional[List[str]] = Field(default=[])
        deduplicate_nodes: Optional[bool] = Field(default=False)
        disable_entity_description: Optional[bool] = Field(default=True)
        disable_entity_extraction: Optional[bool] = Field(default=False)
        chunk_size: Optional[int] = Field(default=500)
        chunk_overlap: Optional[int] = Field(default=10)

    params: GraphIngestionParams


@register_function(config=GraphIngestionConfig)
class GraphIngestionFunc(Function):
    """GraphIngestionFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: GraphStorageTool
    metrics = GraphMetrics()

    def setup(self):
        self.graph_db = self.get_tool("db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.rag = self.get_param("rag")
        self.top_k = self.get_param("top_k", default=DEFAULT_RAG_TOP_K)
        self.batch_size = self.get_param("batch_size")
        self.deduplicate_nodes = self.get_param("deduplicate_nodes", default=False)
        self.disable_entity_description = self.get_param(
            "disable_entity_description", default=True
        )
        self.disable_entity_extraction = self.get_param(
            "disable_entity_extraction", default=False
        )
        self.chunk_size = self.get_param("chunk_size")
        self.chunk_overlap = self.get_param("chunk_overlap")
        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        self.batcher = Batcher(self.batch_size)
        self.embedding_parallel_count = self.get_param("embedding_parallel_count")
        self.duplicate_score_value = self.get_param("duplicate_score_value")
        node_types = self.get_param("node_types")
        if len(node_types) > 0:
            node_types = node_types.split(",")
        else:
            node_types = ["Person", "Vehicle", "Location", "Object"]
        relationship_types = self.get_param("relationship_types")
        if len(relationship_types) > 0:
            relationship_types = relationship_types.split(",")
        else:
            relationship_types = []
        logger.info(f"Node types: {node_types}")

        logger.info(f"Embedding parallel count: {self.embedding_parallel_count}")
        self.graph_ingestion = GraphIngestion(
            batcher=self.batcher,
            llm=self.chat_llm,
            graph_db=self.graph_db,
            embedding_parallel_count=self.embedding_parallel_count,
            node_types=node_types,
            relationship_types=relationship_types,
            duplicate_score_value=self.duplicate_score_value,
            deduplicate_nodes=self.deduplicate_nodes,
            disable_entity_description=self.disable_entity_description,
            disable_entity_extraction=self.disable_entity_extraction,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        self.graph_create_start = None

    async def acall(self, state: dict):
        logger.debug(f"Graph Extraction Acall {state}")

        with Metrics("GraphRAG/Acall/graph-extraction/postprocessing", "green") as tm:
            await self.graph_ingestion.apost_process(
                uuid=state.get("uuid", "default"),
                camera_id=state.get("camera_id", ""),
            )
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
        with Metrics(
            "GraphRAG/aprocess-doc:", "blue", span_kind=Metrics.SPAN_KIND["CHAIN"]
        ) as tm:
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
                with Metrics(
                    "GraphRAG/aprocess-doc/graph-create: "
                    + str(self.batcher.get_batch_index(doc_i)),
                    "green",
                    span_kind=Metrics.SPAN_KIND["CHAIN"],
                ) as tm:
                    try:
                        tm.input({"batch": batch})
                        with get_openai_callback() as cb:
                            await self.graph_ingestion.acreate_graph(batch)
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
                        tm.error(e)
                        logger.error(traceback.format_exc())
                        logger.error(
                            "GraphRAG/aprocess-doc Failed with error %s\n Skipping...",
                            e,
                        )
                        raise e
        if self.graph_create_start is None:
            self.graph_create_start = tm.start_time
        self.metrics.graph_create_latency = tm.end_time - self.graph_create_start
        return "Success"

    async def areset(self, state: dict):
        self.batcher.flush()
        self.graph_create_start = None
        self.graph_ingestion.reset()
        self.metrics.reset()
        self.graph_db.reset(state)

        await asyncio.sleep(0.01)
