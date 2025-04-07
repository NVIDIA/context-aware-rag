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

import os
from pathlib import Path
from re import compile
import traceback
from typing import Optional
import asyncio
from langchain_core.output_parsers import StrOutputParser
from langchain.retrievers import ContextualCompressionRetriever
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.milvus_db import MilvusDBTool
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.utils.globals import DEFAULT_RAG_TOP_K, LLM_TOOL_NAME


class VectorRetrievalFunc(Function):
    """VectorRAG Function"""

    config: dict
    output_parser = StrOutputParser()
    vector_db: MilvusDBTool
    metrics = GraphMetrics()

    def setup(self):
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.vector_db = self.get_tool("vector_db")
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        self.log_dir = os.environ.get("VIA_LOG_DIR", None)
        embeddings_dimension = int(os.environ.get("CA_RAG_EMBEDDINGS_DIMENSION", 1024))
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=embeddings_dimension,
            chunk_overlap=0,
            separators=["\n\n", "\n", "\n-", ".", ";", ",", " ", ""],
        )
        pipeline_compressor = DocumentCompressorPipeline(
            transformers=[splitter, self.vector_db.reranker]
        )
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=pipeline_compressor,
            base_retriever=self.vector_db.vector_db.as_retriever(
                search_kwargs={"filter": {"doc_type": "caption"}, "k": self.top_k}
            ),
        )
        self.g_semantic_sim_chain = RetrievalQA.from_chain_type(
            llm=self.chat_llm, retriever=self.compression_retriever
        )

    async def acall(self, state: dict):
        """QnA function call"""
        if self.log_dir:
            with TimeMeasure("VectorRAG/aprocess-doc/metrics_dump", "yellow"):
                log_path = Path(self.log_dir).joinpath("vector_rag_metrics.json")
                self.metrics.dump_json(log_path.absolute())
        try:
            logger.debug("Running qna with question: %s", state["question"])
            with TimeMeasure("VectorRAG/retrieval", "red"):
                semantic_search_answer = await self.g_semantic_sim_chain.ainvoke(
                    state["question"]
                )
                logger.debug(
                    "Semantic search response: %s", semantic_search_answer["result"]
                )
                state["response"] = semantic_search_answer["result"]
                state["response"] = self.regex_object.sub(r"\g<1>", state["response"])
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        pass

    async def areset(self, expr):
        self.metrics.reset()
        self.vector_db.drop_data("pk > 0")
        await asyncio.sleep(0.01)
