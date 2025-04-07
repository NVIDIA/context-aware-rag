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

import asyncio
import os
from re import compile
import traceback

from langchain_core.output_parsers import StrOutputParser

from vss_ctx_rag.base import Function
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB
from vss_ctx_rag.tools.health.rag_health import GraphMetrics
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.rag.graph_rag.graph_retrieval import GraphRetrieval
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_CHAT_HISTORY,
)
from langchain_core.messages import HumanMessage, AIMessage
from vss_ctx_rag.utils.utils import remove_think_tags


class GraphRetrievalFunc(Function):
    """GraphRetrievalFunc Function"""

    config: dict
    output_parser = StrOutputParser()
    graph_db: Neo4jGraphDB
    metrics = GraphMetrics()

    def setup(self):
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.top_k = (
            self.get_param("params", "top_k", required=False)
            if self.get_param("params", "top_k", required=False)
            else DEFAULT_RAG_TOP_K
        )
        self.multi_channel = (
            self.get_param("params", "multi_channel", required=False)
            if self.get_param("params", "multi_channel", required=False)
            else DEFAULT_MULTI_CHANNEL
        )
        self.chat_history = self.get_param("params", "chat_history", required=False)
        if self.chat_history is None:
            self.chat_history = DEFAULT_CHAT_HISTORY

        uuid = self.get_param("params", "uuid", required=False)
        self.log_dir = os.environ.get("VIA_LOG_DIR", None)

        try:
            self.graph_retrieval = GraphRetrieval(
                llm=self.chat_llm,
                graph=self.graph_db,
                multi_channel=self.multi_channel,
                uuid=uuid,
                top_k=self.top_k,
            )
        except Exception as e:
            logger.error(f"Error initializing GraphRetrieval: {e}")
            raise
        self.regex_object = compile(r"<(\d+[.]\d+)>")

    async def acall(self, state: dict) -> dict:
        try:
            question = state.get("question", "").strip()
            if not question:
                raise ValueError("No input provided in state.")

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.graph_retrieval.clear_chat_history()
                state["response"] = "Cleared chat history"
                return state

            with TimeMeasure("GraphRetrieval/HumanMessage", "blue"):
                user_message = HumanMessage(content=question)
                self.graph_retrieval.add_message(user_message)

            docs = self.graph_retrieval.retrieve_documents()

            if docs:
                formatted_docs = self.graph_retrieval.process_documents(docs)
                ai_response = self.graph_retrieval.get_response(
                    question, formatted_docs
                )
                answer = remove_think_tags(ai_response.content)
                if self.chat_history:
                    with TimeMeasure("GraphRetrieval/AIMsg", "red"):
                        ai_message = AIMessage(content=answer)
                        self.graph_retrieval.add_message(ai_message)

                    self.graph_retrieval.summarize_chat_history()

                    logger.debug("Summarizing chat history thread started.")
                else:
                    self.graph_retrieval.clear_chat_history()
            else:
                formatted_docs = "No documents retrieved."
                answer = "Sorry, I don't see that in the video."
                self.graph_retrieval.chat_history.messages.pop()

            state["response"] = answer
            state["response"] = self.regex_object.sub(r"\g<1>", state["response"])

            if "formatted_docs" in state:
                state["formatted_docs"].append(formatted_docs)
            else:
                state["formatted_docs"] = [formatted_docs]

        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error("Error in QA %s", str(e))
            state["response"] = "That didn't work. Try another question."

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, state: dict):
        self.graph_retrieval.clear_chat_history()
        await asyncio.sleep(0.01)
