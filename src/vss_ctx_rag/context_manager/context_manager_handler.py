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

"""Context manager handler implementation.

This module handles managing the input to LLM by calling the handlers of all
the tools it has access to.
"""

import asyncio
import copy
import traceback
import time
import json
from typing import Dict, Optional
import os

from vss_ctx_rag.base import Function
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.functions.notification import Notifier
from vss_ctx_rag.functions.summarization import (
    BatchSummarization,
)
from vss_ctx_rag.tools.llm import ChatOpenAITool
from vss_ctx_rag.tools.notification import AlertSSETool
from vss_ctx_rag.tools.storage import MilvusDBTool, Neo4jGraphDB
from vss_ctx_rag.utils.globals import (
    DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE,
    DEFAULT_LLM_PARAMS,
    DEFAULT_GRAPH_RAG_BATCH_SIZE,
    DEFAULT_RAG_TOP_K,
    LLM_TOOL_NAME,
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_CHAT_HISTORY,
)
from vss_ctx_rag.functions.rag.chat_function import ChatFunction

from vss_ctx_rag.functions.rag.graph_rag.graph_extraction_func import (
    GraphExtractionFunc,
)
from vss_ctx_rag.functions.rag.graph_rag.graph_retrieval_func import GraphRetrievalFunc
from vss_ctx_rag.functions.rag.vector_rag.vector_retrieval_func import (
    VectorRetrievalFunc,
)
from vss_ctx_rag.utils.utils import RequestInfo, is_openai_model


class ContextManagerHandler:
    """Main controller for RAG system operations.

    This class orchestrates the flow of operations in the RAG system by:
    1. Managing function registration and execution
    2. Handling configuration updates
    3. Coordinating between different handlers (chat, summarization, etc.)
    4. Managing the lifecycle of LLM instances and database connections
    """

    # TODO: Is last separately for live stream case
    # TODO: How do we customize prompts from VIA-UI
    # TODO: Make the functions a list
    # TODO: Unit test for blocking function call when calling another function. Does add_doc block too?
    def __init__(
        self,
        config: Dict,
        process_index: int,
        req_info: Optional[RequestInfo] = None,
    ) -> None:
        """Initialize the context manager handler.

        Args:
            config: Configuration dictionary containing system settings
            process_index: Unique identifier for this handler instance
            req_info: Optional request-specific information
        """
        logger.info(f"Initializing Context Manager Handler no.: {process_index}")

        self._functions: dict[str, Function] = {}
        self.config = config
        self.auto_indexing: Optional[bool] = None
        self.curr_doc_index: int = -1
        self.rag_type = None
        self.milvus_db: MilvusDBTool = None
        self.chat_llm: ChatOpenAITool = None
        self.llm: ChatOpenAITool = None
        self.notification_llm: ChatOpenAITool = None
        self._process_index = process_index
        self.neo4j_uri = None
        self.neo4j_username = None
        self.neo4j_password = None
        self.neo4jDB: Neo4jGraphDB = None
        self.configure_init(config, req_info)

    def setup_neo4j(self, chat_config: Dict):
        try:
            self.neo4j_uri = os.getenv("GRAPH_DB_URI")
            if not self.neo4j_uri:
                raise ValueError("GRAPH_DB_URI not set. Please set GRAPH_DB_URI.")
            self.neo4j_username = os.getenv("GRAPH_DB_USERNAME")
            if not self.neo4j_username:
                raise ValueError(
                    "GRAPH_DB_USERNAME not set. Please set GRAPH_DB_USERNAME."
                )
            self.neo4j_password = os.getenv("GRAPH_DB_PASSWORD")
            if not self.neo4j_password:
                raise ValueError(
                    "GRAPH_DB_PASSWORD not set. Please set GRAPH_DB_PASSWORD."
                )
            if (
                self.neo4j_uri is not None
                and self.neo4j_username is not None
                and self.neo4j_password is not None
            ):
                self.neo4jDB = Neo4jGraphDB(
                    url=self.neo4j_uri,
                    username=self.neo4j_username,
                    password=self.neo4j_password,
                    embedding_model_name=chat_config["embedding"]["model"],
                    embedding_base_url=chat_config["embedding"]["base_url"],
                )
        except Exception as e:
            logger.error(f"Error setting up Neo4j: {e}")
            raise e

    def configure_init(self, config: Dict, req_info: Optional[RequestInfo] = None):
        """Initialize system components based on configuration.

        Sets up LLM instances, database connections, and function handlers
        based on the provided configuration.

        Args:
            config: System configuration dictionary
            req_info: Optional request-specific information
        """
        logger.debug(
            f"Configuring init for {self._process_index} with config: {config}"
        )
        # Init time Milvus DB config
        chat_config = copy.deepcopy(config.get("chat"))
        summ_config = copy.deepcopy(config.get("summarization"))
        collection_name = "summary_till_now_" + str(time.time()).replace(".", "_")
        if req_info and req_info.uuid:
            collection_name = "summary_till_now_" + req_info.uuid
        self.milvus_db = MilvusDBTool(
            collection_name=collection_name,
            host=config["milvus_db_host"],
            port=config["milvus_db_port"],
            reranker_base_url=chat_config["reranker"]["base_url"],
            reranker_model_name=chat_config["reranker"]["model"],
            embedding_base_url=chat_config["embedding"]["base_url"],
            embedding_model_name=chat_config["embedding"]["model"],
        )
        # Init time Notification config
        notification_config = config.get("notification")
        if notification_config and notification_config.get("enable"):
            notification_llm_params = notification_config.get("llm")
            if is_openai_model(notification_llm_params["model"]):
                api_key = os.environ["OPENAI_API_KEY"]
            else:
                api_key = config["api_key"]
            logger.info(
                "Using %s as the notification llm", notification_llm_params["model"]
            )
            notification_llm = ChatOpenAITool(
                api_key=api_key, **notification_llm_params
            )
            self.add_function(
                Notifier("notification")
                .add_tool(LLM_TOOL_NAME, notification_llm)
                .add_tool(
                    "notification_tool",
                    AlertSSETool(endpoint=notification_config.get("endpoint")),
                )
                .config(**notification_config)
                .done()
            )
        else:
            logger.info("Notifications disabled")
        # Init time Chat config
        chat_config = copy.deepcopy(config.get("chat"))
        chat_llm_params = (
            chat_config.get(LLM_TOOL_NAME, DEFAULT_LLM_PARAMS)
            if chat_config
            else DEFAULT_LLM_PARAMS
        )
        if is_openai_model(chat_llm_params["model"]):
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = config["api_key"]
        logger.info("Using %s as the chat llm", chat_llm_params["model"])
        self.chat_llm = ChatOpenAITool(api_key=api_key, **chat_llm_params)
        # Init time Summarization config
        summ_config = copy.deepcopy(config.get("summarization"))
        llm_params = summ_config.get(LLM_TOOL_NAME, DEFAULT_LLM_PARAMS)
        if is_openai_model(llm_params["model"]):
            api_key = os.environ["OPENAI_API_KEY"]
        else:
            api_key = config["api_key"]
        logger.info("Using %s as the summarization llm", llm_params["model"])
        self.llm = ChatOpenAITool(api_key=api_key, **llm_params)
        # Init time Neo4j config
        if chat_config.get("rag", None) == "graph-rag":
            self.setup_neo4j(chat_config)
            self.rag_type = chat_config.get("rag")
        if chat_config.get("rag", None) == "vector-rag":
            self.rag_type = chat_config.get("rag")

        if req_info:
            self.configure_update(config, req_info)

    def configure_update(self, config: Dict, req_info):
        try:
            caption_summarization_prompt = ""
            summary_aggregation_prompt = ""
            if req_info:
                caption_summarization_prompt = req_info.caption_summarization_prompt
                summary_aggregation_prompt = req_info.summary_aggregation_prompt
            summ_config = copy.deepcopy(config.get("summarization"))

            try:
                self.default_caption_prompt = summ_config["prompts"]["caption"]
                caption_summarization_prompt = (
                    caption_summarization_prompt
                    or summ_config["prompts"]["caption_summarization"]
                )
                summ_config["prompts"]["caption_summarization"] = (
                    caption_summarization_prompt
                )
                summary_aggregation_prompt = summary_aggregation_prompt or (
                    summ_config["prompts"]["summary_aggregation"]
                    if "summary_aggregation" in summ_config["prompts"]
                    else ""
                )
                summ_config["prompts"]["summary_aggregation"] = (
                    summary_aggregation_prompt
                )
            except Exception as e:
                raise ValueError("Prompt(s) missing!") from e

            enable_summarization = True
            """
            TODO: Remove this once we have a way to disable summarization
            if req_info is None or req_info.summarize is None:
                enable_summarization = summ_config["enable"]
            else:
                enable_summarization = req_info.summarize
            """

            if enable_summarization and self.get_function("summarization") is None:
                if summ_config["method"] == "batch":
                    summ_config["params"] = summ_config.get(
                        "params",
                        {
                            "batch_size": DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE,
                        },
                    )
                    summ_config["params"]["batch_size"] = summ_config["params"].get(
                        "batch_size", DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE
                    )
                    try:
                        if req_info and req_info.is_live:
                            logger.debug("Req Info: %s", req_info.summary_duration)
                            logger.debug("Req Info: %s", req_info.chunk_size)
                            summ_config["params"]["batch_size"] = int(
                                req_info.summary_duration / req_info.chunk_size
                            )
                            logger.info(
                                "Overriding batch size to %s for live stream",
                                summ_config["params"]["batch_size"],
                            )
                    except Exception as e:
                        logger.error(
                            "Overriding batch size failed for live stream: %s", e
                        )
                    self.add_function(
                        BatchSummarization("summarization")
                        .add_tool(LLM_TOOL_NAME, self.llm)
                        .add_tool("vector_db", self.milvus_db)
                        .config(**summ_config)
                        .done()
                    )
                else:
                    # should never reach here. Should be validated by the config schema
                    raise ValueError("Incorrect summarization config")
            elif enable_summarization is False:
                self.remove_function("summarization")
                logger.info("Summarization disabled with the API call")
            chat_config = copy.deepcopy(config.get("chat"))

            if (
                req_info
                and req_info.rag_type
                and req_info.enable_chat
                and (
                    req_info.rag_type == "vector-rag"
                    or req_info.rag_type == "graph-rag"
                )
            ):
                chat_config["rag"] = req_info.rag_type
            if req_info is None or req_info.enable_chat:
                if (
                    chat_config["rag"] != "vector-rag"
                    and chat_config["rag"] != "graph-rag"
                ):
                    self.remove_function("chat")
                    logger.info(
                        "Both graph_rag and vector_rag are disabled. Q&A is disabled"
                    )
                elif chat_config["rag"] != self.rag_type:
                    self.remove_function("chat")
                    logger.info(
                        f"Removing chat function due to rag type change from {self.rag_type} to {chat_config['rag']}"
                    )
                    self.rag_type = None
                if self.get_function("chat") is None:
                    logger.info("Setting up QnA, rag type: %s", chat_config["rag"])

                    chat_config["params"] = chat_config.get(
                        "params",
                        {
                            "batch_size": DEFAULT_GRAPH_RAG_BATCH_SIZE,
                            "top_k": DEFAULT_RAG_TOP_K,
                        },
                    )
                    chat_config["params"]["batch_size"] = chat_config["params"].get(
                        "batch_size", DEFAULT_GRAPH_RAG_BATCH_SIZE
                    )
                    chat_config["params"]["top_k"] = chat_config["params"].get(
                        "top_k", DEFAULT_RAG_TOP_K
                    )
                    chat_config["params"]["multi_channel"] = chat_config["params"].get(
                        "multi_channel", DEFAULT_MULTI_CHANNEL
                    )
                    chat_config["params"]["chat_history"] = chat_config["params"].get(
                        "chat_history", DEFAULT_CHAT_HISTORY
                    )
                    if chat_config["rag"] == "graph-rag":
                        if self.neo4jDB is None:
                            self.setup_neo4j(chat_config)
                        self.add_function(
                            ChatFunction("chat")
                            .add_function(
                                "extraction_function",
                                GraphExtractionFunc("extraction_function")
                                .add_tool("graph_db", self.neo4jDB)
                                .add_tool(LLM_TOOL_NAME, self.chat_llm)
                                .config(**chat_config)
                                .done(),
                            )
                            .add_function(
                                "retrieval_function",
                                GraphRetrievalFunc("retrieval_function")
                                .add_tool("graph_db", self.neo4jDB)
                                .add_tool(LLM_TOOL_NAME, self.chat_llm)
                                .config(**chat_config)
                                .done(),
                            )
                            .config(**chat_config)
                            .done(),
                        )
                        self.rag_type = "graph-rag"
                    elif chat_config["rag"] == "vector-rag":
                        self.add_function(
                            ChatFunction("chat")
                            .add_function(
                                "retrieval_function",
                                VectorRetrievalFunc("retrieval_function")
                                .add_tool("vector_db", self.milvus_db)
                                .add_tool(LLM_TOOL_NAME, self.chat_llm)
                                .config(**chat_config)
                                .done(),
                            )
                            .config(**chat_config)
                            .done(),
                        )
                        self.rag_type = "vector-rag"
            else:
                if req_info and req_info.enable_chat is False:
                    self.remove_function("chat")
                    self.rag_type = None
                    logger.info("Chat/Q&A disabled with the API call")
        except Exception as e:
            logger.error(traceback.format_exc())
            logger.error(f"Error in updating config: {e}")
            raise

    def add_function(self, f: Function):
        assert f.name not in self._functions, str(self._functions)
        self._functions[f.name] = f
        return self

    def get_function(self, fname):
        return self._functions[fname] if fname in self._functions else None

    def remove_function(self, fname: str):
        if fname in self._functions:
            logger.debug(
                f"Removing function {fname} from Context Manager index: {self._process_index}"
            )
            self._functions[fname].areset({"expr": "pk > 0"})
            for f in self._functions[fname]._functions.values():
                f.areset({"expr": "pk > 0"})
            self._functions[fname]._functions.clear()
            del self._functions[fname]

    def update(self, config):
        config_to_print = copy.deepcopy(config)
        if config_to_print.get("api_key"):
            del config_to_print["api_key"]
        logger.info(
            f"Updating context manager with config:\n{json.dumps(config_to_print, indent=2)}"
        )
        try:
            for fn, fn_config in config.items():
                if fn in self._functions:
                    self._functions[fn].update(**fn_config)
        except Exception as e:
            logger.error(
                "Overriding failed for config %s with error: %s", config_to_print, e
            )
            logger.error(traceback.format_exc())
        del config_to_print

    async def aprocess_doc(
        self, doc: str, doc_i: Optional[int] = None, doc_meta: Optional[dict] = None
    ):
        """Process a document.

        Args:
            doc (str): The document content.
            doc_i (Optional[int]): Document index. Required if auto_indexing is False.
            doc_meta (Optional[dict]): Document metadata.

        Returns:
            List of results from all functions processing the document.
        """
        # Handle document indexing
        if self.curr_doc_index < 0:
            if doc_i is None:
                self.auto_indexing = True
                self.curr_doc_index = 0
            else:
                self.auto_indexing = False
                self.curr_doc_index = doc_i

        if self.auto_indexing:
            doc_i = self.curr_doc_index
            self.curr_doc_index += 1
        elif doc_i is None:
            raise ValueError("Param doc_i missing.")

        # Process document through all functions
        tasks = []
        with TimeMeasure("context_manager/aprocess_doc", "yellow"):
            for _, f in self._functions.items():
                tasks.append(
                    asyncio.create_task(
                        f.aprocess_doc_(doc, doc_i, doc_meta), name=f.name
                    )
                )
            return await asyncio.gather(*tasks)

    async def call(self, state):
        """Execute registered functions with the given state.

        Args:
            state: Dictionary containing function names and their parameters
        Returns:
            Dictionary containing results from all function executions
        """
        results = {}
        with TimeMeasure("context_manager/call", "green"):
            tasks = []
            task_results = []
            for func, call_params in state.items():
                tasks.append(
                    asyncio.create_task(self._functions[func](call_params), name=func)
                )
            task_results = await asyncio.gather(*tasks)
            for index, func in enumerate(state):
                results[func] = task_results[index]
        return results

    async def areset(self, state):
        """Reset the context manager and all registered functions.

        Args:
            state: Reset parameters for each function
        Returns:
            Results from resetting all functions
        """
        self.curr_doc_index = -1
        self.auto_indexing = False
        tasks = []
        for func, reset_params in state.items():
            if func in self._functions:
                tasks.append(
                    asyncio.create_task(
                        self._functions[func].areset(reset_params), name=func
                    )
                )
            else:
                logger.debug("Function %s not found. Not resetting.", func)
        return await asyncio.gather(*tasks)
