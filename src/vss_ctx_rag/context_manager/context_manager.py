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

"""Context manager process implementation.

This module handles managing the input to LLM by calling the handlers of all
the tools it has access to.
"""

import asyncio
import random
import traceback
import time
from threading import Thread
from typing import Dict, Optional
import os
import multiprocessing
import concurrent.futures

from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.utils.otel import init_otel
from vss_ctx_rag.context_manager.context_manager_handler import (
    ContextManagerHandler,
)
from vss_ctx_rag.utils.utils import RequestInfo
from vss_ctx_rag.utils.utils import validate_config

WAIT_ON_PENDING = 10  # Amount of time to wait before clearing the pending

mp_ctx = multiprocessing.get_context("spawn")


class ContextManagerProcess(mp_ctx.Process):
    def __init__(
        self,
        config: Dict,
        process_index: int,
        req_info: Optional[RequestInfo] = None,
    ) -> None:
        logger.info(f"Initializing Context Manager Process no.: {process_index}")
        super().__init__()
        self._lock = mp_ctx.Lock()
        self._pending_requests_lock = mp_ctx.Lock()
        self._queue = mp_ctx.Queue()
        self._response_queue = mp_ctx.Queue()
        self._stop = mp_ctx.Event()
        self._pending_add_doc_requests = []
        self._request_start_times = {}
        self.config = config
        self.process_index = process_index
        self.req_info = req_info
        self._init_done_event = mp_ctx.Event()

    def wait_for_initialization(self):
        """Wait for the process initialization to complete

        Returns:
            Boolean indicating if process initialized successfully or encountered
            an error.
        """
        while not self._init_done_event.wait(1):
            if not self.is_alive():
                return False
        return True

    def _initialize(self):
        if os.environ.get("VIA_CTX_RAG_ENABLE_OTEL", "false").lower() in [
            "true",
            "1",
            "yes",
            "on",
        ]:
            exporter_type = os.environ.get("VIA_CTX_RAG_EXPORTER", "console")
            endpoint = os.environ.get("VIA_CTX_RAG_OTEL_ENDPOINT", "")
            service_name = (
                f"vss-ctx-rag-{self.req_info.uuid if self.req_info else 'default'}"
            )
            init_otel(
                service_name=service_name,
                exporter_type=exporter_type,
                endpoint=endpoint,
            )
        self.cm_handler = ContextManagerHandler(
            self.config, self.process_index, self.req_info
        )
        self._init_done_event.set()

    def start_bg_loop(self) -> None:
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def start(self):
        super().start()

    def stop(self):
        """Stop the process"""
        self._stop.set()
        self.join()

    def run(self) -> None:
        # Run while not signalled to stop
        try:
            logger.debug(
                f"Run called for Context Manager Process no.: {self.process_index}"
            )
            self.event_loop = asyncio.new_event_loop()
            self.t = Thread(target=self.start_bg_loop, daemon=True)
            self.t.start()
            self._initialize()

            while not self._stop.is_set():
                with self._lock:
                    qsize = self._queue.qsize()

                    if (qsize) == 0:
                        time.sleep(0.01)
                        continue

                    item = self._queue.get()
                    if item and "add_doc" in item:
                        logger.debug(
                            f"Processing document "
                            f"{item['add_doc']['doc_content']['doc_i']}: "
                            f"{item['add_doc']['doc_content']['doc']}"
                        )
                        future = asyncio.run_coroutine_threadsafe(
                            self.cm_handler.aprocess_doc(
                                **item["add_doc"]["doc_content"]
                            ),
                            self.event_loop,
                        )
                        self._add_pending_request(future)
                    elif item and "reset" in item:
                        state = item["reset"]
                        with TimeMeasure("context_manager/reset", "green"):
                            stop_time = time.time() + WAIT_ON_PENDING
                            while True:
                                with self._pending_requests_lock:
                                    pending_count = len(self._pending_add_doc_requests)
                                    if not pending_count or time.time() >= stop_time:
                                        break

                                time.sleep(2)
                                logger.info(
                                    f"Completing pending requests...{pending_count}"
                                )

                            with self._pending_requests_lock:
                                self._pending_add_doc_requests = []
                            future = asyncio.run_coroutine_threadsafe(
                                self.cm_handler.areset(state), loop=self.event_loop
                            )
                            future.result()
                    elif item and "call" in item:
                        with TimeMeasure("context_manager/call-manager", "blue"):
                            # TODO: Wait for add docs to finish
                            with TimeMeasure(
                                "context_manager/call/pending_add_doc", "blue"
                            ):
                                with self._pending_requests_lock:
                                    pending_requests_copy = (
                                        self._pending_add_doc_requests.copy()
                                    )
                                done, not_done = concurrent.futures.wait(
                                    pending_requests_copy
                                )
                                # Check each completed future for exceptions
                                for future in done:
                                    try:
                                        future.result()  # This will raise the exception if one occurred
                                    except Exception as e:
                                        logger.error(
                                            f"Some add_doc failed to complete: {e}"
                                        )
                            try:
                                state = item["call"]
                                future = asyncio.run_coroutine_threadsafe(
                                    self.cm_handler.call(state), self.event_loop
                                )
                                self._response_queue.put(future.result())
                            except Exception as e:
                                traceback.print_exc()
                                logger.error(f"Error calling context manager: {e}")
                                self._response_queue.put({"error": f"{e}"})
                    elif item and "update" in item:
                        self.cm_handler.update(item["update"])
                    elif item and "configure_update" in item:
                        try:
                            self.cm_handler.configure_update(
                                item["configure_update"]["config"],
                                item["configure_update"]["req_info"],
                            )
                        except Exception as e:
                            logger.error(f"Error in updating config: {e}")
                            self._response_queue.put({"error": f"{e}"})

        except Exception as e:
            logger.error("Exception %s", str(e))
            logger.error(traceback.format_exc())

    def _add_pending_request(self, future):
        with self._pending_requests_lock:
            self._pending_add_doc_requests.append(future)
            self._request_start_times[id(future)] = time.time()
            future.add_done_callback(self._remove_pending_request)

    def _remove_pending_request(self, future):
        with self._pending_requests_lock:
            try:
                self._pending_add_doc_requests.remove(future)
                # Calculate and log processing time
                future_id = id(future)
                if future_id in self._request_start_times:
                    start_time = self._request_start_times.pop(future_id)
                    duration = time.time() - start_time
                    logger.debug(f"Document processing completed in {duration:.3f}s")
            except ValueError:
                logger.error(
                    f"Attempted to remove future that was already removed: {future}"
                )
                pass  # Already removed

    def add_doc(
        self,
        doc_content: str,
        doc_i: Optional[int] = None,
        doc_meta: Optional[dict] = None,
        callback=None,
    ):
        """
        Thread-safe method to add a document to the context manager.

        Args:
            doc_content (str): The document content to add.
            doc_i (Optional[int]): Document index.
            doc_meta (Optional[dict]): Optional metadata associated with the document.
            callback (Optional[Callable]): Optional callback function
                                            to be called after document is processed.

        """
        with TimeMeasure("context_manager/add_doc", "pink"):
            self._queue.put(
                {
                    "add_doc": {
                        "doc_content": {
                            "doc": doc_content,
                            "doc_i": doc_i,
                            "doc_meta": doc_meta,
                        }
                    }
                }
            )

    def update(self, config):
        self._queue.put({"update": config})

    def configure_update(self, config, req_info):
        self._queue.put({"configure_update": {"config": config, "req_info": req_info}})

    def call(self, state):
        self._queue.put({"call": state})
        return self._response_queue.get()

    def reset(self, state):
        self._queue.put({"reset": state})


class ReqInfo:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class ContextManager:
    def __init__(
        self,
        config: Dict,
        process_index: Optional[int] = random.randint(0, 1000000),
        req_info: Optional[RequestInfo] = None,
    ) -> None:
        try:
            validate_config(config)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
        self._process_index = process_index
        logger.debug(f"Initializing Context Manager index: {self._process_index}")
        try:
            self.process = ContextManagerProcess(config, self._process_index, req_info)
            self.process.start()
            if (
                os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true"
                and not self.process.wait_for_initialization()
            ):
                self.process.stop()
                raise Exception(
                    f"Failed to load Context Manager Process no.: {self._process_index}"
                )
        except Exception as e:
            logger.error(f"Error initializing Context Manager: {e}")
            raise

    def __del__(self):
        logger.debug(f"Stopping Context Manager Process: {self._process_index}")
        self.process.stop()

    def add_doc(
        self,
        doc_content: str,
        doc_i: Optional[int] = None,
        doc_meta: Optional[dict] = None,
        callback=None,
    ):
        """
        Thread-safe method to add a document.

        Args:
            doc_content (str): The document content to add.
            doc_i (Optional[int]): Document index.
            doc_meta (Optional[dict]): Optional metadata associated with the document.
        """
        self.process.add_doc(doc_content, doc_i, doc_meta, callback)

    def update(self, config):
        try:
            validate_config(config)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
        self.process.update(config=config)

    def configure_update(self, config: Dict, req_info):
        try:
            validate_config(config)
        except Exception as e:
            logger.error(f"Config validation failed: {e}")
            raise
        req_info_obj = None
        if req_info:
            req_info_obj = ReqInfo(
                **{
                    "summarize": req_info.summarize,
                    "enable_chat": req_info.enable_chat,
                    "is_live": req_info.is_live,
                    "uuid": req_info.stream_id,
                    "caption_summarization_prompt": req_info.caption_summarization_prompt,
                    "summary_aggregation_prompt": req_info.summary_aggregation_prompt,
                    "chunk_size": req_info.chunk_size,
                    "summary_duration": req_info.summary_duration,
                    "rag_type": req_info.rag_type,
                }
            )
        self.process.configure_update(config=config, req_info=req_info_obj)

    def call(self, state):
        return self.process.call(state)

    def reset(self, state):
        logger.debug(f"Resetting Context Manager index: {self._process_index}")
        return self.process.reset(state)
