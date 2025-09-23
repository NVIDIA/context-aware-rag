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
import concurrent.futures
import multiprocessing
import os
import random
import time
import traceback
from threading import Thread
from typing import Dict, Optional

from vss_ctx_rag.context_manager.context_manager_handler import ContextManagerHandler
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import DEFAULT_CONTEXT_MANAGER_CALL_TIMEOUT
from vss_ctx_rag.utils.otel import init_otel
import datetime


WAIT_ON_PENDING = 10  # Amount of time to wait before clearing the pending

mp_ctx = multiprocessing.get_context("spawn")


class ContextManagerProcess(mp_ctx.Process):
    def __init__(
        self,
        config: Dict,
        process_index: int,
    ) -> None:
        logger.info(f"Initializing Context Manager Process no.: {process_index}")
        super().__init__()
        self._lock = mp_ctx.Lock()
        self._pending_requests_lock = mp_ctx.Lock()
        self._queue = mp_ctx.Queue()
        self._input_queue = mp_ctx.Queue()
        self._response_queue = mp_ctx.Queue()
        self._configure_queue = mp_ctx.Queue()
        self._reset_queue = mp_ctx.Queue()
        self._stop = mp_ctx.Event()
        self._pending_add_doc_requests = []
        self._request_start_times = {}
        self.config = config
        self.process_index = process_index
        self._init_done_event = mp_ctx.Event()
        timeout_env = os.getenv("CONTEXT_MANAGER_CALL_TIMEOUT")
        try:
            self.timeout = (
                int(timeout_env)
                if timeout_env is not None
                else DEFAULT_CONTEXT_MANAGER_CALL_TIMEOUT
            )
        except ValueError:
            logger.warning(
                f"Invalid CONTEXT_MANAGER_CALL_TIMEOUT value: {timeout_env}, using default 3600s (1 hour)"
            )
            self.timeout = DEFAULT_CONTEXT_MANAGER_CALL_TIMEOUT

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
            service_name = f"vss-ctx-rag-{self.config.get('context_manager', {}).get('uuid', 'default')}"
            init_otel(
                service_name=service_name,
                exporter_type=exporter_type,
                endpoint=endpoint,
            )
        self.cm_handler = ContextManagerHandler(self.config, self.process_index)
        self._init_done_event.set()

    def start_bg_loop(self) -> None:
        asyncio.set_event_loop(self.event_loop)
        self.event_loop.run_forever()

    def start(self):
        super().start()

    def stop(self):
        """Stop the process"""
        logger.info(f"Stopping Context Manager Process no.: {self.process_index}")
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
            lastTime = datetime.datetime.now()

            while not self._stop.is_set():
                with self._lock:
                    qsize = self._queue.qsize()

                    if qsize == 0:
                        period = datetime.datetime.now()
                        if (period - lastTime).total_seconds() >= 20:
                            logger.debug(
                                f"Process Index: {self.process_index} - No items in queue, time: {period.strftime('%H:%M:%S')}"
                            )
                            lastTime = period
                        time.sleep(0.01)
                        continue

                    logger.debug(
                        f"Process Index: {self.process_index} - Queue size: {qsize}"
                    )
                    try:
                        item = self._queue.get(timeout=self.timeout)
                    except Exception as e:
                        logger.error(
                            f"Process Index: {self.process_index} - Error getting item from queue: {e}"
                        )
                        continue
                    logger.debug(
                        f"Process Index: {self.process_index} - Processing item: {item}"
                    )
                    if item and "add_doc" in item:
                        logger.debug(
                            f"Process Index: {self.process_index} - Processing document "
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
                        logger.debug(
                            f"Process Index: {self.process_index} - Processing reset request: {item['reset']}"
                        )
                        try:
                            state = item["reset"]
                            with Metrics(
                                "context_manager/reset",
                                "green",
                                span_kind=Metrics.SPAN_KIND["CHAIN"],
                            ) as tm:
                                tm.input(state)
                                stop_time = time.time() + WAIT_ON_PENDING
                                while True:
                                    with self._pending_requests_lock:
                                        pending_count = len(
                                            self._pending_add_doc_requests
                                        )
                                        if (
                                            not pending_count
                                            or time.time() >= stop_time
                                        ):
                                            break
                                    time.sleep(2)
                                    logger.debug(
                                        f"Process Index: {self.process_index} - Completing pending requests...{pending_count}"
                                    )

                                with self._pending_requests_lock:
                                    self._pending_add_doc_requests = []
                                future = asyncio.run_coroutine_threadsafe(
                                    self.cm_handler.areset(state), loop=self.event_loop
                                )
                                future.result(timeout=self.timeout)
                                self._reset_queue.put({"success": "true"})
                        except Exception as e:
                            logger.error(
                                f"Process Index: {self.process_index} - Error resetting context manager process: {e}"
                            )
                            self._reset_queue.put({"error": str(e)})
                    elif item and "call" in item:
                        logger.debug(
                            f"Process Index: {self.process_index} - Processing call request: {item['call']}"
                        )
                        with Metrics(
                            "context_manager/call-manager",
                            "blue",
                            span_kind=Metrics.SPAN_KIND["CHAIN"],
                        ) as tm:
                            # TODO: Wait for add docs to finish
                            with Metrics(
                                "context_manager/call/pending_add_doc", "blue"
                            ) as tm:
                                with self._pending_requests_lock:
                                    pending_requests_copy = (
                                        self._pending_add_doc_requests.copy()
                                    )
                                logger.debug(
                                    f"Process Index: {self.process_index} - Waiting for {len(pending_requests_copy)} add_doc requests to complete, {item['call']}"
                                )
                                done, _ = concurrent.futures.wait(pending_requests_copy)
                                logger.debug(
                                    f"Process Index: {self.process_index} - Pending Add Doc requests Done: {len(done)} requests"
                                )
                                # Check each completed future for exceptions
                                for future in done:
                                    try:
                                        logger.debug(
                                            f"Process Index: {self.process_index} - Processing add_doc request..."
                                        )
                                        future.result(
                                            timeout=self.timeout
                                        )  # This will raise the exception if one occurred
                                    except Exception as e:
                                        logger.error(
                                            f"Process Index: {self.process_index} - Some add_doc failed to complete: {e}"
                                        )

                            try:
                                tm.input(item["call"])
                                state = item["call"]
                                logger.debug(
                                    f"Process Index: {self.process_index} - context manager call"
                                )
                                future = asyncio.run_coroutine_threadsafe(
                                    self.cm_handler.call(state), self.event_loop
                                )
                                try:
                                    result = future.result(timeout=self.timeout)
                                    self._response_queue.put(result)
                                    logger.debug(
                                        f"Process Index: {self.process_index} - Context manager call result: {result}"
                                    )
                                except concurrent.futures.TimeoutError as e:
                                    logger.error(
                                        f"Process Index: {self.process_index} - Timeout waiting for context manager call result {self.timeout}s"
                                    )
                                    self._response_queue.put({"error": str(e)})
                            except Exception as e:
                                tm.error(e)
                                logger.error(
                                    f"Process Index: {self.process_index} - Error calling context manager: {e}"
                                )
                                self._response_queue.put({"error": str(e)})
                    elif item and ("configure" in item):
                        with Metrics(
                            "context_manager/configure",
                            "purple",
                            span_kind=Metrics.SPAN_KIND["UNKNOWN"],
                        ) as tm:
                            if "configure" in item:
                                tm.input(item["configure"])
                                logger.debug(
                                    f"Process Index: {self.process_index} - context manager configure {item['configure']['config']}"
                                )
                                try:
                                    future = asyncio.run_coroutine_threadsafe(
                                        self.cm_handler.aconfigure(
                                            item["configure"]["config"]
                                        ),
                                        loop=self.event_loop,
                                    )
                                    future.result(timeout=self.timeout)
                                    # Update stored config to ensure reinitializations use the latest config
                                    self.config = item["configure"]["config"]
                                    self._configure_queue.put({"success": "true"})
                                except Exception as e:
                                    tm.error(e)
                                    logger.error(
                                        f"Process Index: {self.process_index} - Error in configuring context manager: {e}"
                                    )
                                    self._configure_queue.put({"error": str(e)})

        except Exception as e:
            logger.error(f"Process Index: {self.process_index} - Exception in run: {e}")
            logger.error(traceback.format_exc())
            raise e

    def _add_pending_request(self, future):
        with self._pending_requests_lock:
            self._pending_add_doc_requests.append(future)
            self._request_start_times[id(future)] = time.time()
        # Register callback outside the lock to avoid deadlock if future is already done
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
        try:
            logger.debug(
                f"Process Index: {self.process_index} - context manager add doc"
            )
            with Metrics("context_manager/add_doc", "pink"):
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
            return {"success": "true"}
        except Exception as e:
            logger.error(
                f"Process Index: {self.process_index} - Error adding document: {e}"
            )
            return {"error": f"Error adding document: {e}"}

    def configure(self, config):
        logger.debug(
            f"Process Index: {self.process_index} - context manager configure {config}"
        )
        try:
            self._queue.put({"configure": {"config": config}})
            logger.debug(
                f"Process Index: {self.process_index} - context manager configure queue size: {self._queue.qsize()}"
            )
            output = self._configure_queue.get(timeout=self.timeout)
            logger.debug(
                f"Process Index: {self.process_index} - context manager configure output: {output}"
            )
            if isinstance(output, dict) and "error" in output:
                raise Exception(output["error"])
            return output
        except Exception as e:
            logger.error(
                f"Process Index: {self.process_index} - Error configuring context manager process: {e}"
            )
            return {"error": f"Error configuring context manager process: {e}"}

    def call(self, state):
        try:
            logger.debug(
                f"Process Index: {self.process_index} - context manager call {state}"
            )
            self._queue.put({"call": state})
            logger.debug(
                f"Process Index: {self.process_index} - context manager call queue size: {self._queue.qsize()}"
            )
            output = self._response_queue.get(timeout=self.timeout)
            logger.debug(
                f"Process Index: {self.process_index} - context manager call output: {output}"
            )
            if isinstance(output, dict) and "error" in output:
                raise Exception(output["error"])
            return output
        except Exception as e:
            logger.error(
                f"Process Index: {self.process_index} - Error calling context manager process: {e}"
            )
            logger.error(traceback.format_exc())
            return {"error": f"Error calling context manager process: {e}"}

    def reset(self, state):
        try:
            logger.debug(
                f"Process Index: {self.process_index} - context manager reset {state}"
            )
            self._queue.put({"reset": state})
            logger.debug(
                f"Process Index: {self.process_index} - context manager reset queue size: {self._queue.qsize()}"
            )
            output = self._reset_queue.get(timeout=self.timeout)
            logger.debug(
                f"Process Index: {self.process_index} - context manager reset output: {output}"
            )
            if isinstance(output, dict) and "error" in output:
                raise Exception(output["error"])
            return output
        except Exception as e:
            logger.error(
                f"Process Index: {self.process_index} - Error resetting context manager process: {e}"
            )
            logger.error(traceback.format_exc())
            return {"error": f"Error resetting context manager process: {e}"}


class ContextManager:
    def __init__(
        self,
        config: Dict,
        process_index: Optional[int] = random.randint(0, 1000000),
    ) -> None:
        self._process_index = process_index
        logger.debug(f"Initializing Context Manager index: {self._process_index}")
        try:
            self.process = ContextManagerProcess(config, self._process_index)
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
        logger.debug(
            f"Process Index: {self._process_index} - context manager - add doc"
        )
        output = self.process.add_doc(doc_content, doc_i, doc_meta, callback)
        logger.debug(
            f"Process Index: {self._process_index} - context manager - add doc output: {output}"
        )
        return output

    def configure(self, config):
        logger.debug(
            f"Process Index: {self._process_index} - context manager - configure"
        )
        output = self.process.configure(config=config)
        logger.debug(
            f"Process Index: {self._process_index} - context manager - configure output: {output}"
        )
        return output

    def call(self, state):
        logger.debug(f"Process Index: {self._process_index} - context manager - call")
        output = self.process.call(state)
        logger.debug(
            f"Process Index: {self._process_index} - context manager - call output: {output}"
        )
        return output

    def reset(self, state):
        logger.debug(
            f"Process Index: {self._process_index} - context manager reset {state}"
        )
        output = self.process.reset(state)
        logger.debug(
            f"Process Index: {self._process_index} - context manager - reset output: {output}"
        )
        return output
