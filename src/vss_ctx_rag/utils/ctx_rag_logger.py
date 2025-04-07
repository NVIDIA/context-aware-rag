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

import logging
import time
import nvtx
import os
from opentelemetry import trace

LOG_COLORS = {
    "RESET": "\033[0m",
    "CRITICAL": "\033[1m",
    "ERROR": "\033[91m",
    "WARNING": "\033[93m",
    "INFO": "\033[94m",
    "PERF": "\033[95m",
    "DEBUG": "\033[96m",
}

LOG_DEBUG_LEVEL = 10
LOG_PERF_LEVEL = 15
LOG_INFO_LEVEL = 20
LOG_WARNING_LEVEL = 30
LOG_ERROR_LEVEL = 40
LOG_CRITICAL_LEVEL = 50

logging.addLevelName(LOG_CRITICAL_LEVEL, "CRITICAL")
logging.addLevelName(LOG_ERROR_LEVEL, "ERROR")
logging.addLevelName(LOG_WARNING_LEVEL, "WARNING")
logging.addLevelName(LOG_INFO_LEVEL, "INFO")
logging.addLevelName(LOG_PERF_LEVEL, "PERF")
logging.addLevelName(LOG_DEBUG_LEVEL, "DEBUG")

# Configure the logger
logger = logging.getLogger(__name__)
log_level = os.environ.get("VSS_LOG_LEVEL")
if log_level:
    logger.setLevel(log_level.upper())
else:
    logger.setLevel("INFO")

for handler in logger.handlers[:]:
    logger.removeHandler(handler)


class LogFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["RESET"])
        return f"{self.formatTime(record)} {color}{record.levelname}{LOG_COLORS['RESET']} {record.getMessage()}"


file_logger = logging.FileHandler("/tmp/via-logs/vss_ctx_rag.log")
file_logger.setLevel(LOG_PERF_LEVEL)
file_logger.setFormatter(LogFormatter("%(asctime)s %(levelname)s %(message)s"))


term_out = logging.StreamHandler()
term_out.setLevel(LOG_DEBUG_LEVEL)
term_out.setFormatter(LogFormatter("%(asctime)s %(levelname)s %(message)s"))

logger.addHandler(term_out)
logger.addHandler(file_logger)


class TimeMeasure:
    def __init__(self, string, nvtx_color="grey", print=True) -> None:
        self._string = string
        self._print = print
        self._nvtx_color = nvtx_color
        self._nvtx_trace = None
        self._tracer = trace.get_tracer(__name__)

    def __enter__(self):
        self._start_time = time.time()
        self._nvtx_trace = nvtx.start_range(
            message=self._string, color=self._nvtx_color
        )
        self._span = self._tracer.start_span(self._string)
        return self

    def __exit__(self, type, value, traceback):
        self._end_time = time.time()
        nvtx.end_range(self._nvtx_trace)
        self._execution_time = self._end_time - self._start_time

        self._span.set_attribute("span name", self._string)
        self._span.set_attribute("execution_time_ms", self._execution_time * 1000.0)
        self._span.end()

        if self._print:
            logger.log(
                LOG_PERF_LEVEL,
                "{:s} time = {:.2f} ms".format(
                    self._string, self._execution_time * 1000.0
                ),
            )

    @property
    def execution_time(self):
        return self._execution_time

    @property
    def current_execution_time(self):
        return time.time() - self._start_time

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time
