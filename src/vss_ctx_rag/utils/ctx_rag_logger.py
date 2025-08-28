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
import os
import time
import json

import nvtx

# Conditional OpenTelemetry imports - only import if otel is enabled
try:
    import os

    if os.environ.get("VIA_CTX_RAG_ENABLE_OTEL", "false").lower() in [
        "true",
        "1",
        "yes",
        "on",
    ]:
        from opentelemetry import trace
        from opentelemetry.trace import StatusCode
        from openinference.semconv.trace import (
            OpenInferenceSpanKindValues,
            SpanAttributes,
        )

        OTEL_AVAILABLE = True
    else:
        OTEL_AVAILABLE = False
except ImportError:
    OTEL_AVAILABLE = False


LOG_COLORS = {
    "DEFAULT": "\033[0m",
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

VALID_LOG_LEVELS = {
    "DEBUG": LOG_DEBUG_LEVEL,
    "PERF": LOG_PERF_LEVEL,
    "INFO": LOG_INFO_LEVEL,
    "WARNING": LOG_WARNING_LEVEL,
    "ERROR": LOG_ERROR_LEVEL,
    "CRITICAL": LOG_CRITICAL_LEVEL,
}

for level, value in VALID_LOG_LEVELS.items():
    logging.addLevelName(value, level)


class SecureLogFilter(logging.Filter):
    """Filter to mask sensitive environment variable values in log messages."""

    # Environment variable names whose values should never appear in logs
    _SENSITIVE_ENV_VARS = [
        "NVIDIA_API_KEY",
        "OPEN_API_KEY",
        "OPENAI_API_KEY",
        "NGC_CLI_KEY",
        "NGC_CLI_API_KEY",
        "GRAPH_DB_PASSWORD",
        "GRAPH_DB_USERNAME",
        "ARANGO_DB_PASSWORD",
        "ARANGO_DB_USERNAME",
    ]

    def __init__(self):
        super().__init__()

    def filter(self, record):
        current_secrets = [
            os.getenv(var) for var in self._SENSITIVE_ENV_VARS if os.getenv(var)
        ]

        if not current_secrets:
            # No secrets set, nothing to scrub
            return True

        full_msg = record.getMessage()
        masked_msg = full_msg
        for secret in current_secrets:
            if secret and secret in masked_msg:
                masked_msg = masked_msg.replace(secret, "***MASKED***")

        # If we replaced anything, overwrite the record so downstream formatters cannot recover the secret
        if masked_msg != full_msg:
            record.msg = masked_msg
            record.args = ()
        return True


# Configure the logger
logger = logging.getLogger(__name__)

for handler in logger.handlers[:]:
    logger.removeHandler(handler)


class LogFormatter(logging.Formatter):
    def format(self, record):
        color = LOG_COLORS.get(record.levelname, LOG_COLORS["DEFAULT"])
        return f"{self.formatTime(record)} {color}{record.levelname}{LOG_COLORS['DEFAULT']} {record.getMessage()}"


log_file_path = os.environ.get("VSS_LOG_FILE", "/tmp/via-logs/vss_ctx_rag.log")


log_level = VALID_LOG_LEVELS.get(
    os.environ.get("VSS_LOG_LEVEL", "INFO").upper(), LOG_INFO_LEVEL
)


# Create log directory if it doesn't exist
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)

file_logger = logging.FileHandler(log_file_path)
file_logger.setLevel(log_level)
file_logger.setFormatter(LogFormatter("%(asctime)s %(levelname)s %(message)s"))


term_out = logging.StreamHandler()
term_out.setLevel(log_level)
term_out.setFormatter(LogFormatter("%(asctime)s %(levelname)s %(message)s"))

logger.addFilter(SecureLogFilter())

logger.addHandler(term_out)
logger.addHandler(file_logger)

logger.setLevel(log_level)


def make_serializable(obj):
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    elif hasattr(obj, "__dict__"):
        return obj.__dict__
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    else:
        return obj


class Metrics:
    if OTEL_AVAILABLE:
        SPAN_KIND = {
            "CHAIN": OpenInferenceSpanKindValues.CHAIN.value,
            "LLM": OpenInferenceSpanKindValues.LLM.value,
            "TOOL": OpenInferenceSpanKindValues.TOOL.value,
            "RETRIEVER": OpenInferenceSpanKindValues.RETRIEVER.value,
            "EMBEDDING": OpenInferenceSpanKindValues.EMBEDDING.value,
            "AGENT": OpenInferenceSpanKindValues.AGENT.value,
            "RERANKER": OpenInferenceSpanKindValues.RERANKER.value,
            "UNKNOWN": OpenInferenceSpanKindValues.UNKNOWN.value,
            "GUARDRAIL": OpenInferenceSpanKindValues.GUARDRAIL.value,
            "EVALUATOR": OpenInferenceSpanKindValues.EVALUATOR.value,
        }
    else:
        SPAN_KIND = {
            "CHAIN": "CHAIN",
            "LLM": "LLM",
            "TOOL": "TOOL",
            "RETRIEVER": "RETRIEVER",
            "EMBEDDING": "EMBEDDING",
            "AGENT": "AGENT",
            "RERANKER": "RERANKER",
            "UNKNOWN": "UNKNOWN",
            "GUARDRAIL": "GUARDRAIL",
            "EVALUATOR": "EVALUATOR",
        }

    def __init__(
        self, string, nvtx_color="grey", span_kind="UNKNOWN", print=True
    ) -> None:
        self._string = string
        self._print = print
        self._nvtx_color = nvtx_color
        self._nvtx_trace = None
        if OTEL_AVAILABLE:
            self._tracer = trace.get_tracer(__name__)
        else:
            self._tracer = None
        self._span_context_manager = None

        if OTEL_AVAILABLE and span_kind not in self.SPAN_KIND:
            raise ValueError(f"Invalid span kind: {span_kind}")
        self._span_kind = span_kind

    def __enter__(self):
        self._start_time = time.time()
        self._nvtx_trace = nvtx.start_range(
            message=self._string, color=self._nvtx_color
        )

        if OTEL_AVAILABLE and self._tracer:
            self._span = self._tracer.start_span(self._string)
            self._span.set_attribute(
                SpanAttributes.OPENINFERENCE_SPAN_KIND, self._span_kind
            )
            self._span_context_manager = trace.use_span(self._span, end_on_exit=True)
            self._span_context_manager.__enter__()
        else:
            self._span = None

        return self

    def __exit__(self, type, value, traceback):
        self._end_time = time.time()
        nvtx.end_range(self._nvtx_trace)
        self._execution_time = self._end_time - self._start_time

        if OTEL_AVAILABLE and self._span:
            self._span.set_attribute("span name", self._string)
            self._span.set_attribute("execution_time_ms", self._execution_time * 1000.0)

            if type is not None:
                self._span.set_status(StatusCode.ERROR)
                self._span.record_exception(value)
            else:
                self._span.set_status(StatusCode.OK)

            self._span_context_manager.__exit__(type, value, traceback)

        if self._print:
            logger.log(
                LOG_PERF_LEVEL,
                "{:s} time = {:.2f} ms".format(
                    self._string, self._execution_time * 1000.0
                ),
            )

    def input(self, value):
        if OTEL_AVAILABLE and self._span:
            self._span.set_attribute(SpanAttributes.INPUT_MIME_TYPE, "application/json")

            try:
                serializable_value = make_serializable(value)
                json_value = json.dumps(serializable_value, default=str)
            except Exception:
                json_value = json.dumps(str(value))

            self._span.set_attribute(SpanAttributes.INPUT_VALUE, json_value)

    def output(self, value):
        if OTEL_AVAILABLE and self._span:
            self._span.set_attribute(
                SpanAttributes.OUTPUT_MIME_TYPE, "application/json"
            )

            try:
                serializable_value = make_serializable(value)
                json_value = json.dumps(serializable_value, default=str)
            except Exception:
                json_value = json.dumps(str(value))

            self._span.set_attribute(SpanAttributes.OUTPUT_VALUE, json_value)

    def error(self, value):
        if OTEL_AVAILABLE and self._span:
            self._span.set_status(StatusCode.ERROR)
            self._span.record_exception(value)

    @property
    def span(self):
        return self._span

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
