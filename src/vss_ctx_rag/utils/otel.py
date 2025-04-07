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


from opentelemetry import trace

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger


def init_otel(service_name: str, exporter_type: str, endpoint=None):
    """
    Initializes otel trace and metrics provider. Also creates either a
    ConsoleSpanExporter or OTLPSpanExporter based on whether the user
    wants to export to console or an OTLP endpoint
    """

    resource = Resource(attributes={"service.name": service_name})

    traceProvider = TracerProvider(resource=resource)
    if exporter_type == "console":
        processor = BatchSpanProcessor(
            ConsoleSpanExporter(formatter=lambda span: span.to_json(indent=None))
        )
    elif exporter_type == "otlp" and endpoint:
        TRACES_ENDPOINT = endpoint + "/v1/traces"
        processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=TRACES_ENDPOINT))
    else:
        raise ValueError(
            f"Invalid exporter type: {exporter_type}. Valid types are: console, otlp. Check if endpoint is provided for otlp exporter."
        )
    traceProvider.add_span_processor(processor)
    trace.set_tracer_provider(traceProvider)

    logger.info(
        f"OTEL Initialized Successfully. Exporter Type: {exporter_type}, Endpoint: {endpoint}"
    )
