<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
 *
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
 *
http://www.apache.org/licenses/LICENSE-2.0
 *
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->


# Context Aware RAG

![image](./_static/data_architecture.png)

Context Aware RAG is a flexible library designed to seamlessly integrate into existing data processing workflows to build customized RAG pipelines.

## Key Features

- [**Data Ingestion Service:**](./overview/features.md#ingestion-strategies) Add data to the RAG pipeline from a variety of sources.
- [**Data Retrieval Service:**](./overview/features.md#retrieval-strategies) Retrieve data from the RAG pipeline using natural language queries.
- [**Function and Tool Components:**](./overview/architecture.md#components) Easy to create custom functions and tools to support your existing workflows.
- [**GraphRAG:**](./overview/features.md#retrieval-strategies) Seamlessly extract knowledge graphs from data to support your existing workflows.
- [**Observability:**](./metrics.md) Monitor and troubleshoot your workflows with any OpenTelemetry-compatible monitoring tool.
- [**Experimental Features:**](./overview/experimental.md)
    - **Model Context Protocol Tools**: Enable interaction with CA-RAG library through MCP tools when hosted as a service.
    - **Structured Response**: Enable CA-RAG to return retrieval responses as JSON with specified structure when used as a library or service.

With Context Aware RAG, you can quickly build RAG pipelines to support your existing workflows.

```{toctree}
:hidden:
:maxdepth: 2

Quick Start <./intro/index.md>
Overview <./overview/index.md>
Docker Deployment <./guides/docker/index.md>
Usage <./guides/usage/index.md>
VSS Blueprint Integration <./vss.md>
NAT Plugin Guide <./guides/nat/index.md>
Components <./guides/components/index.md>
Examples <./examples/index.md>
Metrics <./metrics.md>
Experimental Features <./overview/experimental.md>
Troubleshooting <./troubleshooting.md>
Release Notes <./release-notes.md>
Code of Conduct <./code-of-conduct.md>
Contributing <./contributing.md>
```
