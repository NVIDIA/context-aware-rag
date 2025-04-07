<!--
SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


# NVIDIA Context Aware RAG

![image](docs/source/_static/data_architecture.png)

Context Aware RAG is a flexible library designed to seamlessly integrate into existing data processing workflows to build customized data ingestion and retrieval (RAG) pipelines.

## Key Features

- [**Data Ingestion Service:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/features.html#ingestion-strategies) Add data to the RAG pipeline from a variety of sources.
- [**Data Retrieval Service:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/features.html#retrieval-strategies) Retrieve data from the RAG pipeline using natural language queries.
- [**Function and Tool Components:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/architecture.html#components) Easy to create custom functions and tools to support your existing workflows.
- [**Swappable Databases:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/architecture.html#) Use a variety of databases to store and retrieve data.
- [**GraphRAG Support:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/architecture.html#) Seamlessly extract knowledge graphs from data to support your existing workflows.
- [**Observability:**](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/features.html#otel-and-metrics) Monitor and troubleshoot your workflows with any OpenTelemetry-compatible monitoring tool.


With Context Aware RAG, you can quickly build RAG pipelines to support your existing workflows.

## Links

 * [Documentation](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/index.html): Explore the full documentation for Context Aware RAG.
 * [Context Aware RAG Architecture](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/overview/architecture.html): Learn more about how Context Aware RAG works and its components.
 * [Getting Started Guide](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/guides/index.html): Set up your environment and start integrating Context Aware RAG into your workflows.
 * [Examples](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/guides/library.html#document-ingestion): Explore examples of Context Aware RAG workflows.
 * [Troubleshooting](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/troubleshooting.html): Get help with common issues.
 * [Release Notes](https://via.gitlab-master-pages.nvidia.com/via-ctx-rag/release-notes.html): Learn about the latest features and improvements.

## Getting Started

### Prerequisites

Before you begin using Context Aware RAG, ensure that you have the following software installed.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)


### Installation

#### Installing from source


##### Create a virtual environment using uv

```bash
uv venv --seed .venv
source .venv/bin/activate
```

##### Clone the repository and install the dependencies

```bash
git clone

uv pip install -e .
```

##### Optionally: Build the wheel file
```bash
uv build
```

######  Install the wheel file
```bash
uv pip install dist/vss-ctx-rag-0.5.0-py3-none-any.whl
```

## Service Example



### Setting up environment variables


Create a .env file in the root directory and set the following variables:

```bash
   NVIDIA_API_KEY=<IF USING NVIDIA>
   NVIDIA_VISIBLE_DEVICES=<GPU ID>

   OPENAI_API_KEY=<IF USING OPENAI>

   VSS_CTX_PORT_RET=<DATA RETRIEVAL PORT>
   VSS_CTX_PORT_IN=<DATA INGESTION PORT>
```

### Using docker compose

```bash
make -C docker start_compose
```

This will start the following services:


* vss-ctx-rag-data-ingestion

  * Service available at `http://<HOST>:<VSS_CTX_PORT_IN>`

* vss-ctx-rag-data-retrieval

  * Service available at `http://<HOST>:<VSS_CTX_PORT_RET>`

* neo4j

  * UI available at `http://<HOST>:7474`

* milvus

  * UI available at `http://<HOST>:9091`

* otel-collector
* jaeger

  * UI available at `http://<HOST>:16686`

* prometheus

  * UI available at `http://<HOST>:9090`

* cassandra

To change the storage volumes, export DOCKER_VOLUME_DIRECTORY to the desired directory.

### Data Ingestion Example

```python
import requests
import json

base_url = "http://<HOST>:<VSS_CTX_PORT_IN>"

headers = {"Content-Type": "application/json"}

### Initialize the service with a unique uuid
init_data = {"uuid": "1"}
### Optional: Initialize the service with a config file or context config
"""
init_data = {"config_path": "/app/config/config.yaml", "uuid": "1"}
init_data = {"context_config": yaml.safe_load(open("/app/config/config.yaml")), "uuid": "1"}
"""
response = requests.post(
    f"{base_url}/init", headers=headers, data=json.dumps(init_data)
)

# POST request to /add_doc to add documents to the service
add_doc_data_list = [
    {"document": "User1: I went hiking to Mission Peak", "doc_index": 4},
    {
        "document": "User1: Hi how are you?",
        "doc_index": 0,
        "doc_metadata": {"is_first": True},
    },
    {"document": "User1: I went hiking to Mission Peak", "doc_index": 4},
    {"document": "User1: I am great too. Thanks for asking", "doc_index": 2},
    {"document": "User2: I am good. How are you?", "doc_index": 1},
    {"document": "User2: So what did you do over the weekend?", "doc_index": 3},
    {
        "document": "User3: Guys there is a fire. Let us get out of here",
        "doc_index": 5,
        "doc_metadata": {"is_last": True},
    },
]

# Send POST requests for each document
for add_doc_data in add_doc_data_list:
    response = requests.post(
        f"{base_url}/add_doc", headers=headers, data=json.dumps(add_doc_data)
    )
```

### Data Retrieval Example

```python
import requests
import json


base_url = "http://<HOST>:<VSS_CTX_PORT_RET>"

headers = {"Content-Type": "application/json"}

### Initialize the service with the same uuid as the data ingestion service
init_data = {"config_path": "/app/service/config.yaml", "uuid": "1"}
response = requests.post(
    f"{base_url}/init", headers=headers, data=json.dumps(init_data)
)

### Send a retrieval request to the service
call_data = {"chat": {"question": "What happens in this situation?"}}

request_data = {"state": call_data}

response = requests.post(
    f"{base_url}/call", headers=headers, data=json.dumps(request_data)
)
print(response.json()["result"])

```

## Acknowledgements

We would like to thank the following projects that made Context Aware RAG possible:

- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Neo4j](https://github.com/neo4j/neo4j)
- [Milvus](https://github.com/milvus-io/milvus)
- [uv](https://github.com/astral-sh/uv)
- [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python)
