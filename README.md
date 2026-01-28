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

- [**Data Ingestion Service:**](https://nvidia.github.io/context-aware-rag/overview/features.html#ingestion-strategies) Add data to the RAG pipeline from a variety of sources.
- [**Data Retrieval Service:**](https://nvidia.github.io/context-aware-rag/overview/features.html#retrieval-strategies) Retrieve data from the RAG pipeline using natural language queries.
- [**Function and Tool Components:**](https://nvidia.github.io/context-aware-rag/overview/architecture.html#components) Easy to create custom functions and tools to support your existing workflows.
- [**GraphRAG:**](https://nvidia.github.io/context-aware-rag/overview/features.html#retrieval-strategies) Seamlessly extract knowledge graphs from data to support your existing workflows.
- [**Observability:**](https://nvidia.github.io/context-aware-rag/metrics.html) Monitor and troubleshoot your workflows with any OpenTelemetry-compatible monitoring tool.
- [**Experimental Features:**](https://nvidia.github.io/context-aware-rag/overview/experimental.html) CA-RAG also provides structured output mode for response and five important Model Context Protocol (MCP) tools for using CA-RAG with AI agentic workflows.


With Context Aware RAG, you can quickly build RAG pipelines to support your existing workflows.

## Links

 * [Documentation](https://nvidia.github.io/context-aware-rag): Explore the full documentation for Context Aware RAG.
 * [Context Aware RAG Architecture](https://nvidia.github.io/context-aware-rag/overview/architecture.html): Learn more about how Context Aware RAG works and its components.
 * [Getting Started Guide](https://nvidia.github.io/context-aware-rag/intro/setup.html): Set up your environment and start integrating Context Aware RAG into your workflows.
 * [Examples](https://nvidia.github.io/context-aware-rag/examples/pdf_qna.html): Explore examples of Context Aware RAG workflows.
 * [Troubleshooting](https://nvidia.github.io/context-aware-rag/troubleshooting.html): Get help with common issues.
 * [Release Notes](https://nvidia.github.io/context-aware-rag/release-notes.html): Learn about the latest features and improvements.

## Getting Started

### Prerequisites

Before you begin using Context Aware RAG, ensure that you have the following software installed.

- Install [Git](https://git-scm.com/)
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/)


### Installation

#### Clone the repository

```bash
git clone git@github.com:NVIDIA/context-aware-rag.git
cd context-aware-rag/
```

#### Create a virtual environment using uv

```bash
uv venv --seed .venv
source .venv/bin/activate
```

#### Installing from source

```bash
uv pip install -e .
```

#### Installing optional plugins

##### Arango
```bash
uv pip install -e .[arango]
```

##### NAT
```bash
uv pip install -e .[nat]
```

#### Optional: Building and Installing the wheel file

```bash
uv build
uv pip install dist/vss_ctx_rag-1.0.2-py3-none-any.whl
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

   GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME>
   GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD>
   ARANGO_DB_USERNAME=root
   ARANGO_DB_PASSWORD=<ARANGO_DB_PASSWORD>
   MINIO_USERNAME=<MINIO_USERNAME>
   MINIO_PASSWORD=<MINIO_PASSWORD>

```

### Build docker

```bash
make -C docker build
```

### Start using docker compose

```bash
make -C docker start_compose
```

This will start the following services:


* ctx-rag-data-ingestion

  * Service available at `http://<HOST>:<VSS_CTX_PORT_IN>`

* ctx-rag-data-retrieval

  * Service available at `http://<HOST>:<VSS_CTX_PORT_RET>`

* neo4j

  * UI available at `http://<HOST>:7474`

* milvus

* otel-collector

* Phoenix

  * UI available at `http://<HOST>:16686`

* prometheus

  * UI available at `http://<HOST>:9090`


To change the storage volumes, export `DOCKER_VOLUME_DIRECTORY` to the desired directory.

### Stop using docker compose

```bash
make -C docker stop_compose
```

### Data Ingestion Example

```python
import requests
import json
from pyaml_env import parse_config

base_url = "http://<HOST>:<VSS_CTX_PORT_IN>"

headers = {"Content-Type": "application/json"}

### Initialize the service with a unique uuid
init_data = {"uuid": "1"}
### Optional: Initialize the service with a config file or context config
"""
init_data = {"config_path": "/app/config/config.yaml", "uuid": "1"}
init_data = {"context_config": parse_config("/app/config/config.yaml"), "uuid": "1"}
"""
response = requests.post(
    f"{base_url}/init", headers=headers, data=json.dumps(init_data)
)

# POST request to /add_doc to add documents to the service
add_doc_data_list = [
    {
        "document": "User1: Hi how are you?",
        "doc_index": 0,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 0,
            "file": "chat_conversation.txt",
            "is_first": True,
            "is_last": False,
            "uuid": "1"
        },
        "uuid": "1"
    },
    {
        "document": "User2: I am good. How are you?",
        "doc_index": 1,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 1,
            "file": "chat_conversation.txt",
            "uuid": "1"
        },
        "uuid": "1"
    },
    {
        "document": "User1: I am great too. Thanks for asking",
        "doc_index": 2,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 2,
            "file": "chat_conversation.txt",
            "uuid": "1"
        },
        "uuid": "1"
    },
    {
        "document": "User2: So what did you do over the weekend?",
        "doc_index": 3,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 3,
            "file": "chat_conversation.txt",
            "uuid": "1"
        },
        "uuid": "1"
    },
    {
        "document": "User1: I went hiking to Mission Peak",
        "doc_index": 4,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 4,
            "file": "chat_conversation.txt",
            "uuid": "1"
        },
        "uuid": "1"
    },
    {
        "document": "User3: Guys there is a fire. Let us get out of here",
        "doc_index": 5,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 5,
            "file": "chat_conversation.txt",
            "is_first": False,
            "is_last": True,
            "uuid": "1"
        },
        "uuid": "1"
    },
]

# Send POST requests for each document
for add_doc_data in add_doc_data_list:
    response = requests.post(
        f"{base_url}/add_doc", headers=headers, data=json.dumps(add_doc_data)
    )
    print(response.text)

response = requests.post(
    f"{base_url}/complete_ingestion", headers=headers, data=json.dumps({"uuid": "1"})
)
print(response.text)
```

### Data Retrieval Example

```python
import requests
import json


base_url = "http://<HOST>:<VSS_CTX_PORT_RET>"

headers = {"Content-Type": "application/json"}

init_data = {"config_path": "/app/config/config.yaml", "uuid": "1"}
response = requests.post(
    f"{base_url}/init", headers=headers, data=json.dumps(init_data)
)

chat_data = {
    "model": "meta/llama-3.1-70b-instruct",
    "messages": [{"role": "user", "content": "Who mentioned the fire?"}],
    "uuid": "1"
}

response = requests.post(f"{base_url}/chat/completions", headers=headers, data=json.dumps(chat_data))
print(response.json()["choices"][0]["message"]["content"])
```


### Summary Data Retrieval Example

Summary data retrieval can be made to the system using the `/summary` endpoint of the Retrieval Service.

#### Example Query

```python
import requests

url = "http://<HOST>:<VSS_CTX_PORT_RET>/summary"
headers = {"Content-Type": "application/json"}
data = {
    "uuid": "1",
    "summarization": {
        "start_index": 0,
        "end_index": -1
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json()["result"])
```

## Acknowledgements

We would like to thank the following projects that made Context Aware RAG possible:

- [FastAPI](https://github.com/tiangolo/fastapi)
- [LangChain](https://github.com/langchain-ai/langchain)
- [Neo4j](https://github.com/neo4j/neo4j)
- [ArangoDB](https://github.com/arangodb/arangodb)
- [Elasticsearch](https://github.com/elastic/elasticsearch)
- [Milvus](https://github.com/milvus-io/milvus)
- [Qdrant](https://github.com/qdrant/qdrant)
- [uv](https://github.com/astral-sh/uv)
- [OpenTelemetry](https://github.com/open-telemetry/opentelemetry-python)
