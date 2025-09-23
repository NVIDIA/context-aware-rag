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

# NAT Function/Tool

The Context Aware RAG NAT plugin can also be used as a function/tool in custom
NAT workflows.

In ./src/vss_ctx_rag/nat/nat_config/function/ there are two
example config files for using Context Aware RAG as a function/tool for
ingestion and retrieval.

## Retrieval Function

This is an example of the config file for using Context Aware RAG as a
function/tool for retrieval:

```yaml
# config-retrieval-function.yml
general:
  use_uvloop: true

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    max_tokens: 2048
    base_url: "https://integrate.api.nvidia.com/v1"


embedders:
  embedding_llm:
    _type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    truncate: "END"
    base_url: "https://integrate.api.nvidia.com/v1"


functions:
  retrieval_function:
    _type: vss_ctx_rag_retrieval
    llm_name: nim_llm
    retrieval_type: "graph_retrieval"

    db_type: "neo4j"
    db_host: "localhost"
    db_port: "7687"
    db_user: "neo4j"
    db_password: "passneo4j"

    embedding_model_name: embedding_llm

    uuid: "123456"


workflow:
  _type: react_agent
  tool_names: [retrieval_function]
  llm_name: nim_llm
  verbose: true
  retry_parsing_errors: true
  max_retries: 3
```

Here vss_ctx_rag_retrieval function is added as a tool to Langchain
react agent. The react agent is a agent that uses a language model to
decide which tool to use based on the user\'s query. In this example,
the react agent will use the vss_ctx_rag_retrieval function to retrieve
information from the vector database.

## Ingestion Function

This is an example of the config file for using Context Aware RAG as a
function/tool for ingestion:

```yaml
# config-ingestion-function.yml
general:
  use_uvloop: true

llms:
  nim_llm:
    _type: nim
    model_name: meta/llama-3.1-70b-instruct
    max_tokens: 2048
    base_url: "https://integrate.api.nvidia.com/v1"


embedders:
  embedding_llm:
    _type: nim
    model_name: nvidia/llama-3.2-nv-embedqa-1b-v2
    truncate: "END"
    base_url: "https://integrate.api.nvidia.com/v1"


functions:
  ingestion_function:
    _type: vss_ctx_rag_ingestion
    llm_name: nim_llm
    ingestion_type: "graph_ingestion"

    db_type: "neo4j"
    db_host: "localhost"
    db_port: "7687"
    db_user: "neo4j"
    db_password: "passneo4j"

    embedding_model_name: embedding_llm

    uuid: "123456"


workflow:
  _type: tool_call_workflow
  tool_names: [ingestion_function]
  llm_name: nim_llm
```

A custom tool call workflow is defined that will use the
Context Aware RAG ingestion function to ingest documents into the vector
database. This is so the input passed in will be treated as a document
and not a query.

## Running the function

### Exporting environment variables

Ensure you have the correct connection settings for the vector and/or graph databases in the config file.

Refer to the [Setup Guide](../../intro/setup.md) for more details.

## Running Data Ingestion

```bash
nat serve --config_file=./packages/vss_ctx_rag_nat/src/vss_ctx_rag/plugins/nat/nat_config/function/config-ingestion-function.yml --port <PORT>
```

## Running Graph Retrieval

```bash
nat serve --config_file=./packages/vss_ctx_rag_nat/src/vss_ctx_rag/plugins/nat/nat_config/function/config-retrieval-function.yml --port <PORT>
```

## Example Python API calls to the services

Here there are two services running, one for ingestion on port 8000 and
one for retrieval on port 8001.

### Ingestion Python request

```python
import requests

url = "http://localhost:8000/generate"
headers = {"Content-Type": "application/json"}
data = {
    "rag_request": "The bridge is bright blue."
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```

### Retrieval Python request

```python
import requests

url = "http://localhost:8001/generate"
headers = {"Content-Type": "application/json"}
data = {
    "input_message": "Is there a bridge? If so describe it"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
```
