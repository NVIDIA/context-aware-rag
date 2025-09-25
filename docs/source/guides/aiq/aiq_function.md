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

# Using vss_ctx_rag AIQ plugin as a function/tool

The vss_ctx_rag AIQ plugin can also be used as a function/tool in custom
AIQ workflows.

In ./src/vss_ctx_rag/aiq_config/function/ there are two
example config files for using vss_ctx_rag as a function/tool for
ingestion and retrieval.

## Retrieval Function

This is an example of the config file for using vss_ctx_rag as a
function/tool for retrieval:

``` yaml
general:
use_uvloop: true

llms:
nim_llm:
    _type: nim
    model_name: nvidia/llama-3.3-nemotron-super-49b-v1
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

    vector_db_host: localhost
    vector_db_port: "19530"

    graph_db_uri: bolt://localhost:7687
    graph_db_user: neo4j
    graph_db_password: passneo4j

    embedding_model_name: embedding_llm

    rerank_model_name: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    rerank_model_url: "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
    rag_type: "vector-rag" # or "graph-rag"
    chat_batch_size: 1
    summ_batch_size: 5
    summ_batch_max_concurrency: 20

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

This is an example of the config file for using vss_ctx_rag as a
function/tool for ingestion:

``` yaml
general:
use_uvloop: true

llms:
nim_llm:
    _type: nim
    model_name: nvidia/llama-3.3-nemotron-super-49b-v1
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

    vector_db_host: localhost
    vector_db_port: "19530"

    graph_db_uri: bolt://localhost:7687
    graph_db_user: neo4j
    graph_db_password: passneo4j

    embedding_model_name: embedding_llm

    rerank_model_name: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    rerank_model_url: "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
    rag_type: "vector-rag" # or "graph-rag"
    chat_batch_size: 1
    summ_batch_size: 5
    summ_batch_max_concurrency: 20

    uuid: "123456"


workflow:
_type: tool_call_workflow
tool_names: [ingestion_function]
llm_name: nim_llm
```

A custom tool call workflow is defined that will use the
vss_ctx_rag_ingestion function to ingest documents into the vector
database. This is so the input passed in will be treated as a document
and not a query.

## Running the function

### Exporting environment variables

Export environment variables for our vector and/or graph databases. Also
nvidia api key for LLM models.

### Vector-RAG

``` bash
export MILVUS_HOST=<MILVUS_HOST_IP>
export MILVUS_PORT=<MILVUS_DB_PORT>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

### Graph-RAG

``` bash
export GRAPH_DB_URI=<GRAPH_DB_URI>
export GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME>
export GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

## Running Data Ingestion

``` bash
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/function/config-ingestion-function.yml --port <PORT>
```

## Running Graph Retrieval

``` bash
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/function/config-retrieval-function.yml --port <PORT>
```

## Example curl request to the services

Here there are two services running, one for ingestion on port 8000 and
one for retrieval on port 8001.

### Ingestion curl request

``` bash
curl --request POST  \
  --url http://localhost:8000/generate   \
 --header 'Content-Type: application/json'   \
 --data '{
     "rag_workflow": "The bridge is bright blue."
 }'
```

### Retrieval curl request

``` bash
curl --request POST \
  --url http://localhost:8001/generate \
  --header 'Content-Type: application/json' \
  --data '{
      "input_message": "Is there a bridge? If so describe it"
  }'
```
