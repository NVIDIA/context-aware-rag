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

# Running Context Aware RAG AIQ plugin as a service

## Exporting environment variables

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
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/workflow/config-ingestion-workflow.yml --port <PORT>
```

## Running Graph Retrieval

``` bash
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/workflow/config-retrieval-workflow.yml --port <PORT>
```

### Example curl request to the service

Here there are two services running, one for ingestion on port 8000 and
one for retrieval on port 8001.

``` bash
curl --request POST  \
  --url http://localhost:8000/generate   \
 --header 'Content-Type: application/json'   \
 --data '{
     "text": "The bridge is bright blue."
 }'


 curl --request POST  \
  --url http://localhost:8001/generate   \
 --header 'Content-Type: application/json'   \
 --data '{
     "text": "Is there a bridge? If so describe it"
 }'
```
