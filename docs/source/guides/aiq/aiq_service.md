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

# Standalone AIQ Service

## Running Context Aware RAG AIQ plugin as a service

## Exporting environment variables

Export environment variables for our vector and/or graph databases. Also
nvidia api key for LLM models.

### Vector-RAG

``` bash
export MILVUS_DB_HOST=<MILVUS_HOST_IP> #milvus host, e.g. localhost
export MILVUS_DB_PORT=<MILVUS_DB_PORT> #milvus port, e.g. 19530
export NVIDIA_API_KEY=<NVIDIA_API_KEY> #NVIDIA API key
```

### Graph-RAG

``` bash
export GRAPH_DB_URI=<GRAPH_DB_URI> #neo4j uri, e.g. bolt://localhost:7687
export GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME> #neo4j username, e.g. neo4j
export GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD> #neo4j password, e.g. password
export NVIDIA_API_KEY=<NVIDIA_API_KEY> #NVIDIA API key
```

### Foundation-RAG

``` bash
export MILVUS_DB_HOST=<MILVUS_HOST_IP> #milvus host, e.g. localhost
export MILVUS_DB_PORT=<MILVUS_DB_PORT> #milvus port, e.g. 19530
export NVIDIA_API_KEY=<NVIDIA_API_KEY> #NVIDIA API key
```

## Running Data Ingestion

``` bash
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/workflow/config-ingestion-workflow.yml --port <PORT>
```

## Running Graph Retrieval

``` bash
aiq serve --config_file=./src/vss_ctx_rag/aiq_config/workflow/config-retrieval-workflow.yml --port <PORT>
```

### Example Python API calls to the service

Here there are two services running, one for ingestion on port 8000 and
one for retrieval on port 8001.

```python
import requests

# Ingestion request
ingestion_url = "http://localhost:8000/generate"
ingestion_headers = {"Content-Type": "application/json"}
ingestion_data = {
    "text": "The bridge is bright blue."
}

ingestion_response = requests.post(ingestion_url, headers=ingestion_headers, json=ingestion_data)
print("Ingestion Response:", ingestion_response.json())

# Retrieval request
retrieval_url = "http://localhost:8001/generate"
retrieval_headers = {"Content-Type": "application/json"}
retrieval_data = {
    "text": "Is there a bridge? If so describe it"
}

retrieval_response = requests.post(retrieval_url, headers=retrieval_headers, json=retrieval_data)
print("Retrieval Response:", retrieval_response.json())
```
