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

# Standalone NAT Service

## Running Context Aware RAG NAT plugin as a service

Export environment variables

[Environment Variables](../../intro/setup.md)

## Running Data Ingestion

```bash
nat serve --config_file=./packages/vss_ctx_rag_nat/src/vss_ctx_rag/plugins/nat/nat_config/workflow/config-ingestion-workflow.yml --port <PORT>
```

## Running Graph Retrieval

```bash
nat serve --config_file=./packages/vss_ctx_rag_nat/src/vss_ctx_rag/plugins/nat/nat_config/workflow/config-retrieval-workflow.yml --port <PORT>
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
    "input_message": "Is there a bridge? If so describe it"
}

retrieval_response = requests.post(retrieval_url, headers=retrieval_headers, json=retrieval_data)
print("Retrieval Response:", retrieval_response.json())
```
