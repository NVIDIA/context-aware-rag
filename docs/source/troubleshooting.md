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

# Troubleshooting

This section contains common issues and how to troubleshoot them.

## Common issues

If you ingest documents and then run retrieval, but see the following
result

```bash
Response: "I don't know"
```

Make sure that the request id in **both** config/init requests is the
same. If they are the same, then try reinitializing the context manager.

## Health Checks

Both services, ingestion and retrieval, provide health check endpoints
that can be used to verify service status:

```python
import requests

# Check Data Ingestion Service health
response = requests.get("http://localhost:8001/health/live")
print(response.text)

# Check Data Ingestion Service health
response = requests.get("http://localhost:8001/health/ready")
print(response.text)

# Check Retrieval Service health
response = requests.get("http://localhost:8000/health/live")
print(response.text)

# Check Retrieval Service health
response = requests.get("http://localhost:8000/health/ready")
print(response.text)
```

## Common Error Messages and Solutions

1. "Context manager not initialized"
   - **Cause**: Service not initialized or wrong UUID
   - **Solution**: Initialize both services with the same UUID

2. "No context was provided"
   - **Cause**: Documents not properly ingested or wrong UUID
   - **Solution**: Verify documents were added successfully and UUIDs match

3. "Invalid request format"
   - **Cause**: Incorrect JSON structure in request
   - **Solution**: Check request format against examples in the documentation

## Monitoring and Debugging

### Important Log Messages

1. Data Ingestion Service:
   - "Adding doc {index}" - Document being processed
   - "Batch {index} is full. Processing ..." - Batch processing
   - "start_pts or end_pts not found" - Missing timestamp metadata

2. Retrieval Service:
   - "Using {model} as the {type} llm" - Model initialization
   - "Setting up QnA, rag type: {type}" - RAG configuration

### Metrics

Both services expose metrics at `/metrics` endpoint:
- Request latency
- Document processing time
- Error rates
- System resource usage
