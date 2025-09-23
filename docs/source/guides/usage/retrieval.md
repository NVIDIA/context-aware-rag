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

# Retrieval

This guide explains how to query documents in the Context-Aware RAG system.

## Making Queries

Queries can be made to the system using the `/chat/completions` endpoint of the Retrieval Service.

### Request Format

```json
{
  "model": "meta/llama-3.1-70b-instruct",
  "base_url": "https://integrate.api.nvidia.com/v1",
  "messages": [{"role": "user", "content": "Your question here"}],
  "uuid": "unique-request-id"
}
```

### Example Query

```python
import requests
import json

url = "http://localhost:8000/chat/completions"
headers = {"Content-Type": "application/json"}
chat_data = {
    "model": "meta/llama-3.1-70b-instruct",
    "base_url": "https://integrate.api.nvidia.com/v1",
    "messages": [{"role": "user", "content": "Who mentioned the fire?"}],
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, data=json.dumps(chat_data))
print(response.json()["choices"][0]["message"]["content"])
```

### Query Parameters

- `model`: The model to use for the completion (e.g., "meta/llama-3.1-70b-instruct")
- `base_url`: The base URL for the API (e.g., "https://integrate.api.nvidia.com/v1")
- `messages`: Array of message objects with `role` and `content` fields
- `uuid`: Unique identifier for the request


## Summary Query

Summary query can be made to the system using the `/summary` endpoint of the Retrieval Service.

start_index: The start index of the batch summary (e.g., 0)
end_index: The end index of the batch summary (e.g., -1)

### Request Format

```json
{
    "uuid": "your_session_uuid",
    "summarization": {
        "start_index": 0,
        "end_index": -1
    }
}
```

### Example Query

```python
import requests

url = "http://localhost:8000/summary"
headers = {"Content-Type": "application/json"}
data = {
    "uuid": "your_session_uuid",
    "summarization": {
        "start_index": 0,
        "end_index": -1
    }
}

response = requests.post(url, headers=headers, json=data)
print(response.json()["result"])
```

## Best Practices

1. **Question Formulation**
   - Be specific and clear in your questions
   - Use natural language
   - Avoid overly complex or multi-part questions

2. **Message Structure**
   - Use clear role assignments ("user", "assistant", "system")
   - Structure your content clearly within the message
   - Provide meaningful UUIDs for request tracking

3. **Error Handling**
   - Always check response status codes
   - Handle timeouts appropriately
   - Implement retry logic for failed requests
