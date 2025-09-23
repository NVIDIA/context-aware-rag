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

# Initialization

This guide explains how to initialize the Context-Aware RAG services.

## Service Startup

The Context-Aware RAG services follow the startup guide in the [Docker Compose Guide](../../guides/docker/compose.md).

## Service Initialization

Both services must be initialized before use with the same UUID to ensure proper communication.

### Config File

Both services need a config file to initialize. Refer to the [Configuration](../../intro/setup.md) guide for more information on the config file.

Note: Currently vector ingestion and retrieval are not supported for CA-RAG service.

### Initialize Data Ingestion Service

```python
import requests

url = "http://localhost:8001/init"
headers = {
    "Content-Type": "application/json"
}
data = {
    "config_path": "/app/config/config.yaml",
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

### Initialize Retrieval Service

```python
import requests

url = "http://localhost:8000/init"
headers = {
    "Content-Type": "application/json"
}
data = {
    "config_path": "/app/config/config.yaml",
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

**Important**: Use the same UUID for both services to ensure they can access the same context.

## UUID Management

### Best Practices

1. **UUID Generation**
   - Use unique UUIDs for different sessions
   - Consider using timestamp-based UUIDs for easy tracking
   - Avoid reusing UUIDs across different contexts

2. **UUID Consistency**
   - Always use the same UUID for both services
   - Store the UUID securely if needed for later use
   - Document UUID usage in your application

3. **UUID Security**
   - Treat UUIDs as sensitive information
   - Avoid exposing UUIDs in logs or error messages
   - Implement proper UUID validation

## Configuration

The initialization process requires a configuration file at `/app/config/config.yaml`. This file controls various aspects of the system:

- Vector store settings
- Model parameters
- Chunking configuration
- Logging settings

Make sure the configuration file is properly set up before initializing the services.
