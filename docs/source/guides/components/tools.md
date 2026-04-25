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
# Tools

## Adding Custom Tools

Tools are components that can be used in functions. Tools can include LLMs, databases, etc.

To create a custom tool, you need to:

1. Create a configuration class and register it with a unique type name
2. Create a tool implementation class and register it with the config
3. Implement tool-specific methods

### Create and Register a Tool Configuration

First, create a configuration class that defines the parameters your tool needs and register it with a unique type name using the `@register_tool_config` decorator.

```python
from pydantic import BaseModel
from vss_ctx_rag.models.tool_models import register_tool_config
from typing import ClassVar, Dict, List

@register_tool_config("custom_tool")
class CustomToolConfig(BaseModel):
    """Configuration for custom tool."""

    # Define allowed tool types that this tool can depend on (optional)
    # The format is {tool_type: [tool_keywords]}
    # keyword can be any name but will determine the name of the tool in the function
    # get_tool("db_embedding") will return the tool with the keyword "db_embedding" of type "embedding"
    # This is optional and can be omitted if not needed
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "embedding": ["db_embedding"]
    }

    host: str
    port: int
    api_key: str
    timeout: int
```

### Create and Register the Tool Implementation

Next, create the tool implementation class that inherits from `Tool` and register it with the configuration using the `@register_tool` decorator. Make sure to pass the config and tools to the super constructor.

```python
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import register_tool

@register_tool(config=CustomToolConfig)
class CustomTool(Tool):
    def __init__(self, name: str, config: CustomToolConfig, tools=None):
        super().__init__(name, config, tools)
        self.config = config
```


### Implement Tool Methods

**Define Tool-Specific Methods**: Implement any methods that your tool needs to perform its tasks. These methods can interact with external services or perform computations.

```python
def connect(self):
    """Connect to the external service."""
    self.connection = Client(
        host=self.config.params.host,
        port=self.config.params.port,
        api_key=self.config.params.api_key,
        timeout=self.config.params.timeout,
        embedding=self.get_tool("db_embedding")
    )

def process_data(self, data):
    """Process data using the external service."""
    return self.connection.process(data)
```

### Add the Tool to Configuration

With the registration system, tools are now configured declaratively in your configuration YAML file instead of being added programmatically. The tool type you registered (e.g., `"custom_tool"`) can now be used in your config file:

```yaml
tools:
  my_custom_tool:
    type: custom_tool  # This matches the name used in @register_tool_config
    params:
      host: localhost
      port: 8080
      api_key: !ENV ${CUSTOM_API_KEY}
      timeout: 60
    tools:
      embedding: nvidia_embedding  # Reference to another tool if needed

functions:
  my_function:
    type: some_function
    tools:
      custom_tool: my_custom_tool  # Reference the tool defined above
```

Now in your function setup, a tool can be retrieved by name:

```python
def setup(self) -> dict:
    self.custom_tool = self.get_tool("custom_tool")
    return {}

def acall(self, state: dict) -> dict:
    result = self.custom_tool.process_data(state["input_data"])
    return {"processed_result": result}
```


Tools are also created in topologically sorted order so no need to worry about dependencies between tools.

## Configuration Example

Here's a complete example showing how your registered tool would appear in a configuration file:

```yaml
tools:
  my_custom_tool:
    type: custom_tool
    params:
      host: api.example.com
      port: 443
      api_key: !ENV ${CUSTOM_API_KEY}
      timeout: 30
    tools:
      embedding: nvidia_embedding

  nvidia_embedding:
    type: embedding
    params:
      model: nvidia/llama-3.2-nv-embedqa-1b-v2
      api_key: !ENV ${NVIDIA_API_KEY}

  # Example using null embeddings for testing/development
  null_embedding:
    type: embedding
    params:
      enable: false          # Disable to use null embeddings (no API calls)
      dimensions: 1024       # Optional, defaults to 1024

functions:
  my_function:
    type: retrieval_function
    tools:
      custom_tool: my_custom_tool
      embedding: nvidia_embedding  # Or use null_embedding for testing
```

## Built-in Tools

### Embedding Tools

The framework provides an `embedding` tool type that supports both real NVIDIA embeddings and null embeddings for testing.

#### NVIDIA Embeddings

Use NVIDIA embeddings for production deployments:

```yaml
nvidia_embedding:
  type: embedding
  params:
    model: nvidia/llama-3.2-nv-embedqa-1b-v2
    base_url: https://integrate.api.nvidia.com/v1
    api_key: !ENV ${NVIDIA_API_KEY}
    truncate: END
```

#### Null Embeddings (Testing/Development)

Use null embeddings when you want to test your pipeline without making actual API calls. Null embeddings generate deterministic dummy embeddings that are useful for:

- **Testing**: Run unit tests without API dependencies
- **Development**: Develop and debug your pipeline offline
- **Cost Savings**: Avoid API costs during development

```yaml
test_embedding:
  type: embedding
  params:
    enable: false
    dimensions: 1024  # Optional, defaults to 1024
```

**Features:**
- **No API calls**: Generates embeddings locally without network requests
- **Deterministic**: Same text always produces the same embedding (reproducible tests)
- **Fast**: Instant embedding generation
- **Drop-in replacement**: Uses the same interface as NVIDIA embeddings

**Example in complete configuration:**

```yaml
tools:
  # Use null embeddings for testing
  test_embedding:
    type: embedding
    params:
      enable: false
      dimensions: 768

  # Use real embeddings for production
  prod_embedding:
    type: embedding
    params:
      model: nvidia/llama-3.2-nv-embedqa-1b-v2
      api_key: !ENV ${NVIDIA_API_KEY}

  vector_db:
    type: milvus
    params:
      host: localhost
      port: 19530
    tools:
      embedding: test_embedding  # Switch to prod_embedding for production

functions:
  vector_retrieval:
    type: vector_retrieval
    params:
      top_k: 10
    tools:
      llm: nvidia_llm
      db: vector_db
```
