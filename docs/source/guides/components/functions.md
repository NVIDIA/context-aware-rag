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

# Functions

## Writing Custom Functions

The `Function` class serves as a base class for all functions in Context Manager. The idea of the `Function` class is to transform a state dictionary. Tools and functions can be added to the function, as well as configure the function with parameters.

## Creating a Custom Function

To create a custom function, you need to:

1. Create a configuration class and register it with a unique type name
2. Create a function implementation class and register it with the config
3. Implement the abstract methods `setup`, `acall`, and `aprocess_doc`

### Create and Register a Function Configuration

First, create a configuration class that defines the parameters your function needs and register it with a unique type name using the `@register_function_config` decorator.

```python
from pydantic import BaseModel
from vss_ctx_rag.models.function_models import register_function_config
from typing import ClassVar, Dict, List, Optional

@register_function_config("custom_function")
class CustomFunctionConfig(BaseModel):
    """Configuration for custom function."""

    # Define allowed tool types that this function can use
    # The format is {tool_type: [tool_keywords]}
    # keyword can be any name but will determine the name of the tool in the function
    # get_tool("db") will return the tool with the keyword "db" of type "vector_db"
    # This is optional and can be omitted if not needed
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "vector_db": ["db"]
    }

    class CustomFunctionParams(BaseModel):
        param1: Optional[int] = 10
        param2: int

    params: CustomFunctionParams
```

### Create and Register the Function Implementation

Next, create the function implementation class that inherits from `Function` and register it with the configuration using the `@register_function` decorator.

```python
from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.function_models import register_function

@register_function(config=CustomFunctionConfig)
class CustomFunction(Function):
    def __init__(self, name: str):
        super().__init__(name)
```

### Implement the `setup` Method

The `setup` method is used to initialize the function.
In this method, you can get params and tools that are added to the function in the configuration.

```python
def setup(self) -> dict:
    """
    Initialize the function.
    """
    self.param1 = self.get_param("param1", default=10)
    self.param2 = self.get_param("param2")
    self.vector_db = self.get_tool("db")
    self.llm = self.get_tool("llm")
    return {}
```

### Implement the `acall` Method

The `acall` method is used to asynchronously call the function. The input is a state, represented by a dictionary. The output should also be a state dictionary.

```python
def acall(self, state: dict) -> dict:
    """
    Call the function.
    """
    ## Do some work here
    return {"new_key": "new_value"}
```

### Implement the `aprocess_doc` Method

The `aprocess_doc` method is used to process a document.

```python
def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
    """
    Process a document.
    """
    ## Database operations can be done here
```

### Add the Function to Configuration

With the registration system, functions are now configured declaratively in your configuration YAML file instead of being added programmatically. The function type you registered (e.g., `"custom_function"`) can now be used in your config file:

```yaml
functions:
  my_custom_function:
    type: custom_function  # This matches the name used in @register_function_config
    params:
      param1: 15
      param2: 50
      max_results: 50
    tools:
      llm: nvidia_llm      # Reference to a tool defined in the tools section
      vector_db: milvus_db # Reference to another tool
```

The Context Manager will automatically instantiate your function based on the configuration, eliminating the need for manual `add_function` calls.

Functions are also created in topologically sorted order so no need to worry about dependencies.

## Configuration Example

Here's a complete example showing how your registered function would appear in a configuration file:

```yaml
tools:
  nvidia_llm:
    type: llm
    params:
      model: meta/llama-3.1-70b-instruct
      api_key: !ENV ${NVIDIA_API_KEY}

  milvus_db:
    type: milvus
    params:
      host: localhost
      port: 19530

functions:
  my_custom_function:
    type: custom_function
    params:
      param1: 15
      param2: 50
    tools:
      llm: nvidia_llm
      vector_db: milvus_db
```
