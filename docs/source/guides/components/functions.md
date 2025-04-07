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

# Writing Custom Functions

The `Function` class serves as a base class for all functions in Context Manager. The idea of the `Function` class is to transform a state dictionary. Tools and functions can be added to the function, as well as configure the function with parameters.

## Creating a Custom Function

To create a custom function, you need to inherit from the `Function` class and implement the abstract methods `setup`, `acall`, and `aprocess_doc`.

### Inherit from the `Function` Class

Create a new class that inherits from `Function`.

```python
from vss_ctx_rag.base.function import Function

class CustomFunction(Function):
    def __init__(self, name: str):
        super().__init__(name)
```

### Implement the `setup` Method

The `setup` method is used to initialize the function. It should return a dictionary of parameters that will be used by the function.
In this method, you can get params and tools that are added to the function.

```python
def setup(self) -> dict:
    """
    Initialize the function.
    """
    self.batch_size = self.get_param("params", "batch_size")
    self.vector_db = self.get_tool("vector_db")
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

### Add the Function to the Context Manager

To add the function to the Context Manager, use the `add_function` method. Add tools and functions to the function, as well as configure the function with parameters here.
```python
ctx_manager.add_function(
    CustomFunction("custom_function")
    .add_tool(LLM_TOOL_NAME, self.llm)
    .config(**config)
    .done()
)
```

### Optional: Add a Function to another Function

A function can also be added to another function.

```python
function.add_function(
    CustomFunction("custom_function")
    .add_function(
        CustomFunction("custom_function_2")
        .add_tool(LLM_TOOL_NAME, self.llm)
        .config(**config)
        .done()
    )
    .config(**config)
    .done()
)
```
