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

# Adding Custom Tools

Tools are components that can be used in functions. Tools can include LLMs, databases, etc.

### Create a New Tool Class

1. **Inherit from the `Tool` Class**: Start by creating a new class that inherits from the `Tool` class. This will serve as the base for your custom tool.

```python
from vss_ctx_rag.base.tool import Tool

class CustomTool(Tool):
    def __init__(self, name: str):
        super().__init__(name)
```


### Implement Tool Methods

**Define Tool-Specific Methods**: Implement any methods that your tool needs to perform its tasks. These methods can interact with external services or perform computations.

```python
def say_hello(self):
    print("Hello, world!")
```

### Integrate the Tool into a Function

**Add the Tool to a Function**: Use the `add_tool` method to add your custom tool into a function. This allows the function to utilize the tool's capabilities.

```python
custom_tool = CustomTool("custom_tool")

CustomFunction("custom_function_2")
    .add_tool("custom_tool", custom_tool)
    .config(**config)
    .done()
```
Now in our function setup, a tool can be retrieved by name.

```python
tool = self.get_tool("custom_tool")
tool.say_hello()
```
