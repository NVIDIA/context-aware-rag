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

# Installing Context Aware RAG AIQ Plugin

## AIQ Library Setup

Clone the AIQ library from
[here](https://github.com/NVIDIA/AgentIQ/tree/develop)

Complete the following steps to setup the environment

1. Clone the AgentIQ repository to your local machine.
    ```bash
    git clone git@github.com:NVIDIA/AgentIQ.git agentiq
    cd agentiq
    ```

2. Initialize, fetch, and update submodules in the Git repository.
    ```bash
    git submodule update --init --recursive
    ```

3. Create a Python environment.
    ```bash
    uv venv --seed .venv
    source .venv/bin/activate
    uv sync
    ```

If you are having issues with the above steps, reference the [AgentIQ Documentation](https://docs.nvidia.com/agentiq/latest/index.html)

## Install the Context Aware RAG library

### Install from source

Clone the Context Aware RAG library from [here]()

Install the Context Aware RAG library

``` bash
uv pip install -e .
```
