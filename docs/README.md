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

# Building Documentation

### Prerequisites

### Set up virtual environment
1. Install [Astral UV](https://docs.astral.sh/uv/getting-started/installation/)
1. Create virtual environment using UV: `uv venv`.
1. Activate the virtual environment using: `source .venv/bin/activate`.
1. Install packages into the virtualenv using: `uv sync`.

If you don't already have a uv environment set up, refer to the [Prerequisites](./source/intro/setup.md) guide.

## Install Documentation Dependencies

```bash
uv pip install -e ".[docs]"
```

## Build the Documentation

```bash
make -C docs
```

## Verify the Documentation

```bash
firefox docs/build/html/index.html
```
