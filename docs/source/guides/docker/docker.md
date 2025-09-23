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

# Standalone Docker

## Running the Standalone Docker Service

### Prerequisites

-   [Docker](https://docs.docker.com/)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

#### Milvus

``` bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh


bash standalone_embed.sh start
```

This will start the milvus service by default on port 19530.

If using Graph-RAG, you will need to run the following container.

#### Graph-RAG: Neo4J

``` bash
docker run -d \
  --name neo4j \
  -p <NEO4J_PORT>:7687 \
  -e NEO4J_AUTH=<GRAPH_DB_USERNAME>/<GRAPH_DB_PASSWORD> \
  neo4j:5.26.4
```

#### Export Environment Variables

Create a .env file in the root directory and set the following
variables:

``` bash
MILVUS_DB_HOST=<HOST> #milvus host, e.g. localhost
MILVUS_DB_PORT=<MILVUS_PORT> #milvus port, e.g. 19530

GRAPH_DB_URI=bolt://<HOST>:<NEO4J_PORT> #neo4j uri, e.g. bolt://localhost:7687
GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME> #neo4j username, e.g. neo4j
GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD> #neo4j password, e.g. password
```

## Build the vss_ctx_rag image

Make sure you are in the project root directory.

``` bash
make -C docker build
```

## Run the data ingestion service

``` bash
make -C docker start_in
```

## Run the data retrieval service

``` bash
make -C docker start_ret
```
