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

# Running with Docker

### Prerequisites

-   Docker
-   NVIDIA Container Toolkit

#### Setting up env

Create a .env file in the root directory and set the following
variables:

``` bash
NVIDIA_API_KEY=<IF USING NVIDIA>
NVIDIA_VISIBLE_DEVICES=<GPU ID>

OPENAI_API_KEY=<IF USING OPENAI>

VSS_CTX_PORT_RET=<DATA RETRIEVAL PORT>
VSS_CTX_PORT_IN=<DATA INGESTION PORT>


# Milvus Configuration
MILVUS_DB_HTTP_PORT=<MILVUS_HTTP_PORT> #milvus HTTP port, e.g. 9091
MILVUS_DB_PORT=<MILVUS_GRPC_PORT> #milvus GRPC port, e.g. 19530

# Neo4j Configuration
GRAPH_DB_HTTP_PORT=<NEO4J_HTTP_PORT> #neo4j HTTP port, e.g. 7474
GRAPH_DB_PORT=<NEO4J_PORT> #neo4j bolt port, e.g. 7687
GRAPH_DB_USERNAME=<USERNAME> #neo4j username, e.g. neo4j
GRAPH_DB_PASSWORD=<PASSWORD> #neo4j password, e.g. password

# MinIO Configuration
MINIO_PORT=<MINIO_PORT> #minio API port, e.g. 9000
MINIO_WEBUI_PORT=<MINIO_WEBUI_PORT> #minio web UI port, e.g. 9001
MINIO_USERNAME=<MINIO_USER> #minio root user, e.g. minio
MINIO_PASSWORD=<MINIO_PASSWORD> #minio root password, e.g. minio123

# ArangoDB Configuration
ARANGO_DB_PORT=<ARANGO_PORT> #arangodb port, e.g. 8529
ARANGO_DB_USERNAME=<ARANGO_USER> #arangodb username
ARANGO_DB_PASSWORD=<ARANGO_PASSWORD> #arangodb password
```

### Using docker compose


#### Build the Context Aware RAG image

``` bash
make -C docker build
```

#### Starting the container/services

``` bash
make -C docker start_compose
```

This will start the following services:

-   vss-ctx-rag-data-ingestion
-   vss-ctx-rag-retriever
-   neo4j-db
    -   UI available at <http://>\<HOST\>:7474
-   milvus-standalone
-   minio
    -   UI available at <http://>\<HOST\>:9001
-   arango-db
    -   UI available at <http://>\<HOST\>:8529
-   otel-collector
-   phoenix
    -   UI available at <http://>\<HOST\>:16686
-   prometheus
    -   UI available at <http://>\<HOST\>:9090
-   grafana
    -   UI available at <http://>\<HOST\>:3000
-   cassandra
-   cassandra-schema

To change the storage volumes, export DOCKER_VOLUME_DIRECTORY to the
desired directory.

#### Stop the services

``` bash
make -C docker stop_compose
```
