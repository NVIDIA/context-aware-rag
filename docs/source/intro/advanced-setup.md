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


# Advanced setup and Usage

CA-RAG supports the following databases as backends for storing and retrieving documents:
- Milvus
- Elasticsearch
- Neo4j
- ArangoDB

CA-RAG supports the following retrieval methods for Question-Answering and Retrieval:
- VectorRAG (VRAG)
- Basic GraphRAG (GRAG)
- Chain-of-Thought Retrieval and QA (CoT)
- Chain-of-Thought Retrieval with Vision Language Model (VLM)
- Foundation-RAG using [RAG NVIDIA blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) (FRAG)
- Advanced Graph Retrieval with Graph Traversal and VLM (AdvGRAG)

You can choose from one of the following databases and one of the supported Question-Answering configuration:


## Supported Configurations

The following table shows the compatibility matrix between databases and retrieval methods:
| Database / Retrieval Method | VRAG | FRAG | GRAG | CoT | VLM | AdvGRAG |
|----------------------------|:----:|:----:|:----:|:---:|:---:|:--------:|
| **Milvus**                 | ✅   | ✅   | -    | -   | -   | -        |
| **Elasticsearch**          | ✅   | -    | -    | -   | -   | -        |
| **Neo4j**                  | -    | -    | ✅   | ✅  | ✅  | ✅       |
| **ArangoDB**               | -    | -    | ✅   | ✅  | ✅  | ✅       |

## Database setup

Use the following section to setup and start the Database for the desired RAG configuration. Choose the DB to start based on the above table.

### Vector-RAG: Milvus

```bash
export MILVUS_DB_HOST=${MILVUS_DB_HOST} #milvus host, e.g. localhost
export MILVUS_DB_PORT=${MILVUS_DB_PORT} #milvus port, e.g. 19530
export NVIDIA_API_KEY=${NVIDIA_API_KEY} #NVIDIA API key
```
This will start the milvus service by default on port 19530.

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh


bash standalone_embed.sh start
```

### Graph-RAG: Neo4j

```bash
export GRAPH_DB_HOST=${GRAPH_DB_HOST} #neo4j
export GRAPH_DB_PORT=${GRAPH_DB_PORT} #neo4j port, e.g. 7687
export GRAPH_DB_USERNAME=${GRAPH_DB_USERNAME} #neo4j username, e.g. neo4j
export GRAPH_DB_PASSWORD=${GRAPH_DB_PASSWORD} #neo4j password, e.g. password
export NVIDIA_API_KEY=${NVIDIA_API_KEY} #NVIDIA API key
```
```bash
docker run -d \
  --name neo4j \
  -p ${GRAPH_DB_HTTP_PORT:-7474}:7474 \
  -p ${GRAPH_DB_BOLT_PORT:-7687}:7687 \
  -e NEO4J_AUTH=${GRAPH_DB_USERNAME:-neo4j}/${GRAPH_DB_PASSWORD:-passneo4j} \
  -e NEO4J_PLUGINS='["apoc"]' \
  neo4j:5.26.4
```


### Graph-RAG: Arango

1. Export the environment variables
```bash
export ARANGO_DB_HOST=${ARANGO_DB_HOST} #arango
export ARANGO_DB_PORT=${ARANGO_DB_PORT} #arango port, e.g. 8529
export ARANGO_DB_USERNAME=${ARANGO_DB_USERNAME} #arango username, e.g. root
export ARANGO_DB_PASSWORD=${ARANGO_DB_PASSWORD} #arango password, e.g. password
export NVIDIA_API_KEY=${NVIDIA_API_KEY} #NVIDIA API key
```
2. Start docker container for the Arango DB
```bash
docker run -d \
  --name arango-db \
  -p ${ARANGO_DB_PORT:-8529}:${ARANGO_DB_PORT:-8529} \
  -e ARANGO_DB_USERNAME=${ARANGO_DB_USERNAME} \
  -e ARANGO_ROOT_PASSWORD=${ARANGO_DB_PASSWORD} \
  arangodb/arangodb:3.12.4 \
  arangod --experimental-vector-index --server.endpoint http://0.0.0.0:${ARANGO_DB_PORT:-8529}
```
3. Install additional dependencies for Arango.
```bash
uv pip install -e ".[arango]"
```
4. Update `config.yaml`
```yaml
tools:
# ... existing tools
  graph_db_arango:
    type: arango
    params:
      host: !ENV ${ARANGO_DB_HOST}
      port: !ENV ${ARANGO_DB_PORT}
      username: !ENV ${ARANGO_DB_USERNAME}
      password: !ENV ${ARANGO_DB_PASSWORD}
    tools:
      embedding: nvidia_embedding

functions:
# ... existing functions
    summarization:
        type: batch_summarization
        # ... update the db `tools`
        db: graph_db_arango
    ingestion_function:
        type: graph_ingestion
        # ... update the db in `tools`
        db: graph_db_arango
    retriever_function:
        type: graph_retrieval
        # ... update the db in `tools`
        db: graph_db_arango
    summary_retriever:
        type: summary_retriever
        # ... update the db in `tools`
        db: graph_db_arango
```

### Vector-RAG: Elasticsearch

1. Export the required environment variables
```bash
export ES_HOST=${ES_HOST} #elastic search host, e.g. localhost
export ES_PORT=${ES_PORT} #elastic search port eg.
export NVIDIA_API_KEY=${NVIDIA_API_KEY} #NVIDIA API key
```
2. Start the Elasticsearch DB if not already started via [Docker Compose](../guides/docker/compose.md)
```bash
  docker run -d \
    --name elasticsearch \
    -p ${ES_PORT:-9200}:${ES_PORT:-9200} \
    -p ${ES_TRANSPORT_PORT:-9300}:${ES_TRANSPORT_PORT:-9300} \
    -e discovery.type=single-node \
    -e xpack.security.enabled=false \
    --memory=${ES_MEM_LIMIT:-6442450944} \
    elasticsearch:9.1.2
```
3. Update `config.yaml`
```yaml
tools:
# ... existing tools
  elasticsearch_db:
    type: elasticsearch
    params:
      host: !ENV ${ES_HOST}
      port: !ENV ${ES_PORT}
    tools:
      embedding: nvidia_embedding

functions:
# ... existing functions
    summarization:
        type: batch_summarization
        # ... update the db `tools`
        db: elasticsearch_db
    ingestion_function:
        type: vector_ingestion
        # ... update the db in `tools`
        db: elasticsearch_db
    retriever_function:
        type: vector_retrieval
        # ... update the db in `tools`
        db: elasticsearch_db
    summary_retriever:
        type: summary_retriever
        # ... update the db in `tools`
        db: elasticsearch_db
```


## Retrieval Setup

To change the type of retrieval used for Question-Answering, you can choose one of the following configurations:

- VectorRAG (VRAG)
- Basic GraphRAG (GRAG)
- Chain-of-Thought Retrieval and QA (CoT)
- Chain-of-Thought Retrieval with Vision Language Model (VLM)
- Foundation-RAG using [RAG NVIDIA blueprint](https://github.com/NVIDIA-AI-Blueprints/rag) (FRAG)
- Advanced Graph Retrieval with Graph Traversal and VLM (AdvGRAG)


## RAG Type Configuration Examples

Once the required Database is set up, started and the environment variables are set up, the following types of RAG can be setup by modifying the `config.yaml`.
The following sections show configuration snippets for different RAG types and the changes needed from the base configuration above.

### Vector-RAG (VRAG)

Vector-RAG uses vector databases for document storage and retrieval with embedding-based similarity search. During document addition, each document is stored as a Chunk into a vectorstore like Milvus or Elasticsearch. During Retrieval/QA, the relevant documents are fetched and used as context for answering the user's question.

**How it works:**
- Captions generated by the Vision-Language Model (VLM), along with their embeddings, are stored in Milvus DB or Elasticsearch
- Embeddings can be created using any embedding NIM (by default, `nvidia/llama-3_2-nv-embedqa-1b-v2`)
- For a query, the top five most similar chunks are retrieved using vector similarity
- Retrieved chunks are re-ranked using any reranker NIM (by default, `nvidia/llama-3_2-nv-rerankqa-1b-v2`)
- Re-ranked chunks are passed to a Large Language Model (LLM) NIM to generate the final answer

Full setup and example as described at [Setup](../intro/setup.md).
A full config file can be found at [data/configs/vrag.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/vrag.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools ...
  vector_db:
    type: milvus
    params:
      host: !ENV ${MILVUS_DB_HOST}
      port: !ENV ${MILVUS_DB_GRPC_PORT}
    tools:
      embedding: nvidia_embedding

  nvidia_reranker:
    type: reranker
    params:
      model: nvidia/llama-3.2-nv-rerankqa-1b-v2
      base_url: "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
      api_key: !ENV ${NVIDIA_API_KEY}

functions:
  # ... existing functions ...
  summarization:
    type: batch_summarization
    # ... update the db in tools
    tools:
      llm: nvidia_llm
      db: vector_db

  ingestion_function:
    type: vector_ingestion
    # ... update the db in tools
      db: vector_db

  retriever_function:
    type: vector_retrieval
    # ... update the db in tools
    tools:
      llm: nvidia_llm
      db: vector_db
      reranker: nvidia_reranker

# ... rest of config
```

### Basic GraphRAG (GRAG)

GraphRAG uses graph databases (Neo4j/Arango) to store and retrieve documents with entity-relationship graphs. During document addition, nodes and relationships are created. After documents are added, document ingestion is called that finalizes the graph by creating community summaries and making it available for Question-Answering during retrieval. During Retrieval, the relevant nodes/entities and graph community summarizes are used as context to answer user's query.

**How it works:**
- **Graph Extraction:** Entities and relationships are extracted from VLM captions using an LLM and stored in a GraphDB
- Captions and embeddings (generated with any embedding NIM) are linked to these entities
- **Graph Retrieval:** For a given query, relevant entities, relationships, and captions are retrieved from the GraphDB
- Retrieved information is passed to an LLM NIM to generate the final answer


Full setup and example as described at [Setup](../intro/setup.md).

The DB can be one of ArangoDB or Neo4j. Refer to [Configuration](../overview/configuration.md) for more details.

A full config file can be found at [data/configs/grag.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/grag.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools ...
  graph_db:
    type: neo4j
    params:
      host: !ENV ${GRAPH_DB_HOST}
      port: !ENV ${GRAPH_DB_BOLT_PORT}
      username: !ENV ${GRAPH_DB_USERNAME}
      password: !ENV ${GRAPH_DB_PASSWORD}
    tools:
      embedding: nvidia_embedding
functions:
  # ... existing functions ...
  summarization:
    type: batch_summarization
    # ... update the db in tools
      db: graph_db

  ingestion_function:
    type: graph_ingestion
    # ... update the db in tools
      db: graph_db

  retriever_function:
    type: graph_retrieval
    # ... update the db in tools
      db: graph_db

# ... rest of config
```

### Chain-of-Thought Retrieval and QA (CoT)

COT adds Chain-of-Thought reasoning capabilities to graph-based retrieval. This enables the LLM to perform multi-step querying on the graph to collect enough information to answer user query for multi-hop reasoning type queries.

**Key Features:**
- **Iterative Retrieval:** Performs multiple retrieval iterations (up to `max_iterations`) until a confident answer is found
- **Confidence Scoring:** Uses a confidence threshold to determine answer quality (default: 0.7)
- **Question Reformulation:** LLM can suggest updated questions to retrieve better database results
- **Chat History Integration:** Maintains conversation context using the last 3 interactions
- **Visual Data Processing:** Can request and analyze video frames when visual information is needed
- **Structured Response:** Returns JSON-formatted responses with answer, confidence, and additional metadata

**Retrieval Process:**
1. Initial context retrieval based on the user question
2. Integration of relevant chat history from previous interactions
3. Iterative LLM evaluation with structured JSON response format
4. If confidence is below threshold:
   - Request additional context using reformulated questions
   - Process visual data if needed (when image features enabled)
   - Continue iteration until confident answer or max iterations reached
5. Return final answer with confidence score


A full config file can be found at [data/configs/grag.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/cot.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools from GraphRAG ...
  openai_llm:
    type: llm
    params:
      model: gpt-4o
      base_url: https://api.openai.com/v1
      max_tokens: 4096
      temperature: 0.5
      top_p: 0.7
      api_key: !ENV ${OPENAI_API_KEY}

functions:
  # ... existing functions from GraphRAG ...
  summarization:
    type: batch_summarization
    # ... update the db in tools
    tools:
      # ... existing tools
      db: graph_db
  ingestion_function:
    type: graph_ingestion
    # ... update the db in tools
    tools:
      # ... existing tools
      db: graph_db

  retriever_function:
    type: cot_retrieval
    # ... update db and vlm
    tools:
      # ... existing tools
      db: graph_db
      vlm: openai_llm

# ... rest of config
```

### Chain-of-Thought Retrieval with Vision Language Model (VLM)

This type of retrieval strategy uses Chain-of-Thought reasoning to iteratively search graph and documents through multi-step query. It also uses VLM to enable visual understanding capabilities for image-based retrieval and analysis.

**How it works:**
- Captions generated by the Vision-Language Model (VLM), along with their embeddings and video frame paths, are stored in different databases
- Video frames are stored in MinIO object storage
- Based on the user query, the most relevant chunk and related video frames are retrieved
- Retrieved chunks and frames are passed to a Vision Language Model (VLM) NIM along with the query to generate the final answer
- Embeddings are created using any embedding NIM (by default, `nvidia/llama-3_2-nv-embedqa-1b-v2`)


A full config file can be found at [data/configs/vlm.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/vlm.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools from GraphRAG  ...
  openai_llm:
    type: llm
    params:
      model: gpt-4o
      base_url: https://api.openai.com/v1
      max_tokens: 4096
      temperature: 0.5
      top_p: 0.7
      api_key: !ENV ${OPENAI_API_KEY}

  image_fetcher:
    type: image
    params:
      minio_host: !ENV ${MINIO_HOST}
      minio_port: !ENV ${MINIO_PORT}
      minio_username: !ENV ${MINIO_USERNAME}
      minio_password: !ENV ${MINIO_PASSWORD}

functions:
  # ... existing functions from GraphRAG ...
  retriever_function:
    type: vlm_retrieval
    params:
      top_k: 10
    tools:
      llm: nvidia_llm
      db: graph_db
      vlm: openai_llm
      image_fetcher: image_fetcher
# ... rest of config
```

### Foundation-RAG (FRAG)

Foundation-RAG provides enhanced vector-based retrieval with Milvus based on the [NVIDIA RAG blueprint](https://github.com/NVIDIA-AI-Blueprints/rag). This uses NVIDIAs RAG blueprint to retrieve documents with reranking for Question Answering. During document adding, the documents can be added to the Milvus DB provided by NVIDIA RAG blueprint. During retrieval, CA-RAG can connect to the external Milvus and perform Question-Answering over NVIDIA RAG blueprint's document collection.

A full config file can be found at [data/configs/frag.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/frag.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools from VectorRAG...
  vector_db:
    type: milvus
    params:
      host: !ENV ${MILVUS_DB_HOST}
      port: !ENV ${MILVUS_DB_GRPC_PORT}
    tools:
      embedding: nvidia_embedding

  nvidia_reranker:
    type: reranker
    params:
      model: nvidia/llama-3.2-nv-rerankqa-1b-v2
      base_url: "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking"
      api_key: !ENV ${NVIDIA_API_KEY}

functions:
  # ... existing functions from VectorRAG...
  summarization:
    type: batch_summarization
    # ... update the db in tools
    tools:
      llm: nvidia_llm
      db: vector_db

  ingestion_function:
    type: foundation_ingestion
    # ... update the db in tools
    params:
      batch_size: 1
    tools:
      llm: nvidia_llm
      db: vector_db

  retriever_function:
    type: foundation_retrieval
    # ... update the db in tools
    tools:
      llm: nvidia_llm
      db: vector_db
      reranker: nvidia_reranker

# ... rest of config
```

### Advanced Graph Retrieval with Graph Traversal and VLM (AdvGRAG)

This agent combines advanced graph retrieval with Graph Traversal capabilities and vision language models.

**Architecture Components:**
- **Planning Module:** Creates execution plans and evaluates results to determine next steps
- **Execution Engine:** Parses XML-structured plans and creates tool calls
- **Tool Node:** Executes specialized search and analysis tools
- **Response Formatter:** Formats final answers based on all collected information

**Available Traversal Strategies:**
- `chunk_search`: Retrieves the most relevant chunks using vector similarity
- `entity_search`: Retrieves entities and relationships using vector similarity
- `chunk_filter`: Filters chunks based on time ranges and camera IDs
- `chunk_reader`: Analyzes chunks and video frames using VLM for detailed insights
- `bfs`: Performs breadth-first search through entity relationships
- `next_chunk`: Retrieves chronologically adjacent chunks

**Iterative Process:**
1. **Planning Phase:** Planning module creates initial execution plan
2. **Execution Phase:** Execution engine parses plan and calls appropriate tools
3. **Evaluation Phase:** Results are evaluated; if incomplete, cycle repeats with refined plan
4. **Response Phase:** Final answer is generated when sufficient information is gathered

**Key Features:**
- **Multi-Channel Support:** Handles multiple camera streams with runtime camera information
- **Dynamic Tool Selection:** Uses only the tools specified in configuration
- **Iterative Refinement:** Continues until confident answer or max iterations reached
- **XML-Structured Plans:** Uses structured XML format for reliable plan parsing
- **Context Awareness:** Integrates video length and camera metadata into planning

A full config file can be found at [data/configs/planner_vlm.yaml](https://github.com/NVIDIA/context-aware-rag/tree/main/data/configs/planner_vlm.yaml)

**Key Changes from Base Config:**
```yaml
tools:
  # ... existing tools from GraphRAG ...
  openai_llm:
    type: llm
    params:
      model: gpt-4o
      base_url: https://api.openai.com/v1
      max_tokens: 4096
      temperature: 0.5
      top_p: 0.7
      api_key: !ENV ${OPENAI_API_KEY}

  image_fetcher:
    type: image
    params:
      minio_host: !ENV ${MINIO_HOST}
      minio_port: !ENV ${MINIO_PORT}
      minio_username: !ENV ${MINIO_USERNAME}
      minio_password: !ENV ${MINIO_PASSWORD}

functions:
  # ... existing functions from GraphRAG ...
  retriever_function:
    type: adv_graph_retrieval
    params:
      tools: ["chunk_search", "chunk_filter", "entity_search", "chunk_reader"]
      top_k: 10
    tools:
      llm: nvidia_llm
      db: graph_db
      vlm: openai_llm
      image_fetcher: image_fetcher

context_manager:
  functions:
    - summarization
    - ingestion_function
    - retriever_function
```
