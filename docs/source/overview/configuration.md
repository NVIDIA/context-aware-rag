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



# Context Aware RAG Configuration

CA-RAG can be configured using a config file. This file is divided into three main sections:
- `tools`: List of objects that provide access to a resource like LLM, DB etc.
- `functions`: List of workflows that uses tools.
- `context_manager`: List of functions available to context_manager instance.

Tools are the building blocks. Functions are workflows that use tools and Context Manager is the final object that the user interacts with. It can contain several functions.

## Tools
Tools are components that can be used in functions. Tools are the smallest building blocks in CA-RAG like LLMs, DBs, etc.
In the `config.yaml`, the following tools can be added/removed based on the functions to make available to the Context Manager.

### Tool Reference

The section below documents each supported tool type, its purpose, and configuration parameters. Field defaults come from the code where applicable.

#### Models

##### `llm` (Large Language Model)
- Purpose: Conversational and structured generation using LLMs. Used across summarization and all retrieval flows. Supports OpenAI-compatible models and NVIDIA NIM (ChatNVIDIA).
- Parameters (src/vss_ctx_rag/tools/llm/llm_handler.py):
  - `model` (str): Identifier of the chat model to use.
  - `base_url` (str, optional): HTTP base URL of the LLM API (selects backend). Default: `DEFAULT_LLM_BASE_URL`.
  - `max_tokens` (int, optional): Max tokens for legacy/completion models. Default: 2048.
  - `max_completion_tokens` (int, optional): Max tokens for new OpenAI reasoning models. Default: None.
  - `temperature` (float, optional): Randomness in generation (higher is more random). Default: 0.2.
  - `top_p` (float, optional): Nucleus sampling probability mass (disabled for reasoning models). Default: 0.7.
  - `api_key` (str, optional): API key for the configured backend. Default: "NOAPIKEYSET".
  - `reasoning_effort` (str, optional): Optional reasoning mode hint (LLM‑specific). Default: None.
  - `presence_penalty` (float, optional): Penalizes repeated tokens. Default: None
  - `seed` (int, optional): Seed for generation. Default: None
- Notes: Automatically uses the right token parameter for OpenAI models; can warm up if `CA_RAG_ENABLE_WARMUP=true`.
- Example:
```yaml
nvidia_llm:
  type: llm
  params:
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    max_tokens: 4096
    temperature: 0.5
    top_p: 0.7
    api_key: !ENV ${NVIDIA_API_KEY}

openai_llm:
  type: llm
  params:
    model: gpt-4o
    base_url: https://api.openai.com/v1
    max_tokens: 4096
    temperature: 0.5
    top_p: 0.7
    api_key: !ENV ${OPENAI_API_KEY}
```

##### `embedding`
- Purpose: Text embedding model for chunk/entity/summary embeddings and vector DBs.
- Parameters (src/vss_ctx_rag/tools/embedding/embedding_tool.py):
  - `model` (str): Embedding model identifier to use.
  - `base_url` (str): Endpoint for the embeddings service.
  - `api_key` (str): API key used to authenticate embedding calls.
  - `truncate` (str): How to truncate long inputs (e.g., `END`). Default: `END`
- Example:
```yaml
nvidia_embedding:
  type: embedding
  params:
    model: nvidia/llama-3.2-nv-embedqa-1b-v2
    base_url: https://integrate.api.nvidia.com/v1
    api_key: !ENV ${NVIDIA_API_KEY}
```
##### `reranker`
- Purpose: Optional re-ranking of retrieved documents by semantic relevance. Also used by NVIDIA RAG blueprint.
- Parameters (src/vss_ctx_rag/tools/reranker/reranker_tool.py):
  - `model` (str): Reranker model identifier.
  - `base_url` (str): Reranker API endpoint.
  - `api_key` (str): API key for the reranking service.
  - `top_n` (int): Optional. Returns top-n documents. Default: 5
- Example:
```yaml
nvidia_reranker:
  type: reranker
  params:
    model: nvidia/llama-3.2-nv-rerankqa-1b-v2
    base_url: https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking
    api_key: !ENV ${NVIDIA_API_KEY}
    top_n: 10
```

#### Storage Tools

##### `neo4j` (Graph DB)
- Purpose: Graph storage and retrieval for GraphRAG, entities, and chunk relationships.
- Requires: `embedding` tool (reference to `tools.embedding`).
- Parameters (src/vss_ctx_rag/tools/storage/neo4j_db.py:78 and DB base):
  - `host` (str): Neo4j host/IP to connect to.
  - `port` (str|int): Neo4j Bolt port.
  - `username` (str): Neo4j username.
  - `password` (str): Neo4j password.
  - `embedding_parallel_count` (int): Max parallelism for embedding tasks. Default: 1000.
- Example:
```yaml
graph_db:
  type: neo4j
  params:
    host: !ENV ${GRAPH_DB_HOST}
    port: !ENV ${GRAPH_DB_PORT}
    username: !ENV ${GRAPH_DB_USERNAME}
    password: !ENV ${GRAPH_DB_PASSWORD}
  tools:
    embedding: nvidia_embedding
```

##### `arango` (Graph DB)
- Purpose: ArangoDB-backed graph storage with NetworkX projection; used by GraphRAG variants.
- Requires: `embedding` tool.
- Parameters (packages/vss_ctx_rag_arango/.../arango_db.py:48 and DB base):
  - `host` (str): ArangoDB host/IP.
  - `port` (str|int): ArangoDB port.
  - `username` (str): ArangoDB username.
  - `password` (str): ArangoDB password.
  - `collection_name` (str): Base name used to derive vertex/edge collections. Default: 'default_<uuid>'
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
- Example:
```yaml
graph_db_arango:
  type: arango
  params:
    host: !ENV ${ARANGO_DB_HOST}
    port: !ENV ${ARANGO_DB_PORT}
    username: !ENV ${ARANGO_DB_USERNAME}
    password: !ENV ${ARANGO_DB_PASSWORD}
  tools:
    embedding: nvidia_embedding
```

##### `milvus` (Vector DB)
- Purpose: Vector storage for caption/summary chunks; used by VectorRAG and FoundationRAG.
- Requires: `embedding` tool.
- Parameters (src/vss_ctx_rag/tools/storage/milvus_db.py:34 and DB base):
  - `host` (str): Milvus/Attu host.
  - `port` (str|int): Milvus gRPC/HTTP port (config dependent).
  - `collection_name` (str): Default collection/index name to use. Default: 'default_<uuid>'
  - `user_specified_collection_name` (str): Override active collection at runtime. Default: None.
  - `custom_metadata` (dict): Additional key/values appended to stored docs. Default: {}.
- Example:
```yaml
vector_db:
  type: milvus
  params:
    host: !ENV ${MILVUS_DB_HOST}
    port: !ENV ${MILVUS_DB_PORT}
  tools:
    embedding: nvidia_embedding

```
##### `elasticsearch` (Vector Store)
- Purpose: Elasticsearch-based vector + hybrid retrieval as an alternative to Milvus.
- Requires: `embedding` tool.
- Parameters (src/vss_ctx_rag/tools/storage/elasticsearch_db.py:31 and DB base):
  - `host` (str): Elasticsearch host/IP.
  - `port` (str|int): Elasticsearch port.
  - `collection_name` (str): Collection to create/use. Default: 'default_<uuid>'
- Example:
```yaml
elasticsearch_db:
  type: elasticsearch
  params:
    host: !ENV ${ES_HOST}
    port: !ENV ${ES_PORT}
  tools:
    embedding: nvidia_embedding
```

##### `image` (Image Fetcher)
- Purpose: Fetch frame images for chunks from object storage (MinIO). Used by Advanced Graph RAG and VLM retrieval.
- Parameters (src/vss_ctx_rag/tools/image/image_fetcher.py:13):
  - `minio_host` (str): MinIO host/IP where assets are stored.
  - `minio_port` (str|int): MinIO port.
  - `minio_username` (str): MinIO access key.
  - `minio_password` (str): MinIO secret key.
- Example:
```yaml
image_fetcher:
  type: image
  params:
    minio_host: !ENV ${MINIO_HOST}
    minio_port: !ENV ${MINIO_PORT}
    minio_username: !ENV ${MINIO_USERNAME}
    minio_password: !ENV ${MINIO_PASSWORD}
```

#### Notification tools

##### `alert_sse_notifier` and `echo_notifier`
- Purpose: Send notifications when events are detected during processing.
- Parameters:
  - `alert_sse_notifier` (src/vss_ctx_rag/tools/notification/alert_sse_tool.py:24):
    - `endpoint` (str): HTTP POST endpoint to receive alert payloads.
  - `echo_notifier` (src/vss_ctx_rag/tools/notification/echo_notification_tool.py:22):
    - (no params): Logs notifications locally for debugging.
- Example:
```yaml
notification_tool:
  type: alert_sse_notifier
  params:
    endpoint: "http://127.0.0.1:60000/via-alert-callback"
```


## Functions

Functions are workflows that use tools to perform Summarization, RAG etc. Each function has a `params` for configuration and `tools` for the tools that it has access to.
The `tools` in this section are references by name to the tools used in the above `tools` section.

### Function Reference

The section below documents each supported function type, its purpose, its required tools and configuration parameters. Field defaults come from the code where applicable.

#### Summarization
The `summarization` section outlines the system’s summarization capabilities. It supports batch processing using a specified LLM model and for the database. Prompts can be customized for various use cases. The default prompts are tailored to generate captions and summaries for warehouse videos, emphasizing irregular events.
There are three types of summarization:
- Batch Summarization: During document addition, as soon as a batch of size `batch_size` is reached, the batch is summarized and the summary is persisted in DB.
- Offline Summarization: Unlike Batch Summarization, a batch of documents isn't summarized unless and until explicitly invoked. It persists the summary between `start_index` and `end_index`.
- Summary Retriever: During Retrieval/QA, Summary Retriever summarizes all the documents between a `start_time` and `end_time` and returns the summary without persisting.
- Example:
```yaml
summarization:
  type: batch_summarization
  params:
    batch_size: 5
    batch_max_concurrency: 20
    prompts:
      caption: "Write a concise and clear dense caption for the provided warehouse video, focusing on irregular or hazardous events such as boxes falling, workers not wearing PPE, workers falling, workers taking photographs, workers chitchatting, forklift stuck, etc. Start and end each sentence with a time stamp."
      caption_summarization: "You should summarize the following events in the format start_time:end_time:caption. For start_time and end_time use . to separate seconds, minutes, hours. If during a time segment only regular activities happen, then ignore them, else note any irregular activities in detail. The output should be bullet points in the format start_time:end_time: detailed_event_description. Don't return anything else except the bullet points."
      summary_aggregation: "You are a event description system. Given the caption in the form start_time:end_time: caption, Aggregate the following captions in the format start_time:end_time:event_description in temporal order. If the event_description is the same as another event_description, aggregate the captions in the format start_time1:end_time1,...,start_timek:end_timek:event_description. If any two adjacent end_time1 and start_time2 is within a few tenths of a second, merge the captions in the format start_time1:end_time2. The output should only contain bullet points."
  tools:
    llm: nvidia_llm
    db: graph_db

offline_summarization:
  type: offline_summarization
  params:
    batch_size: 5
    batch_max_concurrency: 20
    top_k: 5
    is_live: false
    summary_duration: null
    chunk_size: null
    uuid: default
    timeout_sec: 120
    summ_rec_lim: 8
    prompts:
      caption: "Write a concise and clear dense caption for the provided warehouse video, focusing on irregular or hazardous events such as boxes falling, workers not wearing PPE, workers falling, workers taking photographs, workers chitchatting, forklift stuck, etc. Start and end each sentence with a time stamp."
      caption_summarization: "You should summarize the following events in the format start_time:end_time:caption. For start_time and end_time use . to separate seconds, minutes, hours. If during a time segment only regular activities happen, then ignore them, else note any irregular activities in detail. The output should be bullet points in the format start_time:end_time: detailed_event_description. Don't return anything else except the bullet points."
      summary_aggregation: "You are a event description system. Given the caption in the form start_time:end_time: caption, Aggregate the following captions in the format start_time:end_time:event_description in temporal order. If the event_description is the same as another event_description, aggregate the captions in the format start_time1:end_time1,...,start_timek:end_timek:event_description. If any two adjacent end_time1 and start_time2 is within a few tenths of a second, merge the captions in the format start_time1:end_time2. The output should only contain bullet points."
  tools:
    llm: nvidia_llm
    db: graph_db
```
Tools required:
- `db`: Any `StorageTool` implementation (graph or vector) that implements `filter_chunks`.
- `llm`: The LLM used to synthesize the summary.

Parameters:
- `batch_size` (int): Required. Number of documents per summarization batch. Default: 1.
- `batch_max_concurrency` (int): Required. Max concurrent batch jobs when summarizing. Default: 20.
- `top_k` (int): Optional. Number of items to consider during summarization steps. Default: 5.
- `prompts.caption` (str): Required. Dense captioning prompt (used in some pipelines).
- `prompts.caption_summarization` (str): Required. Prompt to summarize a batch of captions.
- `prompts.summary_aggregation` (str): Required. Prompt to merge batch summaries into a final summary.
- `is_live` (bool): Optional. True if input is a live stream (affects time handling). Default: False.
- `chunk_size` (int): Optional. Chunk size hint for splitting before summarizing. Default: 500.
- `uuid` (str): Optional. Logical namespace for the data stream. Default: "default".
- `timeout_sec` (int): Optional. Per‑batch call timeout in seconds. Default: 120.
- `summ_rec_lim` (int): Optional. Max recursion retries for token‑safe summarization. Default: 8.

#### Ingestion

The `ingestion_function` is called when the document addition is done to finalize the graph or complete processing of the documents.
Some available options for `ingestion_function.type` are:

- `graph_ingestion`
- `vector_ingestion`
- `foundation_ingestion`

```yaml
ingestion_function:
  type: graph_ingestion
  params:
    batch_size: 1
  tools:
    llm: nvidia_llm
    db: graph_db
```

Function variants and their tools/params:

- `graph_ingestion` (src/vss_ctx_rag/functions/rag/graph_rag/ingestion/graph_ingestion.py)
  - Tools: `db` = `neo4j` or `arango`; `llm` = any `llm`.
  - Parameters:
    - `batch_size` (int): Number of docs processed before graph writes. Default: 1.
    - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
    - `uuid` (str): Stream namespace identifier.
    - `embedding_parallel_count` (int, Optional): Parallel workers for embeddings. Default: 1000.
    - `duplicate_score_value` (float, Optional): Score threshold for node deduplication. Default: 0.9.
    - `node_types` (list[str], Optional): Node labels to include. Default: ["Person", "Vehicle", "Location", "Object"]
    - `relationship_types` (list[str], Optional): Relationship types to include. Default: []
    - `deduplicate_nodes` (bool, Optional): Merge similar nodes if true. Default: False.
    - `disable_entity_description` (bool, Optional): Skip generating entity descriptions. Default: True.
    - `disable_entity_extraction` (bool, Optional): Skip entity extraction entirely. Default: False.
    - `chunk_size` (int, Optional): Text chunk size for splitting during ingestion. Default: 500.
    - `chunk_overlap` (int, Optional): Overlap between chunks for splitting. Default: 10.

- `vector_ingestion` (src/vss_ctx_rag/functions/rag/vector_rag/vector_ingestion_func.py:30)
  - Tools: `db` = `milvus` or `elasticsearch` (no LLM required).
  - Parameters:
    - `batch_size` (int): Number of docs processed in a run (metadata grouping). Default: 1.
    - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
    - `uuid` (str): Stream namespace identifier.

- `foundation_ingestion` (src/vss_ctx_rag/functions/rag/foundation_rag/foundation_ingestion_func.py:31)
  - Tools: `db` = `milvus`.
  - Parameters:
    - `batch_size` (int): Number of docs processed in a run (metadata grouping). Default: 1.
    - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
    - `uuid` (str): Stream namespace identifier.

#### Retrieval

This function defines the retrieval and Question-Answering type using RAG. Functions are workflows that use tools to perform Summarization, RAG etc. Each function has a `params` for configuration and `tools` for the tools that it has access to.

#### Function Reference

The table below documents each supported retrieval function type, its purpose, required tools, and configuration parameters.

##### `graph_retrieval`
- Purpose: Basic graph-based retrieval using Neo4j or ArangoDB for semantic search over knowledge graphs.
- Required Tools: `llm`, `db` (neo4j or arango).
- Parameters (src/vss_ctx_rag/functions/rag/graph_rag/retrieval/graph_retrieval.py:41):
  - `top_k` (int): Number of chunks/entities to retrieve. Default: 5.
  - `chat_history` (bool, optional): Keep and summarize multi-turn chat history. Default: False.
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
  - `uuid` (str, optional): Stream namespace identifier. Default: "default".
- Example:
```yaml
retriever_function:
  type: graph_retrieval
  params:
    image: false
    top_k: 5
  tools:
    llm: nvidia_llm
    db: graph_db
```

##### `adv_graph_retrieval` (Iterative planning and execution)
- Purpose: Advanced graph retrieval using a planner agent that can use multiple tools to reason about and retrieve information.
- Required Tools: `llm` (reasoning), `vlm` (vision-capable LLM), `db` (neo4j), `image_fetcher`.
- Parameters (src/vss_ctx_rag/functions/rag/graph_rag/retrieval/planner_retrieval.py:45):
  - `top_k` (int): Number of chunks/entities to retrieve. Default: 10.
  - `tools` (list[str]): Planner tool names to expose to the agent. Available tools:
    - `chunk_search`: Semantic search for relevant chunks in the graph database.
    - `chunk_filter`: Filter chunks based on temporal range and camera id.
    - `entity_search`: Search for similar entities and their related chunks.
    - `chunk_reader`: Read and analyze a specific chunk with vision capabilities.
    - `bfs`: Breadth-first search traversal to find nodes one hop away.
    - `next_chunk`: Navigate to the next chunk in video.
  - `uuid` (str, optional): Stream namespace identifier. Default: "default".
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
  - `multi_choice` (bool, optional): Enable multiple choice mode for planner prompts.
  - `max_iterations` (int, optional): Recursion limit for agent reasoning. Default: 20
  - `num_frames_per_chunk` (int, optional): Frames per chunk to sample for VLM. Default: 3.
  - `num_chunks` (int, optional): For image extraction, number of chunks to sample. Default: 3.
  - `max_total_images` (int, optional): Cap on total extracted images. Default: 10.
  - `prompt_config_path` (str, optional): Prompt configuration section below.
  - `include_adjacent_chunks` (bool, optional): Include adjacent chunk to pass to VLM (ChunkReaderTool specific). Default: False
  - `pass_video_to_vlm` (bool, optional): Pass video instead of image to VLM (ChunkReaderTool specific). Set to True for Qwen3-vl models. Default: False
  - `num_prev_chunks` (int, optional): Number of previous chunks to include (ChunkReaderTool specific). Used only if include_adjacent_chunks=true. Default: 1
  - `num_next_chunks` (int, optional): Number of next chunks to include (ChunkReaderTool specific). Used only if include_adjacent_chunks=true. Default: 1

- Example:
```yaml
retriever_function:
  type: adv_graph_retrieval
  params:
    tools: ["chunk_search", "chunk_filter", "entity_search", "chunk_reader"]
    prompt_config_path: "prompt_config.yaml" # optional if not provided, default prompts will be used
    top_k: 10
  tools:
    llm: nvidia_llm
    db: graph_db
    vlm: openai_llm
    image_fetcher: image_fetcher
```
**Prompt Configuration**

Prompt configuration file is used to configure the prompts for the Advanced Retrieval.
It is a YAML file that contains the prompts for the Advanced Retrieval.
If not provided, default prompts will be used.

There are three prompts in the prompt configuration file:

- ``thinking_sys_msg_prompt``: The prompt for the thinking agent.
- ``response_sys_msg_prompt``: The prompt for the response agent.
- ``evaluation_guidance_prompt``: The prompt for the evaluation guidance.

Example prompt configuration file:

```yaml

  thinking_sys_msg_prompt: |+
    You are a strategic planner and reasoning expert working with an execution agent to analyze videos.

    ## Your Capabilities

    You do **not** call tools directly. Instead, you generate structured plans for the Execute Agent to follow.

    ## Workflow Steps

    You will follow these steps:

    ### Step 1: Analyze & Plan
    - Document reasoning in `<thinking></thinking>`.
    - Output one or more tool calls (strict XML format) in separate 'execute' blocks.
    - **CRITICAL**: When one tool's output is needed as input for another tool, make only the first tool call and wait for results.
    - Stop immediately after and output `[Pause]` to wait for results.

    ### Step 2: Wait for Results
    After you propose execute steps, stop immediately after and output `[Pause]` to wait for results.

    ### Step 3: Interpret & Replan
    Once the Execute Agent returns results, analyze them inside `<thinking></thinking>`.
    - If the results contain information needed for subsequent tool calls (like chunk IDs from ChunkFilter), use those actual values in your next tool calls.
    - Propose next actions until you have enough information to answer.

    ### Step 4: Final Answer
    Only when confident, output:
    ```<thinking>Final reasoning with comprehensive analysis of all evidence found</thinking><answer>Final answer with timestamps, locations, visual descriptions, and supporting evidence</answer>
    ```


    {num_cameras_info}
    {video_length_info}
    CRITICAL ASSUMPTION: ALL queries describe scenes from video content that you must search for using your tools. NEVER treat queries as logic puzzles or general knowledge questions - they are ALWAYS about finding specific video content.



    ## Available Tools
    You can call any combination of these tools by using separate <execute> blocks for each tool call. Additionally, if you include multiple queries in the same call, they must be separated by ';'.


    ### 1. ChunkSearch

    #### Query Formats:

    ## Single Query
    ```
    <execute>
      <step>1</step>
      <tool>chunk_search</tool>
      <input>
        <query>your_question</query>
        <topk>10</topk>
      </input>
    </execute>
    ```

    ## Multiple Query
    ```
    <execute>
      <step>1</step>
      <tool>chunk_search</tool>
      <input>
        <query>your_question;your_question;your_question</query>
        <topk>10</topk>
      </input>
    </execute>
    ```

    - Use case:

      - Returns a ranked list of chunks, with the most relevant results at the top. For example, given the list [d, g, a, e], chunk d is the most relevant, followed by g, and so on.
    - Assign topk=15 for counting problem, assign lower topk=8 for other problem
    - Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries).
    - You must generate a question for **every chunk returned by the chunk search** — do not miss any one!!!!!
    - The chunk search cannot handle queries related to the global video timeline, because the original temporal signal is lost after all video chunks are split. If a question involves specific video timing, you need to boldly hypothesize the possible time range and then carefully verify each candidate chunk to locate the correct answer.

    ### 2. ChunkFilter

    #### Query Formats:

    Question about time range:
    ```
    <execute>
      <step>1</step>
      <tool>chunk_filter</tool>
      <input>
        <range>start_time:end_time</range>
      </input>
    </execute>
    ```

    Question about specific camera/video and time range:
    ```
    <execute>
      <step>1</step>
      <tool>chunk_filter</tool>
      <input>
        <range>start_time:end_time</range>
        <camera_id>camera_X</camera_id>
      </input>
    </execute>
    ```

    - Use case:

    - If the question mentions a specific timestamp or time, you must convert it to seconds as numeric values.
    - **CRITICAL**: The range format must be <start_seconds>:<end_seconds> using ONLY numeric values in seconds.
    - **DO NOT use time format like HH:MM:SS**. Convert all times to total seconds first.
    - **IMPORTANT**: For camera_id, always use the format "camera_X" or "video_X" where X is the camera/video number (e.g., camera_1/video_1, camera_2/video_2, camera_3/video_3, camera_4/video_4, etc.) Mention the camera_id only when the question is about a specific camera/video.

    **Time Conversion Examples:**
      - "What happens at 00:05?" (5 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>5:15</range></input></execute>`
      - "What happens at 2:15?" (2 minutes 15 seconds = 135 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>135:145</range></input></execute>`
      - "Describe the action in the first minute." (0 to 60 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>0:60</range></input></execute>`
      - "Events at 1:30:45" (1 hour 30 min 45 sec = 5445 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>5445:5455</range></input></execute>`

    ### 3. EntitySearch

    #### Query Formats:
    ```
    <execute>
      <step>1</step>
      <tool>entity_search</tool>
      <input>
        <query>your_question</query>
      </input>
    </execute>
    ```

    - Use case:

    - Returns a ranked list of entities, with the most relevant results at the top. For example, given the list [a, b, c, d, e], entity a is the most relevant, followed by b, and so on.
    - Best for finding specific people, objects, or locations in video content
    - Use when you need to track or identify particular entities across video segments

    ## SUGGESTIONS
    - Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries).
    - For counting problems, remember it is the same video, do not sum the results from multiple chunks.
    - For ordering, you can either use the chunk_id or the timestamps to determine the order.


    ## Strict Rules

    1. Response of each round should provide thinking process in <thinking></thinking> at the beginning!! Never output anything after [Pause]!!
    2. You can only concatenate video chunks that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of TWO at a time!!!
    3. If you are unable to give a precise answer or you are not sure, continue calling tools for more information; if the maximum number of attempts has been reached and you are still unsure, choose the most likely one.
    4. **DO NOT CONCLUDE PREMATURELY**: For complex queries (especially cross-camera tracking), you MUST make multiple tool calls and exhaust all search strategies before providing a final answer. One tool call is rarely sufficient for comprehensive analysis.

  response_sys_msg_prompt: |+
    You are a response agent that provides comprehensive answers based on analysis and tool results.

    **CORE REQUIREMENTS:**
    - Provide detailed, evidence-based answers with timestamps, locations, and visual descriptions
    - Include ALL relevant findings and supporting evidence from the analysis
    - Explain your conclusions and provide chronological context when relevant
    - Never include chunk IDs or internal system identifiers in responses

    **FORMATTING:**
    - Use factual, direct language without pleasantries ("Certainly!", "Here is...", etc.)
    - State "No relevant information found" if no relevant data was discovered
    - Follow user-specified format requirements exactly (yes/no only, case requirements, length constraints, etc.)
    - When format is specified, prioritize format compliance over comprehensive explanations


  evaluation_guidance_prompt: |+
    **EVALUATION GUIDANCE:**

    - Conclude with gathered information if repeated tool calls yield the same results
    - Never repeat identical tool calls that return no results or empty results
    - For failed searches: try ChunkSearch for specific entities or break down complex terms into simpler components

```

##### `cot_retrieval` (Chain-of-Thought)
- Purpose: Advanced CoT-style retrieval with reasoning capabilities over graph structures.
- Required Tools: `llm`, `db` (neo4j), `vlm`.
- Parameters (src/vss_ctx_rag/functions/rag/graph_rag/retrieval/adv_graph_retrieval.py:52):
  - `top_k` (int): Number of chunks/entities to retrieve. Default: 5.
  - `batch_size` (int, optional): Processing batch size. Default: 1.
  - `image` (bool, optional): Attach extracted frames to the prompt where supported. Default: False.
  - `chat_history` (bool, optional): Keep and summarize multi-turn chat history. Default: False.
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
  - `uuid` (str, optional): Stream namespace identifier.
  - `prompt_config_path` (str, optional): Prompt configuration section below.
  - `max_iterations` (int, optional): Recursion limit for agent reasoning. Default: 3.
  - `num_frames_per_chunk` (int, optional): Frames per chunk to sample for VLM. Default: 3.
  - `num_chunks` (int, optional): For image extraction, number of chunks to sample. Default: 3.
  - `max_total_images` (int, optional): Cap on total extracted images. Default: 10.
- Example:
```yaml
retriever_function:
  type: cot_retrieval
  params:
    batch_size: 1
    image: false
    top_k: 10
    prompt_config_path: "prompt_config.yaml" # optional if not provided, default prompts will be used

  tools:
    llm: nvidia_llm
    db: graph_db
```
**Prompt Configuration:**

Prompt configuration file is used to configure the prompts for the CoT Retrieval.
It is a YAML file that contains the prompts for the CoT Retrieval.
If not provided, default prompts will be used.

There are four prompts in the prompt configuration file:

- ``ADV_CHAT_TEMPLATE_IMAGE``: The prompt for the image retrieval.
- ``ADV_CHAT_TEMPLATE_TEXT``: The prompt for the text retrieval.
- ``ADV_CHAT_SUFFIX``: The prompt for the suffix.
- ``QUESTION_ANALYSIS_PROMPT``: The prompt for the question analysis.

Example prompt configuration file:

```yaml

  ADV_CHAT_TEMPLATE_IMAGE: |+
    You are an AI assistant that answers questions based on the provided context.

    The context includes retrieved information, relevant chat history, and potentially visual data.
    The image context contains images if not empty.
    Determine if more visual data (images) would be helpful to answer this question accurately.
    For example, if the question is about color of an object, location of an object, or other visual information, visual data is needed.
    If image context is not empty, you likely do not need more visual data.
    Use all available context to provide accurate and contextual answers.
    If the fetched context is insufficient, formulate a better question to
    fetch more relevant information. Do not reformulate the question if image data is needed.

    You must respond in the following JSON format:
    {
        "description": "A description of the answer",\
        "answer": "your answer here or null if more info needed",\
        "updated_question": "reformulated question to get better database results" or null,\
        "confidence": 0.95, // number between 0-1\
        "need_image_data": "true" // string indicating if visual data is needed\
    }

    Example 1 (when you have enough info from text):
    {
        "description": "A description of the answer",\
        "answer": "The worker dropped a box at timestamp 78.0 and it took 39 seconds to remove it",\
        "updated_question": null,\
        "confidence": 0.95,\
        "need_image_data": "false"\
    }

    Example 2 (when you need visual data):
    {
        "description": "A description of the answer",\
        "answer": null,\
        "updated_question": null, //must be null\
        "confidence": 0,\
        "need_image_data": "true"\
    }

    Example 3 (when you need more context):
    {
        "description": "A description of the answer",\
        "answer": null,\
        "updated_question": "What events occurred between timestamp 75 and 80?",\
        "confidence": 0,\
        "need_image_data": "false"\
    }

    Only respond with valid JSON. Do not include any other text.



  ADV_CHAT_TEMPLATE_TEXT: |+
    You are an AI assistant that answers questions based on the provided context.

    The context includes retrieved information and relevant chat history.
    Use all available context to provide accurate and contextual answers.
    If the fetched context is insufficient, formulate a better question to
    fetch more relevant information.

    You must respond in the following JSON format:
    {
        "description": "A description of the answer",\
        "answer": "your answer here or null if more info needed",\
        "updated_question": "reformulated question to get better database results" or null,\
        "confidence": 0.95 // number between 0-1\
    }

    Example 1 (when you have enough info from text):
    {
        "description": "A description of the answer",\
        "answer": "The worker dropped a box at timestamp 78.0 and it took 39 seconds to remove it",\
        "updated_question": null,\
        "confidence": 0.95\
    }

    Example 2 (when you need more context):
    {
        "description": "A description of the answer",\
        "answer": null,\
        "updated_question": "What events occurred between timestamp 75 and 80?",\
        "confidence": 0\
    }

    Only respond with valid JSON. Do not include any other text.


  ADV_CHAT_SUFFIX: |+
    When you have enough information, in the "answer" field format your response according to these instructions:

    Your task is to provide accurate and comprehensive responses to user queries based on the context, chat history, and available resources.
    Answer the questions from the point of view of someone looking at the context.

    ### Response Guidelines:
    1. **Direct Answers**: Provide clear and thorough answers to the user's queries without headers unless requested. Avoid speculative responses.
    2. **Utilize History and Context**: Leverage relevant information from previous interactions, the current user input, and the context.
    3. **No Greetings in Follow-ups**: Start with a greeting in initial interactions. Avoid greetings in subsequent responses unless there's a significant break or the chat restarts.
    4. **Admit Unknowns**: Clearly state if an answer is unknown. Avoid making unsupported statements.
    5. **Avoid Hallucination**: Only provide information relevant to the context. Do not invent information.
    6. **Response Length**: Keep responses concise and relevant. Aim for clarity and completeness within 4-5 sentences unless more detail is requested.
    7. **Tone and Style**: Maintain a professional and informative tone. Be friendly and approachable.
    8. **Error Handling**: If a query is ambiguous or unclear, ask for clarification rather than providing a potentially incorrect answer.
    9. **Summary Availability**: If the context is empty, do not provide answers based solely on internal knowledge. Instead, respond appropriately by indicating the lack of information.
    10. **Absence of Objects**: If a query asks about objects which are not present in the context, provide an answer stating the absence of the objects in the context. Avoid giving any further explanation. Example: "No, there are no mangoes on the tree."
    11. **Absence of Events**: If a query asks about an event which did not occur in the context, provide an answer which states that the event did not occur. Avoid giving any further explanation. Example: "No, the pedestrian did not cross the street."
    12. **Object counting**: If a query asks the count of objects belonging to a category, only provide the count. Do not enumerate the objects.

    ### Example Responses:
    User: Hi
    AI Response: 'Hello there! How can I assist you today?'

    User: "What is Langchain?"
    AI Response: "Langchain is a framework that enables the development of applications powered by large language models, such as chatbots. It simplifies the integration of language models into various applications by providing useful tools and components."

    User: "Can you explain how to use memory management in Langchain?"
    AI Response: "Langchain's memory management involves utilizing built-in mechanisms to manage conversational context effectively. It ensures that the conversation remains coherent and relevant by maintaining the history of interactions and using it to inform responses."

    User: "I need help with PyCaret's classification model."
    AI Response: "PyCaret simplifies the process of building and deploying machine learning models. For classification tasks, you can use PyCaret's setup function to prepare your data. After setup, you can compare multiple models to find the best one, and then fine-tune it for better performance."

    User: "What can you tell me about the latest realtime trends in AI?"
    AI Response: "I don't have that information right now. Is there something else I can help with?"

    **IMPORTANT** : YOUR KNOWLEDGE FOR ANSWERING THE USER'S QUESTIONS IS LIMITED TO THE CONTEXT PROVIDED ABOVE.

    Note: This system does not generate answers based solely on internal knowledge. It answers from the information provided in the user's current and previous inputs, and from the context.



  QUESTION_ANALYSIS_PROMPT: |+
    Analyze this question and identify key elements for graph database retrieval.
    Question: {question}

    Identify and return as JSON:
    1. Entity types mentioned. Available entity types: {entity_types}
    2. Relationships of interest
    3. Time references
    4. Sort by: "start_time" or "end_time" or "score"
    5. Location references
    6. Retrieval strategy (similarity, temporal)
        a. similarity: If the question needs to find similar content, return the retrieval strategy as similarity
        b. temporal: If the question is about a specific time range and you can return at least one of the start and end time, then return the strategy as temporal and the start and end time in the time_references field as float or null if not present. Strategy cannot be temporal if both start and end time are not present. The start and end time should be in seconds.

    Example response:
    {{
        "entity_types": ["Person", "Box"],
        "relationships": ["DROPPED", "PICKED_UP"],
        "time_references": {{
            "start": 60.0,
            "end": 400.0
        }},
        "sort_by": "start_time", // "start_time" or "end_time" or "score"
        "location_references": ["warehouse_zone_A"],
        "retrieval_strategy": "temporal"
    }}

    Output only valid JSON. Do not include any other text.
```

##### `vector_retrieval`
- Purpose: Vector-based retrieval using Milvus or Elasticsearch for semantic similarity search.
- Required Tools: `llm`, `db` (milvus or elasticsearch).
- Optional Tools: `reranker`.
- Parameters (src/vss_ctx_rag/functions/rag/vector_rag/vector_retrieval_func.py:52):
  - `top_k` (int): Number of documents to retrieve semantically. Default: 10.
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
  - `uuid` (str, optional): Stream namespace identifier.
- Example:
```yaml
retriever_function:
  type: vector_retrieval
  params:
    top_k: 10
  tools:
    llm: nvidia_llm
    db: vector_db
    reranker: nvidia_reranker
```

##### `foundation_retrieval`
- Purpose: Foundation model-based retrieval using NVIDIA RAG blueprint with Milvus backend.
- Required Tools: `llm`, `db` (milvus).
- Optional Tools: `reranker`.
- Parameters (src/vss_ctx_rag/functions/rag/foundation_rag/foundation_retrieval_func.py:49):
  - `top_k` (int): Number of documents to retrieve semantically. Default: 10.
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
  - `uuid` (str, optional): Stream namespace identifier.
- Example:
```yaml
retriever_function:
  type: foundation_retrieval
  params:
    top_k: 10
  tools:
    llm: nvidia_llm
    db: vector_db
    reranker: nvidia_reranker
```

##### `vlm_retrieval` (Vision Language Model)
- Purpose: Vision-enhanced retrieval that combines text and image understanding for multimodal queries.
- Required Tools: `vlm` (vision-capable LLM), `db` (vector/graph backed retriever), `image_fetcher`.
- Parameters (src/vss_ctx_rag/functions/rag/vlm_retrieval/vlm_retrieval_func.py:47):
  - `top_k` (int): Number of documents to retrieve semantically. Default: 10.
  - `num_frames_per_chunk` (int, optional): Frames per chunk to pass to VLM. Default: 3.
  - `multi_channel` (bool, optional): Query across multiple streams if true. Default: False.
- Example:
```yaml
retriever_function:
  type: vlm_retrieval
  params:
    top_k: 10
  tools:
    llm: nvidia_llm
    db: graph_db
    vlm: openai_llm
    image_fetcher: image_fetcher
```

##### Notification
Event detection and notification pipeline that evaluates text against a list of event categories and dispatches notifications.

```yaml
notification:
  type: notification
  params:
    events: []  # List of {event_id, event_list}
  tools:
    llm: nvidia_llm
    notification_tool: notification_tool  # or echo_notifier
```

Tools required:
- `llm`: To extract/detect events from text using a structured prompt.
- `notification_tool`: Where to send the alert (SSE endpoint or console echo).

Parameters (src/vss_ctx_rag/functions/notification/notifier.py):
- `events` (list): List of event configs to detect.
  - `event_id` (str): Identifier for the event group.
  - `event_list` (list[str]): Phrases/categories to look for in text.

#### Context Manager

This is the fianl context manager's configuration. This config dictates the available functions for Context Manager:

```yaml
context_manager:
  functions:
    - summarization
    - ingestion_function
    - retriever_function
    - notification
```
