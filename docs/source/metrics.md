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

# Metrics

## Otel and TimeMeasure Metrics

The codebase uses OpenTelemetry for tracing and metrics. The following
environment variables can be set to enable metrics:

```bash
export VIA_CTX_RAG_ENABLE_OTEL=true
export VIA_CTX_RAG_EXPORTER=otlp # or console
export VIA_CTX_RAG_OTEL_ENDPOINT=http://otel_collector:4318 # only used if VIA_CTX_RAG_EXPORTER is otlp
```

Traces capture TimeMeasure metrics which are used to monitor the execution time of the different components.

#### Example Span

```json
{
  "name": "GraphRetrieval/Neo4jRetriever",
  "context": {
    "trace_id": "0x0ddaa0e6800dd0f4172746f53a3fc12b",
    "span_id": "0xbf4c3dc3c9050e0e",
    "trace_state": "[]"
  },
  "kind": "SpanKind.INTERNAL",
  "parent_id": null,
  "start_time": "2025-04-09T05:37:28.633505Z",
  "end_time": "2025-04-09T05:37:28.752445Z",
  "status": {
    "status_code": "UNSET"
  },
  "attributes": {
    "span name": "GraphRetrieval/Neo4jRetriever",
    "execution_time_ms": 119.0345287322998
  },
  "events": [],
  "links": [],
  "resource": {
    "attributes": {
      "service.name": "vss-ctx-rag-default"
    },
    "schema_url": ""
  }
}
```

#### Important TimeMeasure Metrics

##### Context Manager
- `context_manager/reset`: Time taken to reset the Context Manager process and clear pending requests.
- `context_manager/configure`: Time taken to apply a new configuration to the Context Manager.
- `context_manager/add_doc`: Time taken to enqueue a document into the Context Manager.
- `context_manager/aprocess_doc/total`: Time taken to process a document across all registered functions.
- `context_manager/aprocess_doc/{func.name}`: Time taken to process a document within a specific function.
- `context_manager/call-manager`: Time taken to orchestrate a call request inside the worker process.
- `context_manager/call/pending_add_doc`: Time taken to wait for in‑flight add_doc requests before a call.
- `context_manager/call`: Time taken to execute all registered functions for a given state payload.


##### Vector Storage (Milvus / Elasticsearch)
- `milvusdb/add caption`: Time taken to add a single caption document to Milvus.
- `Milvus/AddSummries`: Time taken to bulk‑ingest summary documents into Milvus after splitting.
- `elasticsearch/add caption`: Time taken to add a single caption document to Elasticsearch.
- `Elasticsearch/AddSummaries`: Time taken to bulk‑ingest summary documents into Elasticsearch after splitting.

##### Graph RAG — Extraction (Base)
- `GraphRAG/aprocess-doc:`: Time taken to prepare and batch documents for graph creation.
- `GraphRAG/aprocess-doc/graph-create:`: Time taken to create graph structures for a specific batch index.
- `GraphRAG/Base/acreate_graph`: Time taken to end‑to‑end graph extraction for a batch.
- `GraphRAG/Base/add_graph_documents_to_db`: Time taken to persist extracted GraphDocuments via the DB tool.
- `GraphRAG/Base/create_relation_between_chunks`: Time taken to create FIRST_CHUNK and NEXT_CHUNK relations and summary links.
- `GraphRAG/Base/update_embedding_chunks`: Time taken to compute embeddings for chunk nodes and persist them.
- `GraphRAG/Base/FetchEntEmbd`: Time taken to fetch entities requiring embeddings.
- `GraphRAG/Base/UpdateEmbdingBatch`: Time taken to compute embeddings for a batch of entities.
- `GraphRAG/Base/FetchSummaryEmbd`: Time taken to fetch summaries requiring embeddings.
- `GraphRAG/Base/merge_chunk_entity_relationships`: Time taken to link chunks to entities (HAS_ENTITY) in bulk.
- `GraphRAG/Base/apost_process`: Time taken to post‑process after all batches: create doc node, link chunks, embeddings, KNN, dedup.
- `GraphRAG/Acall/graph-extraction/postprocessing`: Time taken to run post‑processing from the GraphRAG function wrapper after batching completes.

##### Graph RAG — Backend: Neo4j
- `GraphRAG/Neo4j/add_graph_documents`: Time taken to bulk add GraphDocuments to Neo4j.
- `GraphRAG/Neo4j/persist_chunk_data`: Time taken to create chunk nodes, PART_OF links, FIRST_CHUNK/NEXT_CHUNK edges.
- `GraphRAG/Neo4j/persist_summary_relations`: Time taken to create IN_SUMMARY/SUMMARY_OF edges for summaries.
- `GraphRAG/Neo4j/persist_chunk_embeddings`: Time taken to persist vector embeddings on Chunk nodes.
- `GraphRAG/Neo4j/merge_chunk_entity_rels`: Time taken to create HAS_ENTITY relations between chunks and entities.
- `GraphRAG/Neo4j/UpdateKNN`: Time taken to build/update KNN relations between chunks from embeddings.
- `GraphRAG/Neo4j/fetch_summaries_for_embedding`: Time taken to read Summary nodes missing embeddings.
- `GraphRAG/Neo4j/merge_duplicate_nodes`: Time taken to merge duplicate entity nodes above a similarity threshold.
- `GraphRAG/Neo4j/persist_entity_embeddings`: Time taken to persist vector embeddings on Entity nodes.
- `GraphRAG/Neo4j/persist_summary_embeddings`: Time taken to persist vector embeddings on Summary nodes.
- `GraphExtraction/VectorIndex`: Time taken to create the Chunk vector index (drop/create lifecycle).
- `GraphExtraction/FetchEntEmbd`: Time taken to fetch nodes to embed during Neo4j ingestion.
- `GraphExtraction/UpdatEmbding`: Time taken to compute embeddings for fetched nodes and prepare persistence.

##### Graph RAG — Backend: Arango / NetworkX
- `NXGraphExtraction/VectorIndex`: Time taken to create vector indexes for both Chunk and Entity collections.
- `NXGraphRAG/add_graph_documents`: Time taken to add GraphDocuments into in‑memory NetworkX graph.
- `NXGraphRAG/persist_chunk_data`: Time taken to persist Chunk nodes and PART_OF/FIRST_CHUNK/NEXT_CHUNK edges in NetworkX.
- `NXGraphRAG/persist_summary_relations`: Time taken to create IN_SUMMARY and SUMMARY_OF edges in NetworkX.
- `NXGraphRAG/persist_chunk_embeddings`: Time taken to persist embeddings on Chunk nodes in NetworkX.
- `NXGraphRAG/persist_chunk_entity_rels`: Time taken to create HAS_ENTITY edges in NetworkX.
- `NXGraphRAG/UpdateKNN`: Time taken to compute KNN over Chunk embeddings and add SIMILAR edges in NetworkX.
- `NXGraphRAG/fetch_entities_for_embedding`: Time taken to read Entity nodes missing embeddings from NetworkX.
- `NXGraphRAG/persist_entity_embeddings`: Time taken to persist embeddings on Entity nodes in NetworkX.
- `NXGraphRAG/fetch_summaries_for_embedding`: Time taken to read Summary nodes missing embeddings from NetworkX.
- `NXGraphRAG/persist_summary_embeddings`: Time taken to persist embeddings on Summary nodes in NetworkX.
- `GraphRAG/ArangoDB/merge_duplicate_nodes`: Time taken to merge duplicate entity groups inside Arango collections.

##### Graph Retrieval
- `GraphRetrieval/RetrieveDocuments`: Time taken to run vector+graph retrieval and format results (Arango backend).
- `GraphRetrieval/GetResponse`: Time taken to ask the LLM to answer using formatted docs (optionally with images).
- `GraphRetrieval/SummarizeChat`: Time taken to summarize chat history and store a concise summary.

##### Planner & Advanced Retrieval
- `Planner/call`: Time taken to run the iterative planner agent across tool calls to answer a question.
- `AdvGraphRetrieval/retrieve_context`: Time taken to analyze the question and retrieve relevant context iteratively.
- `AdvImgGraphRAG/call`: Time taken to advanced Graph RAG call with optional image reasoning.

##### Vector RAG
- `VectorRAG/aprocess-doc/metrics_dump`: Time taken to dump VectorRAG metrics to JSON (when VIA_LOG_DIR is set).
- `VectorRAG/retrieval`: Time taken to end‑to‑end vector‑only retrieval and response generation.

##### Summarization — Online (BatchSummarization)
- `summ/aprocess_doc`: Time taken to ingest a caption, assign batch info, and trigger per‑batch summarization when full.
- `summ/acall/batch-aggregation-summary`: Time taken to aggregate batch summaries into a final summary.
- `OffBatSumm/CombindAgg`: Time taken to combine partial batch summaries and re‑summarize (token‑safe path).
- `OffBatchSumm/Acall`: Time taken to fetch batch texts from storage and aggregate across the requested range.

##### Summarization — Offline (OfflineBatchSummarization)
- `OfflineBatchSumm/aprocess_doc`: Time taken to accumulate docs for later batch processing.
- `OfflineBatchSumm/ProcessAccumulatedBatches`: Time taken to process all full batches gathered so far.
- `OfflineBatchSumm/ProcessBatch_{batch.get_batch_index()}`: Time taken to summarize a single batch and persist.
- `OfflineBatchSumm/acall/batch-aggregation-summary`: Time taken to aggregate stored batch summaries into a final summary.
- `OfflineBatchSumm/Acall`: Time taken to orchestrate offline summarization (fetch, aggregate, and output).

##### Summary Retriever
- `summary_retriever/acall`: Time taken to filter chunks by time/camera, then summarize with an LLM prompt.

##### Notifications
- `notifier/llm_call`: Time taken to run the LLM classifier over configured events for a document.
- `notifier/notify_call`: Time taken to send notifications for detected events with metadata.

##### VLM Retrieval
- `VLMRetrieval/retrieval`: Time taken to retrieve captions, extract images, and answer using a vision‑capable LLM.

##### Foundation RAG
- `FoundationRAG/aprocess-doc/metrics_dump`: Time taken to dump FoundationRAG metrics to JSON (when VIA_LOG_DIR is set).
- `FoundationRAG/retrieval`: Time taken to retrieve with NVIDIA RAG service (hybrid search + optional reranker) and respond.
