# Running the Service

### Prerequisites

Either have a running Milvus or Neo4j instance depending on the
configuration you want to use (milvus for Vector-RAG or neo4j for
Graph-RAG).

#### Export the following envs as needed

``` bash
# Required if using NVIDIA Endpoints
NVIDIA_API_KEY=<NVIDIA_API_KEY>

# If using OpenAI Endpoints
OPENAI_API_KEY=<OPENAI_API_KEY>

# For Vector-RAG
MILVUS_HOST=<HOST>
MILVUS_PORT=<MILVUS_PORT>

# For Graph-RAG
GRAPH_DB_URI=bolt://<HOST>:<NEO4J_PORT>
GRAPH_DB_USERNAME=<USERNAME>
GRAPH_DB_PASSWORD=<PASSWORD>
```

#### Start the ingestion service

``` bash
export VIA_CTX_RAG_ENABLE_RET=false
uvicorn service.service:app --host 0.0.0.0 --port <INGEST_PORT>
```

#### Start the retrieval service

``` bash
export VIA_CTX_RAG_ENABLE_RET=true
uvicorn service.service:app --host 0.0.0.0 --port <RETRIEVAL_PORT>
```

## Next Steps

Once the services are running, refer to the [API Documentation](api.md) for detailed information about:
- Service initialization
- Adding documents
- Querying the system
- Best practices and troubleshooting
