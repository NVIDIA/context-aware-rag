# Qdrant Integration Guide

This guide explains how to use Qdrant as a vector database backend for the NVIDIA Context Aware RAG system.

## Overview

Qdrant is a high-performance vector similarity search engine with a convenient API. This integration provides seamless support for Qdrant as an alternative to Milvus and Elasticsearch for vector storage and retrieval operations.

## Features

- **Full VectorStorageTool Implementation**: Complete support for all RAG operations
- **LangChain Integration**: Uses `langchain-qdrant` for seamless LangChain compatibility
- **Flexible Configuration**: Support for local and cloud deployments
- **gRPC Support**: Optional high-performance gRPC protocol
- **Collection Management**: Dynamic collection switching and management
- **Metadata Filtering**: Advanced filtering capabilities for time-based and metadata queries

## Installation

The Qdrant dependencies are included in the main package:

```bash
pip install vss_ctx_rag
```

Or if installing from source:

```bash
uv sync
```

The following packages will be installed:
- `qdrant-client>=1.12.0` - Official Qdrant Python client
- `langchain-qdrant>=0.2.0` - LangChain integration for Qdrant

## Quick Start

### 1. Start Qdrant Server

Using Docker:

```bash
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
```

Or using Docker Compose (add to your `docker-compose.yaml`):

```yaml
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"  # HTTP API
      - "6334:6334"  # gRPC API
    volumes:
      - ./qdrant_storage:/qdrant/storage:z
    environment:
      - QDRANT__SERVICE__GRPC_PORT=6334
```

### 2. Configure Environment Variables

Set the following environment variables:

```bash
export QDRANT_DB_HOST=localhost
export QDRANT_DB_PORT=6333
export NVIDIA_API_KEY=your_nvidia_api_key

# For Qdrant Cloud (optional)
export QDRANT_API_KEY=your_qdrant_cloud_api_key
```

### 3. Update Configuration

Modify your `config.yaml` to use Qdrant:

```yaml
tools:
  vector_db:
    type: qdrant
    params:
      host: !ENV ${QDRANT_DB_HOST:localhost}
      port: !ENV ${QDRANT_DB_PORT:6333}
      # Optional configurations:
      # api_key: !ENV ${QDRANT_API_KEY}
      # prefer_grpc: true
      # collection_name: my_custom_collection
    tools:
      embedding: nvidia_embedding
```

See `config/config-qdrant-example.yaml` for a complete example.

## Configuration Options

### QdrantDBConfig Parameters

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `host` | string | Yes | - | Qdrant server hostname |
| `port` | string | Yes | - | Qdrant server port (6333 for HTTP, 6334 for gRPC) |
| `api_key` | string | No | None | API key for Qdrant Cloud authentication |
| `prefer_grpc` | boolean | No | false | Use gRPC protocol instead of HTTP |
| `collection_name` | string | No | auto-generated | Name of the Qdrant collection to use |
| `custom_metadata` | dict | No | {} | Custom metadata to add to all documents |
| `user_specified_collection_name` | string | No | None | Dynamic collection name for runtime switching |

### Environment Variables

- `QDRANT_DB_HOST`: Qdrant server host (default: `localhost`)
- `QDRANT_DB_PORT`: Qdrant server port (default: `6333`)
- `QDRANT_API_KEY`: API key for Qdrant Cloud (optional)
- `VSS_CTX_RAG_ENABLE_RET`: Enable retention mode (default: `False`)
- `VIA_CTX_RAG_ENABLE_RET`: Enable ingestion retention (default: `True`)

## Using Qdrant Cloud

To use Qdrant Cloud instead of a local instance:

```yaml
tools:
  vector_db:
    type: qdrant
    params:
      host: "your-cluster.aws.cloud.qdrant.io"
      port: "6333"
      api_key: !ENV ${QDRANT_API_KEY}
      prefer_grpc: true  # Recommended for cloud
    tools:
      embedding: nvidia_embedding
```

## Architecture

### Class Hierarchy

```
StorageTool (ABC)
└── VectorStorageTool (ABC)
    └── QdrantDBTool
```

### Key Components

1. **QdrantDBConfig**: Configuration model with validation
2. **QdrantDBTool**: Main implementation class
3. **LangChain Integration**: Uses `QdrantVectorStore` for LangChain compatibility

### Implemented Methods

The `QdrantDBTool` class implements all required abstract methods:

#### Storage Operations
- `add_summary(summary, metadata)` - Add a single document
- `add_summaries(batch_summary, batch_metadata)` - Add multiple documents
- `aadd_summary(summary, metadata)` - Async add document

#### Retrieval Operations
- `search(search_query, top_k)` - Similarity search
- `query(query, params)` - Raw query execution
- `filter_chunks(...)` - Time-based and metadata filtering
- `as_retriever(search_kwargs)` - Get LangChain retriever

#### Data Management
- `drop_data(expr)` - Delete filtered data
- `drop_collection()` - Drop and recreate collection
- `reset(state)` - Reset collection state
- `update_tool(config, tools)` - Update configuration

#### Async Operations
- `aget_text_data(start_batch_index, end_batch_index, uuid)` - Get text data
- `aget_max_batch_index(uuid)` - Get max batch index

## Usage Examples

### Basic Vector Storage

```python
from vss_ctx_rag.tools.storage.qdrant_db import QdrantDBTool, QdrantDBConfig

# Create configuration
config = QdrantDBConfig(
    params={
        "host": "localhost",
        "port": "6333",
        "collection_name": "my_collection"
    }
)

# Initialize tool
tool = QdrantDBTool(config=config, tools={"embedding": embedding_tool})

# Add documents
tool.add_summary(
    summary="This is a sample document",
    metadata={
        "source": "example.pdf",
        "batch_i": 0,
        "uuid": "test-uuid-123"
    }
)

# Search
results = tool.search("sample query", top_k=5)
```

### Filtering by Metadata

```python
# Filter chunks by time range
chunks = tool.filter_chunks(
    min_start_time=1000.0,
    max_start_time=2000.0,
    uuid="test-uuid-123"
)

# Get text data for batch range
data = await tool.aget_text_data(
    start_batch_index=0,
    end_batch_index=10,
    uuid="test-uuid-123"
)
```

### Using as LangChain Retriever

```python
# Get retriever with custom search parameters
retriever = tool.as_retriever(
    search_kwargs={
        "k": 10,
        "filter": {
            "must": [
                {"key": "source", "match": {"value": "example.pdf"}}
            ]
        }
    }
)

# Use in LangChain pipeline
results = retriever.get_relevant_documents("query text")
```

## Comparison with Other Vector Databases

| Feature | Qdrant | Milvus | Elasticsearch |
|---------|--------|--------|---------------|
| **Performance** | High | Very High | High |
| **Ease of Setup** | Easy | Moderate | Moderate |
| **Cloud Support** | Yes (Qdrant Cloud) | Yes (Zilliz) | Yes (Elastic Cloud) |
| **gRPC Support** | Yes | Yes | No |
| **Filtering** | JSON-based | Expression-based | Query DSL |
| **Metadata Types** | All JSON types | Limited types | All JSON types |
| **HNSW Index** | Yes | Yes | Yes |

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to Qdrant server

**Solution**:
1. Verify Qdrant is running: `curl http://localhost:6333/health`
2. Check host/port configuration
3. Ensure firewall allows connections on port 6333/6334

### Collection Not Found

**Problem**: Collection doesn't exist

**Solution**:
Qdrant automatically creates collections on first insert. If you see this error:
1. Check collection name is valid (no special characters)
2. Ensure the embedding dimension matches your model
3. Try creating collection manually via Qdrant API

### Performance Issues

**Problem**: Slow search operations

**Solution**:
1. Enable gRPC: Set `prefer_grpc: true` in config
2. Increase `batch_size` for bulk operations
3. Tune HNSW parameters (requires custom collection creation)
4. Use Qdrant's quantization features for large datasets

### Memory Issues

**Problem**: High memory usage

**Solution**:
1. Enable disk-backed storage in Qdrant config
2. Use quantization for vectors
3. Reduce batch sizes during ingestion
4. Consider using Qdrant Cloud for large datasets

## Advanced Configuration

### Custom Collection Creation

For advanced use cases, you can create collections with custom parameters:

```python
from qdrant_client.models import Distance, VectorParams

client = QdrantClient(host="localhost", port=6333)

client.create_collection(
    collection_name="custom_collection",
    vectors_config=VectorParams(
        size=768,  # Embedding dimension
        distance=Distance.COSINE,
        on_disk=True  # Enable disk storage
    )
)
```

### Using Multiple Collections

The tool supports dynamic collection switching:

```python
# Update to use a different collection
config.params.user_specified_collection_name = "collection_2"
tool.update_tool(config, tools)

# Add data to new collection
tool.add_summary(summary, metadata)

# Reset back to default
config.params.user_specified_collection_name = None
tool.update_tool(config, tools)
```

## Migration from Milvus

To migrate from Milvus to Qdrant:

1. **Export data from Milvus**: Use `aget_text_data()` to retrieve all documents
2. **Update configuration**: Change `type: milvus` to `type: qdrant`
3. **Import data to Qdrant**: Use `add_summaries()` to bulk import
4. **Verify**: Run test queries to ensure data integrity

Example migration script:

```python
# Export from Milvus
milvus_tool = MilvusDBTool(config=milvus_config)
data = await milvus_tool.aget_text_data(0, -1, "")

# Import to Qdrant
qdrant_tool = QdrantDBTool(config=qdrant_config)
summaries = [d['content'] for d in data]
metadata = [{k: v for k, v in d.items() if k != 'content'} for d in data]
qdrant_tool.add_summaries(summaries, metadata)
```

## Best Practices

1. **Collection Names**: Use descriptive, lowercase names with underscores
2. **Batch Operations**: Use `add_summaries()` for bulk inserts (better performance)
3. **Metadata Structure**: Keep metadata consistent across documents
4. **Error Handling**: Always wrap operations in try-except blocks
5. **Connection Pooling**: Reuse tool instances instead of creating new ones
6. **Monitoring**: Use Qdrant's web UI at `http://localhost:6333/dashboard`

## API Reference

For detailed API documentation, see:
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [LangChain Qdrant Integration](https://python.langchain.com/docs/integrations/vectorstores/qdrant)
- [NVIDIA Context Aware RAG Documentation](https://nvidia.github.io/context-aware-rag/)

## Support

For issues specific to Qdrant integration:
1. Check Qdrant logs: `docker logs <container-id>`
2. Enable debug logging: Set `logger.setLevel(logging.DEBUG)`
3. Review Qdrant metrics in the dashboard
4. Open an issue on the GitHub repository

## License

This integration is licensed under Apache 2.0, same as the main project.
