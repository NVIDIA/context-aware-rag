# Ingestion

This guide explains how to add documents to the Context-Aware RAG system.

## Adding Documents

Documents can be added to the system using the `/add_doc` endpoint of the Data Ingestion Service.

### Request Format

```json
{
  "document": "Your document text here",
  "doc_index": 0,
  "doc_metadata": {
    "streamId": "unique_stream_id",
    "chunkIdx": 0,
    "file": "source_file.txt",
    "is_first": true,  // Required for first document in a stream
    "is_last": false,  // Required for last document in a stream
    "uuid": "your_session_uuid"
  },
  "uuid": "your_session_uuid"
}
```

### Metadata Flags

- `is_first`: Set to `true` for the first document in a stream
- `is_last`: Set to `true` for the last document in a stream
- At least one document must have `is_first: true` and one must have `is_last: true`

### Example: Adding Multiple Documents

1. Add documents

```python
import requests
import json

base_url = "http://localhost:8001"
headers = {"Content-Type": "application/json"}

add_doc_data_list = [
    {
        "document": "User1: Hi how are you?",
        "doc_index": 0,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 0,
            "file": "chat_conversation.txt",
            "is_first": True,
            "is_last": False,
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
    {
        "document": "User2: I am good. How are you?",
        "doc_index": 1,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 1,
            "file": "chat_conversation.txt",
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
    {
        "document": "User1: I am great too. Thanks for asking",
        "doc_index": 2,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 2,
            "file": "chat_conversation.txt",
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
    {
        "document": "User2: So what did you do over the weekend?",
        "doc_index": 3,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 3,
            "file": "chat_conversation.txt",
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
    {
        "document": "User1: I went hiking to Mission Peak",
        "doc_index": 4,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 4,
            "file": "chat_conversation.txt",
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
    {
        "document": "User3: Guys there is a fire. Let us get out of here",
        "doc_index": 5,
        "doc_metadata": {
            "streamId": "stream1",
            "chunkIdx": 5,
            "file": "chat_conversation.txt",
            "is_first": False,
            "is_last": True,
            "uuid": "your_session_uuid"
        },
        "uuid": "your_session_uuid"
    },
]

# Send POST requests for each document
for add_doc_data in add_doc_data_list:
    response = requests.post(
        f"{base_url}/add_doc", headers=headers, data=json.dumps(add_doc_data)
    )
    print(response.text)

```

2. Complete ingestion
```python
import requests

url = "http://localhost:8001/complete_ingestion"
headers = {"Content-Type": "application/json"}
data = {
    "uuid": "your_session_uuid"
}

response = requests.post(url, headers=headers, json=data)
print(response.text)
```

## Best Practices

### Document Structure
- Keep documents between 100-1000 words for optimal retrieval
- Use clear, well-formatted text
- Include relevant metadata

### Document Indexing
- Use sequential indices starting from 0
- Maintain consistent indexing within a stream
- Include relevant metadata for better context

### Performance Optimization
- Batch similar documents together
- Use appropriate chunk sizes
- Monitor system resources
