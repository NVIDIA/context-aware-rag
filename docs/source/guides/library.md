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

# Using the Python Library

## Setting up config file

First create a config file to set the llms, prompts, and parameters.
Here is an example of the config file:

``` yaml
summarization:
  enable: true
  method: "batch"
  llm:
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    max_tokens: 2048
    temperature: 0.2
    top_p: 0.7
  embedding:
    model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: https://integrate.api.nvidia.com/v1
  params:
    batch_size: 5
    batch_max_concurrency: 20
  prompts:
    caption: "Write a concise and clear dense caption for the provided warehouse video, focusing on irregular or hazardous events such as boxes falling, workers not wearing PPE, workers falling, workers taking photographs, workers chitchatting, forklift stuck, etc. Start and end each sentence with a time stamp."
    caption_summarization: "You should summarize the following events of a warehouse in the format start_time:end_time:caption. For start_time and end_time use . to seperate seconds, minutes, hours. If during a time segment only regular activities happen, then ignore them, else note any irregular activities in detail. The output should be bullet points in the format start_time:end_time: detailed_event_description. Don't return anything else except the bullet points."
    summary_aggregation: "You are a warehouse monitoring system. Given the caption in the form start_time:end_time: caption, Aggregate the following captions in the format start_time:end_time:event_description. If the event_description is the same as another event_description, aggregate the captions in the format start_time1:end_time1,...,start_timek:end_timek:event_description. If any two adjacent end_time1 and start_time2 is within a few tenths of a second, merge the captions in the format start_time1:end_time2. The output should only contain bullet points.  Cluster the output into Unsafe Behavior, Operational Inefficiencies, Potential Equipment Damage and Unauthorized Personnel"

chat:
  rag: vector-rag # graph-rag or vector-rag
  params:
    batch_size: 1
  llm:
    model: meta/llama-3.1-70b-instruct
    base_url: https://integrate.api.nvidia.com/v1
    max_tokens: 2048
    temperature: 0.5
  embedding:
    model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
    base_url: https://integrate.api.nvidia.com/v1
  reranker:
    model: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
    base_url: https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking
```

### ENV Setup

Now setup the environment variables depending on the type of RAG.

### Vector-RAG

``` bash
export MILVUS_HOST=<MILVUS_HOST_IP>
export MILVUS_PORT=<MILVUS_DB_PORT>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

### Graph-RAG

``` bash
export GRAPH_DB_URI=<GRAPH_DB_URI>
export GRAPH_DB_USERNAME=<GRAPH_DB_USERNAME>
export GRAPH_DB_PASSWORD=<GRAPH_DB_PASSWORD>
export NVIDIA_API_KEY=<NVIDIA_API_KEY>
```

## Context Manager Setup

Now setup the context manager. Context manager is used to both add
documents and retrieve documents.

``` python
with open("config/config.yaml", mode="r", encoding="utf8") as c:
        config = yaml.safe_load(c)
    ### IF USING VECTOR-RAG
    config["milvus_db_host"] = os.environ["MILVUS_HOST"]
    config["milvus_db_port"] = os.environ["MILVUS_PORT"]

    config["api_key"] = os.environ["NVIDIA_API_KEY"]

    DOC_META = {
       "streamId": "",
       "chunkIdx": -1,
       "file": "",
       "pts_offset_ns": 0,
       "start_pts": 0,
       "end_pts": 0,
       "start_ntp": "1970-01-01T00:01:00.000Z",
       "end_ntp": "1970-01-01T00:02:00.000Z",
       "start_ntp_float": 0.0,
       "end_ntp_float": 0.0,
       "is_first": False,
       "is_last": False,
       "uuid": "",
       "cv_meta": "[]",
   }

    class RequestInfo:
        def __init__(self):
            self.summarize = True
            self.enable_chat = False
            self.is_live = False
            self.uuid = "test_context_manager"
            self.caption_summarization_prompt = (
                "Return the input in it's entirety as is without any changes"
            )
            self.summary_aggregation_prompt = (
                "Combine the conversation into a single summary"
            )
            self.chunk_size = 0
            self.summary_duration = 0
            self.summarize_top_p = None
            self.summarize_temperature = None
            self.summarize_max_tokens = None
            self.chat_top_p = None
            self.chat_temperature = None
            self.chat_max_tokens = None
            self.notification_top_p = None
            self.notification_temperature = None
            self.notification_max_tokens = None
            self.rag_type = config["chat"]["rag"]

    req_info = RequestInfo()

    cm = ContextManager(config=config, process_index=random.randint(0, 1000000))
    cm.configure_update(config=config, req_info=req_info)
    ## cm doing work here
    cm.process.stop()
```

## Document Ingestion

Context manager can be used to ingest documents.

``` python
cm.add_doc("User1: I went hiking to Mission Peak", doc_meta=DOC_META) ## Add documents to the context manager
```

## Document Retrieval

To retrieve documents, use the following code as an example:

``` python
question = "Where did the user go hiking?"
result = cm.call(
    {
        "chat": {
            "question": question,
            "is_live": False,
            "is_last": False,
        }
    }
)
logger.info(f"Response {result['chat']['response']}")
```
