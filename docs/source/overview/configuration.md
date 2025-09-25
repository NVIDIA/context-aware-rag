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

CA-RAG can be configured using a config file.

Summarization example:

```yaml
summarization:
   enable: true
   method: "batch"
   llm:
      model: "nvidia/llama-3.3-nemotron-super-49b-v1"
      base_url: "http://localhost:8000/v1"
      max_tokens: 2048
      temperature: 0.2
      top_p: 0.7
   embedding:
      model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
      base_url: "http://localhost:8000/v1"
   params:
      batch_size: 5
   prompts:
      caption: "Write a concise and clear dense caption for the provided warehouse video, focusing on irregular or hazardous events such as boxes falling, workers not wearing PPE, workers falling, workers taking photographs, workers chitchatting, forklift stuck, etc. Start and end each sentence with a time stamp."
      caption_summarization: "You should summarize the following events of a warehouse in the format start_time:end_time:caption. For start_time and end_time use . to separate seconds, minutes, hours. If during a time segment only regular activities happen, then ignore them, else note any irregular activities in detail. The output should be bullet points in the format start_time:end_time: detailed_event_description. Don't return anything else except the bullet points."
      summary_aggregation: "You are a warehouse monitoring system. Given the caption in the form start_time:end_time: caption, Aggregate the following captions in the format start_time:end_time:event_description. If the event_description is the same as another event_description, aggregate the captions in the format start_time1:end_time1,...,start_timek:end_timek:event_description. If any two adjacent end_time1 and start_time2 is within a few tenths of a second, merge the captions in the format start_time1:end_time2. The output should only contain bullet points.  Cluster the output into Unsafe Behavior, Operational Inefficiencies, Potential Equipment Damage and Unauthorized Personnel"
```

The `summarization` section outlines the system's summarization capabilities. It supports batch processing using a specified LLM model and embedding model. Prompts can be customized for various use cases. The default prompts are tailored to generate captions and summaries for warehouse videos, emphasizing irregular events.

Attributes:

- **`enable`**: Enables the summarization. Default: true
- **`method`**: Can be `batch` or `refine`. Refer to summarization for more details about each method. Default: `batch`
- **`batch_size`**: For method `batch`, this is the batch size used for combining a batch summary. Default: 5
- **`prompts`**: Users can update the prompts to change the behavior of CA RAG.
   - **`caption`**: This prompt is used in VSS only and are not used in Context Aware RAG and can be safely ignored if only
   using CA RAG.

   - **`caption_summarization`**: This prompt generates a summary from a batch of captions. The `batch_size` parameter specifies the number of captions to be combined.

   - **`summary_aggregation`**: After generating all batch summaries, this prompt is used to combine them into the final summary.

Q&A Example:

```yaml
chat:
   rag: graph-rag # graph-rag or vector-rag
   params:
      batch_size: 1
      top_k: 5
      multi_channel: true # Enable/Disable multi-stream processing.
      chat_history: false # Enable/Disable chat history.
   llm:
      model: "gpt-4o"
      max_tokens: 2048
      temperature: 0.2
      top_p: 0.7
   embedding:
      model: "nvidia/llama-3.2-nv-embedqa-1b-v2"
      base_url: "http://localhost:8000/v1"
   reranker:
      model: "nvidia/llama-3.2-nv-rerankqa-1b-v2"
      base_url: "http://localhost:8000/v1"
```

Attributes:

- **`rag`**: Can be `graph-rag` or `vector-rag`. Refer to qna for more details for each option. Default `graph-rag`
- **`batch_size`**: Number of vlm captions to be batched together for creating graph.
- **`top_k`**: top-k most relevant retrieval results for QnA.
- **`multi_channel`**: Enable/Disable multi-stream processing. Default `false`. Only supported for `graph-rag`.
- **`chat_history`**: Enable/Disable chat history. Default `true`. Only supported for `graph-rag`.

Alerts example:

```yaml
notification:
  enable: true
  endpoint: "http://127.0.0.1:60000/via-alert-callback"
  llm:
    model: "nvidia/llama-3.3-nemotron-super-49b-v1"
    base_url: "http://<IP ADDRESS>:<PORT>/v1/"
    max_tokens: 2048
    temperature: 0.2
    top_p: 0.7
```
