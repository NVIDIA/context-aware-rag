# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DEFAULT_GRAPH_RAG_BATCH_SIZE = 1
DEFAULT_MULTI_CHANNEL = False
DEFAULT_CHAT_HISTORY = False
DEFAULT_BATCH_SUMMARIZATION_BATCH_SIZE = 5
DEFAULT_RAG_TOP_K = 5
DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_CONFIG_PATH = "config/config.yaml"
DEFAULT_LLM_PARAMS = {
    "model": "meta/llama3-70b-instruct",
    "max_tokens": 1024,
    "top_p": 1,
    "temperature": 0.4,
    "seed": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
}
DEFAULT_SUMM_RECURSION_LIMIT = 8
DEFAULT_SUMM_TIMEOUT_SEC = 120
LLM_TOOL_NAME = "llm"
DEFAULT_EMBEDDING_DIMENSION = 2048
DEFAULT_EMBEDDING_PARALLEL_COUNT = 1000
DEFAULT_CHUNKS = 3
DEFAULT_FRAMES_PER_CHUNK = 3
MAX_FRAMES = 10
DEFAULT_TRAVERSAL_STRATEGY = "chunk"
GNN_TRAVERSAL_STRATEGY = "gnn_chunk"
DEFAULT_NUM_FRAMES_PER_CHUNK = 30
DEFAULT_CONTEXT_MANAGER_CALL_TIMEOUT = 3600

## POLLING CONFIGURATION
DEFAULT_POLL_INTERVAL = 1.0
DEFAULT_POLL_TIMEOUT = 120.0

## LOAD BALANCING
DEFAULT_CONCURRENT_EMBEDDING_LIMIT = 250
DEFAULT_CONCURRENT_DOC_PROCESSING_LIMIT = 100
