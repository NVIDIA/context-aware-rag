# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import ClassVar, Dict, List, Optional, Any
from pydantic import BaseModel, Field
from vss_ctx_rag.models.function_models import FunctionModel
from vss_ctx_rag.utils.globals import (
    DEFAULT_GRAPH_RAG_BATCH_SIZE,
    DEFAULT_RAG_TOP_K,
    DEFAULT_CHAT_HISTORY,
    DEFAULT_MULTI_CHANNEL,
)


class IngestionConfig(FunctionModel):
    ALLOWED_FUNCTION_TYPES: ClassVar[Dict[str, List[str]]] = {
        "graph_ingestion": ["graph_ingestion"],
        "vector_ingestion": ["vector_ingestion"],
        "foundation_ingestion": ["foundation_ingestion"],
    }

    FUNCTION_TYPE_CONSTRAINTS: ClassVar[Dict[str, Any]] = {
        "max_count": 1,
        "mutually_exclusive_groups": [
            ["graph_ingestion", "vector_ingestion", "foundation_ingestion"]
        ],
    }

    class IngestionParams(BaseModel):
        batch_size: Optional[int] = Field(default=DEFAULT_GRAPH_RAG_BATCH_SIZE, ge=1)
        multi_channel: Optional[bool] = Field(default=DEFAULT_MULTI_CHANNEL)
        uuid: Optional[str] = Field(default="default")

    params: IngestionParams


class RetrieverConfig(FunctionModel):
    class RetrieverParams(BaseModel):
        top_k: Optional[int] = Field(default=DEFAULT_RAG_TOP_K, ge=1)
        chat_history: Optional[bool] = Field(default=DEFAULT_CHAT_HISTORY)
        multi_channel: Optional[bool] = Field(default=DEFAULT_MULTI_CHANNEL)
        uuid: Optional[str] = Field(default="default")
        image: Optional[bool] = Field(default=False)
        citations: Optional[dict] = Field(default={})
        prompt_config_path: Optional[str] = Field(default=None)
        max_concurrency: Optional[int] = Field(
            default=20,
            ge=1,
            description="Maximum number of concurrent retrieval requests to process",
        )

    params: RetrieverParams
