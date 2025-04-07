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

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List


class LLMConfig(BaseModel):
    model: str
    base_url: Optional[str] = None
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)


class EmbeddingConfig(BaseModel):
    model: str
    base_url: str


class SummarizationParams(BaseModel):
    batch_size: int = Field(default=6, ge=1)
    batch_max_concurrency: int = Field(default=20, ge=1)
    top_k: Optional[int] = Field(default=5, ge=1)


class Prompts(BaseModel):
    caption: str
    caption_summarization: str
    summary_aggregation: str


class SummarizationConfig(BaseModel):
    enable: Optional[bool] = Field(default=True)
    # This field specifies the summarization method to use
    # Currently only "batch" method is supported, enforced by regex pattern
    method: str = Field(default="batch", pattern="^batch$")
    llm: LLMConfig
    embedding: EmbeddingConfig
    params: SummarizationParams
    prompts: Prompts


class ChatParams(BaseModel):
    batch_size: int = Field(default=1, ge=1)
    top_k: Optional[int] = Field(default=5, ge=1)
    chat_history: Optional[bool] = Field(default=True)
    multi_channel: Optional[bool] = Field(default=False)
    uuid: Optional[str] = Field(default="default")


class RerankerConfig(BaseModel):
    model: str
    base_url: str


class ChatConfig(BaseModel):
    rag: str = Field(default="vector-rag")
    params: ChatParams
    llm: LLMConfig
    embedding: EmbeddingConfig
    reranker: RerankerConfig


class NotificationConfig(BaseModel):
    enable: bool
    endpoint: str
    llm: LLMConfig


class ContextManagerConfig(BaseModel):
    summarization: SummarizationConfig
    chat: ChatConfig
    notification: NotificationConfig
    milvus_db_host: str = Field(default="localhost")
    milvus_db_port: str = Field(default="19530")

    @field_validator("chat")
    def validate_chat(cls, v):
        if v.rag not in ["vector-rag", "graph-rag"]:
            raise ValueError("Invalid rag value")
        return v


class Event(BaseModel):
    event_id: str
    event_list: List[str]


class Alert(BaseModel):
    events: List[Event]


class AlertConfig(BaseModel):
    notification: Alert
