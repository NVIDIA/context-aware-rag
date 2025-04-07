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

from pydantic import BaseModel, model_validator
from typing import Optional
from vss_ctx_rag.context_manager.context_manager_models import ContextManagerConfig
from vss_ctx_rag.utils.ctx_rag_logger import logger


class BaseConfigRequest(BaseModel):
    config_path: Optional[str] = None
    context_config: Optional[dict] = None

    @model_validator(mode="before")
    def validate_context_config(cls, values: dict) -> dict:
        context_config = values.get("context_config")
        if context_config is not None:
            if isinstance(context_config, dict):
                try:
                    _ = ContextManagerConfig(**context_config)
                except Exception as e:
                    raise ValueError(f"Invalid context_config: {e}")
        return values

    @model_validator(mode="before")
    def check_exclusivity(cls, values: dict) -> dict:
        config_path = values.get("config_path")
        context_config = values.get("context_config")
        if config_path is not None and context_config is not None:
            raise ValueError(
                "Must provide exactly one of config_path or context_config"
            )
        if config_path is None and context_config is None:
            logger.info("Using default config path /app/config/config.yaml")
        return values


class AddRequest(BaseModel):
    document: str
    doc_index: int
    doc_metadata: dict = {}


class RequestInfo(BaseModel):
    summarize: bool = True
    enable_chat: bool = True
    is_live: bool = False
    uuid: str = "0"
    stream_id: str = ""
    caption_summarization_prompt: str = "Return input as is"
    summary_aggregation_prompt: str = "Return input as is"
    chunk_size: int = 0
    summary_duration: int = 0
    summarize_top_p: Optional[float] = None
    summarize_temperature: Optional[float] = None
    summarize_max_tokens: Optional[int] = None
    chat_top_p: Optional[float] = None
    chat_temperature: Optional[float] = None
    chat_max_tokens: Optional[int] = None
    notification_top_p: Optional[float] = None
    notification_temperature: Optional[float] = None
    notification_max_tokens: Optional[int] = None
    rag_type: str = "vector-rag"


class CallRequest(BaseModel):
    state: dict


class ResetRequest(BaseModel):
    state: dict


class DCFileRequest(BaseModel):
    dc_file_path: str


class InitRequest(BaseConfigRequest):
    uuid: Optional[str] = "0"


class UpdateConfigRequest(BaseConfigRequest):
    request_info: Optional[RequestInfo] = None
