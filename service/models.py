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
from typing import Optional, List
from copy import deepcopy
import os
from vss_ctx_rag.utils.ctx_rag_logger import logger

MAX_DEPTH = 20


def resolve_env_vars(config: dict, depth: int = 0) -> dict:
    """
    Recursively resolve environment variables in config dictionary.
    Environment variables should be in the format ${ENV_VAR_NAME}.
    """
    resolved_config = {}

    if depth == MAX_DEPTH:
        logger.warning("Max depth reached, returning deepcopy of resolved config")
        return deepcopy(config)

    for key, value in config.items():
        if isinstance(value, dict):
            resolved_config[key] = resolve_env_vars(value, depth + 1)
        elif isinstance(value, list):
            resolved_list = []
            for item in value:
                if isinstance(item, dict):
                    resolved_list.append(resolve_env_vars(item, depth + 1))
                elif (
                    isinstance(item, str)
                    and item.startswith("${")
                    and item.endswith("}")
                ):
                    env_var_name = item[2:-1]
                    env_value = os.environ.get(env_var_name)
                    if env_value is not None:
                        resolved_list.append(env_value)
                    else:
                        logger.warning(
                            f"Environment variable '{env_var_name}' not found, keeping original value"
                        )
                        resolved_list.append(item)
                else:
                    resolved_list.append(item)
            resolved_config[key] = resolved_list
        elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
            env_var_name = value[2:-1]
            env_value = os.environ.get(env_var_name)
            if env_value is not None:
                resolved_config[key] = env_value
            else:
                logger.warning(
                    f"Environment variable '{env_var_name}' not found, keeping original value"
                )
                resolved_config[key] = value
        else:
            resolved_config[key] = value
    return deepcopy(resolved_config)


class ConfigModel(BaseModel):
    config_path: Optional[str] = None
    context_config: Optional[dict] = None
    uuid: str

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


class AddModel(BaseModel):
    document: str
    doc_index: int
    doc_metadata: dict = {}
    uuid: str


class CallModel(BaseModel):
    state: dict
    uuid: str


class ResetModel(BaseModel):
    state: dict
    uuid: str


class DCFileModel(BaseModel):
    dc_file_path: str
    uuid: str


class SummaryModel(BaseModel):
    class SummaryState(BaseModel):
        start_index: int = 0
        end_index: int = -1  # -1 means "until end"

        @model_validator(mode="before")
        def validate_indices(cls, values: dict) -> dict:
            start_index = values.get("start_index", 0)
            end_index = values.get("end_index", -1)

            if start_index < 0:
                raise ValueError("start_index must be non-negative")

            if end_index != -1 and end_index < start_index:
                raise ValueError(
                    "end_index must be -1 or greater than or equal to start_index"
                )

            return values

    summarization: SummaryState
    uuid: str


class CompleteIngestionModel(BaseModel):
    uuid: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatModel(BaseModel):
    model: str
    base_url: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    max_tokens: Optional[int] = 4096
    uuid: str
