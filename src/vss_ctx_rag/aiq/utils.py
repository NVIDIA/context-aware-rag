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

from typing import Dict, Optional
from aiq.data_models.function import FunctionBaseConfig
from aiq.data_models.component_ref import EmbedderRef, LLMRef

##TODO: Remove this class dependency.
## Everything should be defined in the aiq config file


class RequestInfo:
    """Configuration container for request-specific parameters.

    This class holds configuration parameters that can be customized per request,
    including summarization settings, chat settings, and LLM parameters.
    """

    def __init__(self):
        self.summarize = True
        self.enable_chat = True
        self.is_live = False
        self.uuid = "1"
        self.caption_summarization_prompt = "Return input as is"
        self.summary_aggregation_prompt = "Return input as is"
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
        self.rag_type = "vector-rag"


def aiq_to_vss_config(
    config: Dict,
    llm_dict: Dict,
    embedder_dict: Dict,
) -> dict:
    """
    Convert an instance of Config into a vss_ctx_rag config dict

    """
    return {
        "summarization": {
            "enable": config.summarize,
            "method": "batch",
            "llm": {
                "model": llm_dict.get("model", "meta/llama-3.1-70b-instruct"),
                "base_url": llm_dict.get(
                    "base_url", "https://integrate.api.nvidia.com/v1"
                ),
                "temperature": llm_dict.get("temperature", 0.5),
                "top_p": llm_dict.get("top_p", 0.3),
                "max_tokens": llm_dict.get("max_tokens", 1024),
            },
            "embedding": {
                "model": embedder_dict.get(
                    "model", "nvidia/llama-3.2-nv-embedqa-1b-v2"
                ),
                "base_url": embedder_dict.get("base_url", ""),
            },
            "params": {
                "batch_size": config.summ_batch_size,
                "batch_max_concurrency": config.summ_batch_max_concurrency,
            },
            "prompts": {
                "caption": "Describe the following text in detail.",
                "caption_summarization": "Summarize the following text:",
                "summary_aggregation": "Summarize the following text:",
            },
        },
        "chat": {
            "rag": config.rag_type,  # e.g. "vector-rag" or "graph-rag"
            "params": {
                "batch_size": config.chat_batch_size,
            },
            "llm": {
                "model": llm_dict.get("model", "meta/llama-3.1-70b-instruct"),
                "base_url": llm_dict.get(
                    "base_url", "https://integrate.api.nvidia.com/v1"
                ),
                "temperature": llm_dict.get("temperature", 0.2),
                "top_p": llm_dict.get("top_p", 0.4),
                "max_tokens": llm_dict.get("max_tokens", 2000),
            },
            "embedding": {
                "model": embedder_dict.get(
                    "model", "nvidia/llama-3.2-nv-embedqa-1b-v2"
                ),
                "base_url": embedder_dict.get(
                    "base_url", "https://integrate.api.nvidia.com/v1"
                ),
            },
            "reranker": {
                "model": config.rerank_model_name,
                "base_url": config.rerank_model_url,
            },
        },
        "milvus_db_host": config.vector_db_host,
        "milvus_db_port": config.vector_db_port,
    }


def create_vss_ctx_rag_config(name: str):
    class VssCtxRagToolConfig(FunctionBaseConfig, name=name):
        llm_name: LLMRef

        vector_db_host: str = "localhost"
        vector_db_port: str = "19530"

        graph_db_uri: str = "bolt://localhost:7687"
        graph_db_user: str = "neo4j"
        graph_db_password: str = "passneo4j"

        embedding_model_name: EmbedderRef

        rerank_model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        rerank_model_url: str = "https://integrate.api.nvidia.com/v1"
        rag_type: str = "vector-rag"  # or "graph-rag"
        chat_batch_size: int = 1
        summ_batch_size: int = 5
        summ_batch_max_concurrency: int = 20

        summarize: Optional[bool] = True
        enable_chat: Optional[bool] = True
        is_live: Optional[bool] = False
        uuid: Optional[str] = "1"

    return VssCtxRagToolConfig


def update_request_info(config, req_info):
    """
    Updates RequestInfo object with values from VssCtxRagToolConfig

    Args:
        config: VssCtxRagToolConfig instance
        req_info: RequestInfo instance to be updated
    """
    for field in vars(req_info).keys():
        if hasattr(config, field):
            setattr(req_info, field, getattr(config, field))
