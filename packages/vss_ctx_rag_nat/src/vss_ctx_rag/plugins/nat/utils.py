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

import os
from typing import Dict, Optional

from nat.data_models.function import FunctionBaseConfig
from nat.data_models.component_ref import LLMRef, EmbedderRef

from vss_ctx_rag.utils.ctx_rag_logger import logger

##TODO: Remove this class dependency.
## Everything should be defined in the nat config file


def nat_to_vss_config(
    config: Dict,
    llm_dict: Dict,
    embedder_dict: Dict,
) -> dict:
    """
    Convert an instance of Config into a vss_ctx_rag config dict

    """
    logger.debug(f"LLM Dict: {llm_dict}")
    logger.debug(f"Embedder Dict: {embedder_dict}")

    ret_config = {
        "tools": {
            "nvidia_llm": {
                "type": "llm",
                "params": {
                    "model": llm_dict.get("model")
                    or llm_dict.get("model_name", "meta/llama-3.1-70b-instruct"),
                    "base_url": llm_dict.get("openai_api_base")
                    or llm_dict.get("base_url", "https://integrate.api.nvidia.com/v1"),
                    "temperature": llm_dict.get("temperature", 0.5),
                    "top_p": llm_dict.get("top_p", 0.7),
                    "max_tokens": llm_dict.get("max_tokens", 4096),
                    "api_key": os.getenv("NVIDIA_API_KEY")
                    if llm_dict.get("openai_api_key") is None
                    else os.getenv("OPENAI_API_KEY"),
                },
            },
            "nvidia_embedding": {
                "type": "embedding",
                "params": {
                    "model": embedder_dict.get("model")
                    or embedder_dict.get(
                        "model_name", "nvidia/llama-3.2-nv-embedqa-1b-v2"
                    ),
                    "base_url": embedder_dict.get(
                        "base_url", "https://integrate.api.nvidia.com/v1"
                    ),
                    "api_key": os.getenv("NVIDIA_API_KEY"),
                },
            },
            "nvidia_reranker": {
                "type": "reranker",
                "params": {
                    "model": config.rerank_model_name,
                    "base_url": config.rerank_model_url,
                    "api_key": os.getenv("NVIDIA_API_KEY"),
                },
            },
            "db": {
                "type": config.db_type,
                "params": {
                    "host": config.db_host,
                    "port": config.db_port,
                    "username": getattr(config, "db_user", "neo4j"),
                    "password": getattr(config, "db_password", "passneo4j"),
                },
                "tools": {
                    "embedding": "nvidia_embedding",
                },
            },
        },
        "functions": {
            "summarization": {
                "type": "batch_summarization",
                "params": {
                    "batch_size": config.summ_batch_size,
                    "batch_max_concurrency": config.summ_batch_max_concurrency,
                    "prompts": {
                        "caption": "Describe the following text in detail.",
                        "caption_summarization": "Summarize the following text:",
                        "summary_aggregation": "Summarize the following text:",
                    },
                },
                "tools": {
                    "llm": "nvidia_llm",
                    "db": "db",
                },
            },
            "ingestion_function": {
                "type": f"{config.ingestion_type}",
                "params": {
                    "batch_size": 1,
                },
                "tools": {
                    "llm": "nvidia_llm",
                    "db": "db",
                },
            },
            "retriever_function": {
                "type": f"{config.retrieval_type}",
                "params": {
                    "batch_size": config.chat_batch_size,
                    **config.retrieval_extra_args,
                },
                "tools": {
                    "llm": "nvidia_llm",
                    "db": "db",
                    "reranker": "nvidia_reranker",
                },
            },
        },
    }

    functions_to_add = []
    if config.summarize:
        functions_to_add.append("summarization")
    if config.enable_chat:
        functions_to_add.append("ingestion_function")
        functions_to_add.append("retriever_function")
    ret_config["context_manager"] = {"functions": functions_to_add}

    return ret_config


def create_vss_ctx_rag_config(name: str):
    class VssCtxRagToolConfig(FunctionBaseConfig, name=name):
        # Basic configuration
        summarize: Optional[bool] = True
        enable_chat: Optional[bool] = True
        ingestion_type: Optional[str] = "graph_ingestion"
        retrieval_type: Optional[str] = "graph_retrieval"
        retrieval_extra_args: Optional[dict] = {}

        uuid: Optional[str] = "1"

        # Batch configuration
        chat_batch_size: int = 1
        summ_batch_size: int = 5
        summ_batch_max_concurrency: int = 20

        db_type: str = "neo4j"
        db_host: str = "localhost"
        db_port: str = "7687"
        db_user: str
        db_password: str

        # Model configuration
        llm_name: LLMRef
        embedding_model_name: EmbedderRef

        rerank_model_name: str = "nvidia/llama-3.2-nv-rerankqa-1b-v2"
        rerank_model_url: str = "https://integrate.api.nvidia.com/v1"

    return VssCtxRagToolConfig
