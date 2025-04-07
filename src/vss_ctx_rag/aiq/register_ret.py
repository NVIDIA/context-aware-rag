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

from aiq.builder.builder import Builder
from aiq.builder.function_info import FunctionInfo
from aiq.cli.register_workflow import register_function
from aiq.builder.framework_enum import LLMFrameworkEnum
from vss_ctx_rag.context_manager import ContextManager
import os
import time
from vss_ctx_rag.aiq.utils import (
    create_vss_ctx_rag_config,
    RequestInfo,
    aiq_to_vss_config,
    update_request_info,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger

RetrievalToolConfig = create_vss_ctx_rag_config("vss_ctx_rag_retrieval")


@register_function(config_type=RetrievalToolConfig)
async def vss_ctx_rag_retrieval(config, builder: Builder):
    embedder = await builder.get_embedder(
        config.embedding_model_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    logger.info(f"llm: {llm.__dict__}")
    embedder = await builder.get_embedder(
        config.embedding_model_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    llm = await builder.get_llm(
        config.llm_name, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )
    vss_ctx_rag_config = aiq_to_vss_config(config, llm.__dict__, embedder.__dict__)

    vss_ctx_rag_config["api_key"] = os.environ["NVIDIA_API_KEY"]
    logger.debug(f"vss_ctx_rag_config: {vss_ctx_rag_config}")
    req_info = RequestInfo()
    update_request_info(config, req_info)
    ctx_mgr = ContextManager(config=vss_ctx_rag_config, req_info=req_info)
    time.sleep(5)

    async def _call_wrapper(text: str) -> str:
        state = {
            "chat": {
                "question": text,
                "is_live": False,
                "is_last": False,
            }
        }
        res = ctx_mgr.call(state)
        return res["chat"]["response"]

    # Create a Generic AI-Q tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_call_wrapper)

    ctx_mgr.process.stop()
