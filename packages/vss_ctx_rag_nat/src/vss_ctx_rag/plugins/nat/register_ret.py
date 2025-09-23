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

import time

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.builder.framework_enum import LLMFrameworkEnum

from vss_ctx_rag.plugins.nat.utils import create_vss_ctx_rag_config, nat_to_vss_config
from vss_ctx_rag.context_manager import ContextManager
from vss_ctx_rag.utils.ctx_rag_logger import logger

RetrievalToolConfig = create_vss_ctx_rag_config("vss_ctx_rag_retrieval")


@register_function(config_type=RetrievalToolConfig)
async def vss_ctx_rag_retrieval(config, builder: Builder):
    # Generate VSS configuration directly from the config object
    llm_dict = await builder.get_llm(config.llm_name, LLMFrameworkEnum.LANGCHAIN)
    embedder_dict = await builder.get_embedder(
        config.embedding_model_name, LLMFrameworkEnum.LANGCHAIN
    )
    vss_ctx_rag_config = nat_to_vss_config(
        config, llm_dict.model_dump(), embedder_dict.model_dump()
    )
    vss_ctx_rag_config["context_manager"]["uuid"] = config.uuid
    logger.debug(f"vss_ctx_rag_config: {vss_ctx_rag_config}")
    ctx_mgr = ContextManager(config=vss_ctx_rag_config)
    time.sleep(5)

    async def _call_wrapper(input_message: str) -> str:
        state = {
            "retriever_function": {
                "question": input_message,
                "is_live": False,
                "is_last": False,
                "uuid": config.uuid,
            }
        }
        res = ctx_mgr.call(state)
        return res["retriever_function"]["response"]

    # Create a Generic AI-Q tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_call_wrapper)

    ctx_mgr.process.stop()
