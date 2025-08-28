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
from copy import deepcopy

from nat.builder.builder import Builder
from nat.builder.function_info import FunctionInfo
from nat.cli.register_workflow import register_function
from nat.builder.framework_enum import LLMFrameworkEnum

from vss_ctx_rag.plugins.nat.utils import create_vss_ctx_rag_config, nat_to_vss_config
from vss_ctx_rag.context_manager import ContextManager
from vss_ctx_rag.utils.ctx_rag_logger import logger

IngestionToolConfig = create_vss_ctx_rag_config("vss_ctx_rag_ingestion")


@register_function(config_type=IngestionToolConfig)
async def vss_ctx_rag_ingestion(config, builder: Builder):
    # Generate VSS configuration directly from the config object
    logger.debug(f"LLM Name: {config.llm_name}")
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

    current_doc_i = 0
    doc_meta_store = {}

    doc_meta = {
        "chunkIdx": -1,
        "file": "default",
        "pts_offset_ns": 0,
        "start_pts": 0,
        "end_pts": 0,
        "start_ntp": "",
        "end_ntp": "",
        "start_ntp_float": 0.0,
        "end_ntp_float": 0.0,
        "uuid": config.uuid,
        "batch_i": -1,
        "is_first": False,
        "is_last": False,
    }

    async def _call_wrapper(text: str) -> str:
        nonlocal current_doc_i, doc_meta_store

        final_doc_meta = deepcopy(doc_meta)

        final_doc_meta["chunkIdx"] = current_doc_i

        doc_meta_store[current_doc_i] = final_doc_meta

        logger.debug(f"Adding document {current_doc_i} with meta: {final_doc_meta}")
        ctx_mgr.add_doc(text, doc_i=current_doc_i, doc_meta=final_doc_meta)

        ctx_mgr.call({"ingestion_function": {"uuid": final_doc_meta["uuid"]}})

        current_doc_i += 1

        return f"Document added with doc_i={current_doc_i - 1}"

    # Create a Generic NAT tool that can be used with any supported LLM framework
    yield FunctionInfo.from_fn(_call_wrapper)

    time.sleep(5)
    ctx_mgr.process.stop()
