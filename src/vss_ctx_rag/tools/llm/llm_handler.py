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

import os

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_core.runnables.utils import ConfigurableField
from vss_ctx_rag.base import Tool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.globals import DEFAULT_LLM_BASE_URL
from vss_ctx_rag.utils.utils import is_openai_model
from langchain_core.runnables.base import Runnable
from langchain_nvidia_ai_endpoints import register_model, Model, ChatNVIDIA


class LLMTool(Tool, Runnable):
    """A Tool class wrapper for LLMs.

    Returns:
        LLMTool: A Tool that wraps an LLM.
    """

    llm: BaseChatModel

    def __init__(self, llm, name="llm_tool") -> None:
        Tool.__init__(self, name)
        self.llm = llm

    def __getattr__(self, attr):
        return getattr(self.llm, attr)

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self.llm.stream(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.llm.batch(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.llm.ainvoke(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self.llm.astream(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.llm.abatch(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.name


class ChatOpenAITool(LLMTool):
    def __init__(
        self, model=None, api_key=None, base_url=DEFAULT_LLM_BASE_URL, **llm_params
    ) -> None:
        if model and is_openai_model(model):
            base_url = ""
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        elif model and "llama-3.1-70b-instruct" in model and "nvcf" in base_url:
            register_model(
                Model(
                    id=model, model_type="chat", client="ChatNVIDIA", endpoint=base_url
                )
            )
            super().__init__(
                ChatNVIDIA(
                    model=model, api_key=api_key, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        else:
            super().__init__(
                llm=ChatOpenAI(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                ).configurable_fields(
                    top_p=ConfigurableField(id="top_p"),
                    temperature=ConfigurableField(id="temperature"),
                    max_tokens=ConfigurableField(id="max_tokens"),
                )
            )
        try:
            if os.getenv("CA_RAG_ENABLE_WARMUP", "false").lower() == "true":
                self.warmup(model)
        except Exception as e:
            logger.error(f"Error warming up LLM: {e}")
            raise

    def warmup(self, model_name):
        try:
            logger.info(f"Warming up LLM {model_name}")
            logger.info(str(self.invoke("Hello, world!")))
        except Exception as e:
            logger.error(f"Error warming up LLM {model_name}: {e}")
            raise

    def update(self, top_p=None, temperature=None, max_tokens=None):
        configurable_dict = {}
        if top_p is not None:
            configurable_dict["top_p"] = top_p
        if temperature is not None:
            configurable_dict["temperature"] = temperature
        if max_tokens is not None:
            configurable_dict["max_tokens"] = max_tokens
        logger.debug(f"Updating LLM with config:{configurable_dict}")
        self.llm = self.llm.with_config(configurable=configurable_dict)
