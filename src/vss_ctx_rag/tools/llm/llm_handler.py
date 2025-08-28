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
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.runnables.base import Runnable
from langchain_core.runnables.utils import ConfigurableField
from langchain_openai import ChatOpenAI
from langchain_core.language_models import LanguageModelInput

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.globals import DEFAULT_LLM_BASE_URL
from vss_ctx_rag.models.tool_models import (
    register_tool_config,
    register_tool,
    ToolBaseModel,
)
from pydantic import Field
from typing import Optional, Any, override
from langchain_nvidia_ai_endpoints import ChatNVIDIA, Model, register_model


class ChatOpenAINoTokenRewrite(ChatOpenAI):
    """
    Smart ChatOpenAI that handles max_tokens/max_completion_tokens based on model capabilities.

    This class intelligently chooses the correct token parameter based on the model:
    - Uses max_completion_tokens for newer models (GPT-5, o1 series)
    - Uses max_tokens for older models (GPT-4 and earlier)

    This ensures compatibility across all OpenAI models.
    """

    @override
    @property
    def _default_params(self) -> dict[str, Any]:
        """Get the default parameters, applying smart token parameter handling."""
        params = super(ChatOpenAI, self)._default_params

        if self.model_name.startswith("gpt-5") or re.match(r"^o\d", self.model_name):
            return super()._default_params
        else:
            return params

    @override
    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> dict:
        """Get request payload with smart token parameter and role handling."""

        if self.model_name.startswith("gpt-5") or re.match(r"^o\d", self.model_name):
            payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        else:
            payload = super(ChatOpenAI, self)._get_request_payload(
                input_, stop=stop, **kwargs
            )

        if self.model_name and re.match(r"^o\d", self.model_name):
            for message in payload.get("messages", []):
                if message["role"] == "system":
                    message["role"] = "developer"

        return payload


@register_tool_config("llm")
class LLMConfig(ToolBaseModel):
    model: str
    base_url: Optional[str] = DEFAULT_LLM_BASE_URL
    max_tokens: Optional[int] = Field(default=2048, ge=1)
    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    api_key: Optional[str] = "NOAPIKEYSET"
    reasoning_effort: Optional[str] = Field(default=None)


class LLMTool(Tool, Runnable):
    """A Tool class wrapper for LLMs.

    Returns:
        LLMTool: A Tool that wraps an LLM.
    """

    llm: BaseChatModel

    def __init__(self, llm, name="llm_tool", config=None, tools=None) -> None:
        Tool.__init__(self, name, config, tools)
        self.llm = llm

    def __getattr__(self, attr):
        return getattr(self.llm, attr)

    def invoke(self, *args, **kwargs):
        return self.llm.invoke(*args, **kwargs)

    def stream(self, *args, **kwargs):
        return self.llm.stream(*args, **kwargs)

    def batch(self, *args, **kwargs):
        return self.llm.batch(*args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        return self.llm.with_structured_output(*args, **kwargs)

    async def ainvoke(self, *args, **kwargs):
        return await self.llm.ainvoke(*args, **kwargs)

    async def astream(self, *args, **kwargs):
        return await self.llm.astream(*args, **kwargs)

    async def abatch(self, *args, **kwargs):
        return await self.llm.abatch(*args, **kwargs)

    @property
    def _llm_type(self) -> str:
        return self.name


@register_tool(config=LLMConfig)
class ChatOpenAITool(LLMTool):
    def __init__(self, name="llm", config=None, tools=None):
        self.name = name
        self.update_tool(config, tools)

    def update_tool(self, config, tools=None):
        self.config = config

        model = (self.config.params.model or "").strip()
        api_key = (self.config.params.api_key or "").strip()
        if not api_key or api_key.strip() == "":
            api_key = "NOAPIKEYSET"
        base_url = (self.config.params.base_url or "").strip()

        llm_params = {
            "temperature": self.config.params.temperature,
        }
        configurable_fields = {
            "temperature": ConfigurableField(id="temperature"),
            "max_tokens": ConfigurableField(id="max_tokens"),
        }
        if not (model.startswith("gpt-5") or re.match(r"^o\d", model)):
            llm_params["top_p"] = self.config.params.top_p

        if self.config.params.reasoning_effort is not None:
            llm_params["reasoning_effort"] = self.config.params.reasoning_effort
            configurable_fields["reasoning_effort"] = ConfigurableField(
                id="reasoning_effort"
            )

        if self.config.params.max_completion_tokens is not None:
            llm_params["max_completion_tokens"] = (
                self.config.params.max_completion_tokens
            )
        elif model.startswith("gpt-5") or re.match(r"^o\d", model):
            llm_params["max_completion_tokens"] = self.config.params.max_tokens
            logger.debug(
                f"Using max_completion_tokens {self.config.params.max_tokens} for {model}"
            )
        else:
            llm_params["max_tokens"] = self.config.params.max_tokens
            logger.debug(
                f"Using max_tokens {self.config.params.max_tokens} for {model}"
            )

        if not (model.startswith("gpt-5") or re.match(r"^o\d", model)):
            configurable_fields["top_p"] = ConfigurableField(id="top_p")

        if model and "nvcf" in base_url:
            register_model(
                Model(
                    id=model, model_type="chat", client="ChatNVIDIA", endpoint=base_url
                )
            )
            super().__init__(
                llm=ChatNVIDIA(
                    model=model, api_key=api_key, **llm_params
                ).configurable_fields(**configurable_fields),
                name=self.name,
                config=self.config,
                tools=tools,
            )
            logger.debug(f"Using NVIDIA LLM {model} with base URL {base_url}")
        else:
            super().__init__(
                llm=ChatOpenAINoTokenRewrite(
                    model=model, api_key=api_key, base_url=base_url, **llm_params
                ).configurable_fields(**configurable_fields),
                name=self.name,
                config=self.config,
                tools=tools,
            )
            logger.debug(f"Using OpenAI LLM {model} with base URL {base_url}")

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

    def update(
        self, top_p=None, temperature=None, max_tokens=None, max_completion_tokens=None
    ):
        configurable_dict = {}
        model = self.config.params.model

        if top_p is not None and not (
            model.startswith("gpt-5") or re.match(r"^o\d", model)
        ):
            configurable_dict["top_p"] = top_p

        if temperature is not None:
            configurable_dict["temperature"] = temperature

        if max_completion_tokens is not None:
            configurable_dict["max_tokens"] = max_completion_tokens
        elif max_tokens is not None:
            configurable_dict["max_tokens"] = max_tokens

        logger.debug(f"Updating LLM with config:{configurable_dict}")
        self.llm = self.llm.with_config(configurable=configurable_dict)
