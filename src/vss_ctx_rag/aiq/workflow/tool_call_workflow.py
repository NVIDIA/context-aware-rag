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

import logging

from langchain_core.tools.structured import StructuredTool
from aiq.builder.builder import Builder
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.builder.function_info import FunctionInfo
from aiq.data_models.api_server import AIQChatResponse

logger = logging.getLogger(__name__)


async def build_workflow_fn(config, builder: Builder):
    tools: list[StructuredTool] = builder.get_tools(
        config.tool_names, wrapper_type=LLMFrameworkEnum.LANGCHAIN
    )

    async def run_workflow(rag_request: str) -> AIQChatResponse:
        tool = get_document_ingestion_tool(tools)
        return await tool.ainvoke(rag_request)

    return FunctionInfo.from_fn(
        run_workflow, description="Run the tool ingestion workflow"
    )


def get_document_ingestion_tool(tools: list[StructuredTool]) -> StructuredTool:
    for tool in tools:
        if tool.name == "ingestion_function":
            return tool
    raise ValueError("ingestion_function not found in tools")
