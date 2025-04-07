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
from aiq.builder.framework_enum import LLMFrameworkEnum
from aiq.cli.register_workflow import register_function
from aiq.data_models.component_ref import FunctionRef
from aiq.data_models.component_ref import LLMRef
from aiq.data_models.function import FunctionBaseConfig

from vss_ctx_rag.aiq.workflow.tool_call_workflow import build_workflow_fn


class ToolCallWorkflowConfig(FunctionBaseConfig, name="tool_call_workflow"):
    tool_names: list[FunctionRef] = []
    llm_name: LLMRef


@register_function(
    config_type=ToolCallWorkflowConfig, framework_wrappers=[LLMFrameworkEnum.LANGCHAIN]
)
async def tool_call_workflow(config: ToolCallWorkflowConfig, builder: Builder):
    yield await build_workflow_fn(config, builder)
