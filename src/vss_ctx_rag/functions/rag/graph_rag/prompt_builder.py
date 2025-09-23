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

import copy
from typing import List, Any
from .prompt_config import (
    PromptConfig,
    BasePromptSection,
    CORE_SECTIONS,
    SUGGESTIONS,
    STRICT_RULES,
    RESPONSE_GUIDELINES,
    MULTIPLE_CHOICE_RULES,
    EVALUATION_GUIDANCE,
)


class PromptBuilder:
    """Builder class for creating modular prompts."""

    def __init__(self, config: PromptConfig):
        self.config = config
        self.sections: List[BasePromptSection] = []

    def add_section(self, section: BasePromptSection) -> "PromptBuilder":
        """Add a section to the prompt if conditions are met."""
        if section.should_include(self.config):
            self.sections.append(section)
        return self

    def add_tools_section(self, tools: List[Any]) -> "PromptBuilder":
        """Add tool documentation sections dynamically."""
        if not tools:
            return self

        tools_content = "## Available Tools\nYou can call any combination of these tools by using separate <execute> blocks for each tool call. Additionally, if you include multiple queries in the same call, they must be separated by ';'.\n\n"

        tool_counter = 1
        for tool in tools:
            if hasattr(tool, "generate_prompt_section"):
                # Use the tool's built-in prompt generation
                tool_section = tool.generate_prompt_section(
                    tool_number=tool_counter,
                    camera_format="camera_X",  # For ChunkFilter
                )
                tools_content += tool_section + "\n\n"
                tool_counter += 1

        section = BasePromptSection(
            title="Available Tools", content=tools_content.strip(), order=5
        )
        return self.add_section(section)

    def add_suggestions_section(self, tools: List[Any]) -> "PromptBuilder":
        """Add suggestions section based on configuration."""
        content = "## Suggestions\n"

        # Add general suggestions
        for i, suggestion in enumerate(SUGGESTIONS["general"], 1):
            content += f"{i}. {suggestion}\n"

        current_num = len(SUGGESTIONS["general"]) + 1

        # Add conditional chunk fallback
        chunk_fallback = SUGGESTIONS["chunk_fallback"]
        if (
            not self.config.multi_channel
            and tools
            and any("chunk_reader" in tool.name for tool in tools)
        ):  # !multi_channel condition
            content += f"{current_num}. {chunk_fallback['content']}\n"
            current_num += 1

        # Add counting suggestions
        for suggestion in SUGGESTIONS["counting"]:
            content += f"{current_num}a. {suggestion}\n"
            current_num += 1

        # Add temporal suggestions
        for suggestion in SUGGESTIONS["temporal"]:
            content += f"{current_num}. {suggestion}\n"
            current_num += 1

        # Add chunk reader and chunk search suggestions
        if (
            tools
            and any("chunk_reader" in tool.name for tool in tools)
            and any("chunk_search" in tool.name for tool in tools)
        ):
            for suggestion in SUGGESTIONS["chunk_reader_and_search"]:
                content += f"{current_num}. {suggestion}\n"
                current_num += 1

        # Add output suggestions
        for suggestion in SUGGESTIONS["output"]:
            content += f"{current_num}. {suggestion}\n"
            current_num += 1

        section = BasePromptSection(title="Suggestions", content=content, order=8)
        return self.add_section(section)

    def add_strict_rules_section(self, tools: List[Any]) -> "PromptBuilder":
        """Add strict rules section based on configuration."""
        content = "## Strict Rules\n"

        # Add general rules
        for i, rule in enumerate(STRICT_RULES["general"], 1):
            content += f"{i}. {rule}\n"

        # Add conditional chunk reader and chunk search rules
        if (
            tools
            and any("chunk_reader" in tool.name for tool in tools)
            and any("chunk_search" in tool.name for tool in tools)
        ):
            for rule in STRICT_RULES["chunk_reader_and_search"]:
                content += f"{len(STRICT_RULES['general']) + 1}. {rule}\n"

        section = BasePromptSection(title="Strict Rules", content=content, order=9)
        return self.add_section(section)

    def build(self) -> str:
        """Build the final prompt string."""
        # Sort sections by order
        sorted_sections = sorted(self.sections, key=lambda x: x.order)

        # Combine all sections
        prompt_parts = []
        for section in sorted_sections:
            prompt_parts.append(section.content)

        return "\n\n".join(prompt_parts)


class ThinkingPromptBuilder(PromptBuilder):
    """Specialized builder for thinking agent prompts."""

    def build_thinking_prompt(self, tools: List[Any] = None) -> str:
        """Build a complete thinking agent prompt."""
        # Add core sections
        self.add_section(CORE_SECTIONS["agent_role"])
        self.add_section(CORE_SECTIONS["workflow_steps"])

        # Add appropriate final answer section
        final_answer_section = CORE_SECTIONS["final_answer"]
        if self.config.multi_choice:
            answer_format = "(only the letter (A, B, C, D, ...))"
        else:
            answer_format = "Final answer with timestamps, locations, visual descriptions, and supporting evidence"

        # Create a copy of the section to avoid modifying the original

        final_answer_section = copy.deepcopy(final_answer_section)

        # Format only the answer_format placeholder, leaving num_cameras_info for runtime
        final_answer_section.content = final_answer_section.content.replace(
            "{answer_format}", answer_format
        )
        self.add_section(final_answer_section)

        # Add critical assumption
        self.add_section(CORE_SECTIONS["critical_assumption"])

        # Add tools if provided
        if tools:
            self.add_tools_section(tools)

        # Add cross-camera tracking for multi-channel
        if self.config.multi_channel:
            self.add_section(CORE_SECTIONS["cross_camera_tracking"])

        # Add suggestions and rules
        self.add_suggestions_section(tools)
        self.add_strict_rules_section(tools)

        return self.build()


class ResponsePromptBuilder(PromptBuilder):
    """Specialized builder for response agent prompts."""

    def build_response_prompt(self) -> str:
        """Build a complete response agent prompt."""
        content = """You are a response agent that provides comprehensive, informative answers to users based on analysis and tool results.

Your role is to:
1. Take the final analysis from the thinking agent
2. Provide a detailed, informative answer to the original user query
3. Include relevant context, evidence, and supporting details

RESPONSE GUIDELINES:
"""

        # Add comprehensive guidelines
        for guideline in RESPONSE_GUIDELINES["comprehensive"]:
            content += f"- {guideline}\n"

        # Add formatting guidelines
        for guideline in RESPONSE_GUIDELINES["formatting"]:
            content += f"- {guideline}\n"

        # Add format compliance guidelines
        content += "\nFORMAT COMPLIANCE:\n"
        for guideline in RESPONSE_GUIDELINES["format_compliance"]:
            content += f"- {guideline}\n"

        if self.config.multi_choice:
            content += "\nMULTIPLE CHOICE QUESTIONS:\n"
            content += MULTIPLE_CHOICE_RULES

        section = BasePromptSection(title="Response Agent", content=content, order=1)
        self.add_section(section)

        return self.build()


class EvaluationPromptBuilder(PromptBuilder):
    """Specialized builder for evaluation guidance prompts."""

    def build_evaluation_prompt(self, tools: List[Any] = None) -> str:
        """Build evaluation guidance prompt."""
        content = ""
        # Add general guidelines
        for i, guideline in enumerate(EVALUATION_GUIDANCE["general"], 1):
            content += f"{i}. {guideline}\n"

        current_num = len(EVALUATION_GUIDANCE["general"]) + 1

        # Add chunk reader and chunk search guidelines
        if (
            tools
            and any("chunk_reader" in tool.name for tool in tools)
            and any("chunk_search" in tool.name for tool in tools)
        ):
            for guideline in EVALUATION_GUIDANCE["chunk_reader_and_search"]:
                content += f"{current_num}. {guideline}\n"
                current_num += 1

        if self.config.multi_choice:
            for guideline in EVALUATION_GUIDANCE["multi_choice"]:
                content += f"{current_num}. {guideline}\n"
                current_num += 1

        section = BasePromptSection(
            title="Evaluation Guidance", content=content, order=8
        )
        self.add_section(section)
        return self.build()


class PromptFactory:
    """Factory class for creating different types of prompts."""

    @staticmethod
    def create_thinking_prompt(config: PromptConfig, tools: List[Any] = None) -> str:
        """Create a thinking agent prompt."""
        builder = ThinkingPromptBuilder(config)
        return builder.build_thinking_prompt(tools)

    @staticmethod
    def create_response_prompt(config: PromptConfig) -> str:
        """Create a response agent prompt."""
        builder = ResponsePromptBuilder(config)
        return builder.build_response_prompt()

    @staticmethod
    def create_evaluation_prompt(config: PromptConfig, tools: List[Any] = None) -> str:
        """Create an evaluation guidance prompt."""
        builder = EvaluationPromptBuilder(config)
        return builder.build_evaluation_prompt(tools)
