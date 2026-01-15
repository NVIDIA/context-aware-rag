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

from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain_core.prompts import PromptTemplate


class PromptCapableTool(BaseTool, ABC):
    """
    Base class for tools that can generate their own prompt templates.

    This enables self-documenting tools where each tool defines its own
    XML format, description, and usage rules for dynamic prompt generation.
    """

    @classmethod
    @abstractmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """
        Return prompt template information for this tool.

        Returns:
            Dictionary with keys:
            - xml_format: XML template for tool usage
            - description: Tool description and use cases
            - rules: Important rules and constraints (optional)
        """
        pass

    @classmethod
    def get_dynamic_rules(cls, tools: List[Any], config: Optional[Any] = None) -> str:
        """
        Get dynamic rules specific to this tool based on configuration.
        Override this method in tool subclasses to provide config-specific rules.

        Args:
            tools: List of available tools for context-aware rule generation
            config: Configuration object to customize rule generation

        Returns:
            Additional rules as a formatted string
        """
        return ""

    @classmethod
    def generate_prompt_section(
        cls, tool_number: int, tools: List[Any], config: Optional[Any] = None, **kwargs
    ) -> str:
        """
        Generate a complete prompt section for this tool with dynamic content based on config.

        Args:
            tool_number: The number to assign to this tool in the prompt
            tools: List of available tools for context-aware rule generation
            config: Configuration object to customize prompt generation
            **kwargs: Additional template variables (e.g., camera_format)

        Returns:
            Complete prompt section for this tool
        """
        template_info = cls.get_prompt_template_info()

        # Start with base rules from template
        dynamic_rules = template_info.get("rules", "")

        # Add tool-specific dynamic rules based on config
        tool_dynamic_rules = cls.get_dynamic_rules(tools, config)
        if tool_dynamic_rules:
            dynamic_rules += "\n\n" + tool_dynamic_rules

        # Create prompt template
        template = PromptTemplate(
            input_variables=["tool_number"] + list(kwargs.keys()),
            template=f"""
### {{tool_number}}. {cls.__name__}

{template_info['xml_format']}

- Use case:

  {template_info['description']}

{dynamic_rules}
""",
        )

        return template.format(tool_number=tool_number, **kwargs)


def _get_tool_section(tools: List[Any]) -> str:
    """Generate the tools section dynamically from provided tools."""
    if not tools:
        return ""

    content = "\n\n## Available Tools\n"

    content += "You can call any combination of these tools by using separate <execute> blocks for each tool call. Additionally, if you include multiple queries in the same call, they must be separated by ';'.\n\n"

    # Generate tool documentation
    for i, tool in enumerate(tools, 1):
        if hasattr(tool, "generate_prompt_section"):
            tool_section = tool.generate_prompt_section(
                tool_number=i,
                tools=tools,
                camera_format="camera_X",
            )
            content += tool_section + "\n\n"

    return content


def _has_chunk_reader(tools: List[Any]) -> bool:
    """Check if ChunkReader tool is present."""
    return tools and any("ChunkReader" in tool.name for tool in tools)


def _has_chunk_search(tools: List[Any]) -> bool:
    """Check if ChunkSearch tool is present."""
    return tools and any("ChunkSearch" in tool.name for tool in tools)


def _has_both_chunk_tools(tools: List[Any]) -> bool:
    """Check if both ChunkReader and ChunkSearch tools are present."""
    return tools and _has_chunk_reader(tools) and _has_chunk_search(tools)


def _get_base_agent_workflow() -> str:
    """Get the base agent role and workflow description."""
    return """You are a strategic planner and reasoning expert working with an execution agent to analyze videos.

## Your Capabilities

You do **not** call tools directly. Instead, you generate structured plans for the Execute Agent to follow.

## Workflow Steps

You will follow these steps:

### Step 1: Analyze & Plan
- Document reasoning in `<thinking></thinking>`.
- Output one or more tool calls (strict XML format) in separate 'execute' blocks.
- **CRITICAL**: When one tool's output is needed as input for another tool, make only the first tool call and wait for results.
- Stop immediately after and output `[Pause]` to wait for results.

### Step 2: Wait for Results
After you propose execute steps, stop immediately after and output `[Pause]` to wait for results.

### Step 3: Interpret & Replan
Once the Execute Agent returns results, analyze them inside `<thinking></thinking>`.
- If the results contain information needed for subsequent tool calls (like chunk IDs from ChunkFilter), use those actual values in your next tool calls.
- Propose next actions until you have enough information to answer.

### Step 4: Final Answer
Only when confident, output:
```<thinking>Final reasoning with comprehensive analysis of all evidence found</thinking><answer>
Final answer with timestamps, locations, visual descriptions, and supporting evidence</answer>

{num_cameras_info}
{video_length_info}
"""


def _get_critical_assumptions(tools: List[Any]) -> str:
    if _has_chunk_reader(tools):
        return """
CRITICAL ASSUMPTIONS:
1. ALL queries describe scenes from video content that you must search for using your tools. NEVER treat queries as logic puzzles or general knowledge questions - they are ALWAYS about finding specific video content.
2. 🚨 **VISUAL ANALYSIS MANDATORY:** For ANY question about visual details (colors, appearance, objects, actions, spatial relationships, etc.) - if you find relevant content but text doesn't specify the visual information needed, you MUST use ChunkReader immediately. Do NOT skip to other content without visual examination first!
"""
    else:
        return """CRITICAL ASSUMPTION: ALL queries describe scenes from video content that you must search for using your tools. NEVER treat queries as logic puzzles or general knowledge questions - they are ALWAYS about finding specific video content.

"""


def _get_cross_camera_tracking(multi_channel: bool) -> str:
    """Get cross-camera tracking instructions if multi-channel is enabled."""
    if not multi_channel:
        return ""

    return """**CROSS-CAMERA ENTITY TRACKING**: When a query identifies a person/object in a specific camera at a specific time and asks about that entity's location in other cameras, you MUST follow this two-step approach:
- Step 1: First use ChunkFilter to examine the specified camera at the specified time to identify and describe the person/object (use camera_id format: camera_1, camera_2, camera_3, camera_4, etc.)
- Step 2: Then use EntitySearch with the description from Step 1 to find where that same person/object appears in other cameras at various time periods (NOT necessarily at the same timestamp)
- Do NOT assume the entity appears at the same timestamp across all cameras - search broadly across time ranges for each camera
- NEVER conclude after just Step 1 - you MUST complete both steps for cross-camera queries
- If EntitySearch doesn't find the entity in other cameras, try alternative search terms and time ranges before concluding

"""


def _get_suggestions() -> str:
    """Get general suggestions for tool usage."""
    return """
## SUGGESTIONS
- Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries).
- For counting problems, remember it is the same video, do not sum the results from multiple chunks.
- For ordering, you can either use the chunk_id or the timestamps to determine the order.
"""


def _get_chunk_reader_search_logic(tools: List[Any]) -> str:
    """Get chunk reader and search specific logic."""

    if _has_both_chunk_tools(tools):
        return """
- To save the calling budget, it is suggested that you include as many tool calls as possible in the same response, but you can only concatenate video chunks that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of two at a time!!
- Suppose the `<chunk_search>event</chunk_search>` returns a list of segments [a,b,c,d,e]. If the `chunk_reader` checks each chunk in turn and finds that none contain the event, but you still need to locate the chunk where the event occurs, then by default, assume the event occurs in the top-1 chunk a.**
- Use the chunk_reader tool only when the retrieved chunk metadata is insufficient to answer the question confidently, or when visual confirmation is specifically required.
- When the question explicitly refers to a specific scene for answering (e.g., "What did Player 10 do after scoring the first goal?\"), first use chunk_search to locate relevant chunks. Only use chunk_reader if the chunk metadata doesn't provide sufficient detail to identify the scene with confidence. Once the key scene is identified—e.g., the moment of Player 10's first goal in chunk N—you should then generate follow-up questions based only on that chunk and its adjacent chunks. For example, to answer what happened after the first goal, you should ask questions targeting chunk N and chunk N+1.
- SEQUENTIAL EXECUTION: When using ChunkFilter or ChunkSearch followed by ChunkReader, you MUST wait for the first tool's results to get actual chunk_ids before calling ChunkReader. Never use placeholder values like 'chunk_N' - always use the real chunk_ids returned from the previous tool.
- The chunk_search may make mistakes, and the chunk_reader is more accurate than the chunk_search. If the chunk_search retrieves a chunk but the chunk_reader indicates that the chunk is irrelevant to the current query, the result from the chunk_reader should be trusted.
- Each time the user returns retrieval results, you should query all the retrieved chunks in the next round. If clips retrieved by different queries overlap, you can merge all the queries into a single question and access the overlapping chunk only once using chunk_reader
"""

    return ""


def _get_core_rules() -> str:
    """Get core rules for the thinking agent."""
    return """\n\n## Strict Rules

1. Response of each round should provide thinking process in <thinking></thinking> at the beginning!! Never output anything after [Pause]!!
2. You can only concatenate video chunks that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of TWO at a time!!!
3. If you are unable to give a precise answer or you are not sure, continue calling tools for more information; if the maximum number of attempts has been reached and you are still unsure, choose the most likely one.
"""


def _get_tool_specific_rules(tools: List[Any]) -> str:
    """Get rules specific to available tools."""
    if not _has_both_chunk_tools(tools):
        return ""

    return """
5. **TOOL DEPENDENCY RULE**: When one tool's output is required as input for another tool (e.g., ChunkFilter → ChunkReader, ChunkSearch → ChunkReader), execute them sequentially, not in parallel. Wait for the first tool's results before calling the dependent tool.
6. Analyze each chunk returned by the chunk search carefully. Only use chunk_reader for verification when you are genuinely uncertain about the answer based on the chunk metadata alone.
"""


def _get_multi_channel_rules(multi_channel: bool) -> str:
    """Get rules specific to multi-channel questions."""
    if not multi_channel:
        return ""

    return """
4. **DO NOT CONCLUDE PREMATURELY**: For complex queries (especially cross-camera tracking), you MUST make multiple tool calls and exhaust all search strategies before providing a final answer. One tool call is rarely sufficient for comprehensive analysis.
"""


def _get_final_rule() -> str:
    """Get rules specific to final rule."""
    return """
- Don't output anything after [Pause] !!!!!!
"""


def create_thinking_prompt(
    tools: List[Any] = None,
    multi_channel: bool = False,
) -> str:
    """
    Create a thinking agent prompt with specified configuration.

    Args:
        tools: List of tools to include in the prompt
        multi_channel: Enable cross-camera entity tracking

    Returns:
        Complete thinking agent prompt
    """
    # Build the prompt by combining all sections
    sections = [
        # Base workflow and capabilities
        _get_base_agent_workflow(),
        # Context and assumptions
        _get_critical_assumptions(tools),
        # Tools and features
        _get_tool_section(tools),
        _get_cross_camera_tracking(multi_channel),
        # Guidelines and examples
        _get_suggestions(),
        _get_chunk_reader_search_logic(tools),
        _get_final_rule(),
        # Rules
        _get_core_rules(),
        _get_multi_channel_rules(multi_channel),
        _get_tool_specific_rules(tools),
    ]

    # Filter out empty sections and join
    return "".join(section for section in sections if section)


def create_response_prompt() -> str:
    """Create a response agent prompt."""
    return """
You are a response agent that provides comprehensive answers based on analysis and tool results.

**CORE REQUIREMENTS:**
- Provide detailed, evidence-based answers with timestamps, locations, and visual descriptions
- Include ALL relevant findings and supporting evidence from the analysis
- Explain your conclusions and provide chronological context when relevant
- Never include chunk IDs or internal system identifiers in responses

**FORMATTING:**
- Use factual, direct language without pleasantries ("Certainly!", "Here is...", etc.)
- State "No relevant information found" if no relevant data was discovered
- Follow user-specified format requirements exactly (yes/no only, case requirements, length constraints, etc.)
- When format is specified, prioritize format compliance over comprehensive explanations
"""


def _get_evaluation_base_guidance(tools: List[Any]) -> str:
    """Get the base evaluation guidance."""

    if _has_both_chunk_tools(tools):
        return """
EVALUATION GUIDANCE:
1. If you're getting similar or repetitive results from ChunkSearch that don't match the exact scene described, consider using ChunkReader with the most relevant chunk_id from the available results instead of continuing to search unsuccessfully.
2. "CRITICAL: If multiple search attempts have failed to find the specific information requested, try various strategies if needed.",
3. "Even after multiple tool calls, you are getting the same answer, conclude the answer with the information you have gathered.",
🚨 **CRITICAL: NO INFINITE LOOPS** 🚨
   - If a tool returns no results or empty results, DO NOT repeat the exact same tool call
   - IMMEDIATELY switch to fallback strategies
   - For visual questions: if chunk_search fails, try different descriptions or keywords
"""
    else:
        return """
EVALUATION GUIDANCE:
- Conclude with gathered information if repeated tool calls yield the same results
- Never repeat identical tool calls that return no results or empty results
- For failed searches: try ChunkSearch for specific entities or break down complex terms into simpler components
"""


def _get_evaluation_tool_specific_guidance(tools: List[Any]) -> str:
    """Get tool-specific evaluation guidance."""
    common_guidance = """CRITICAL: If multiple search attempts have failed to find the specific information requested, try:
1. ChunkSearch for specific people/objects mentioned in the query
2. Break down complex search terms into simpler components"""

    if not _has_both_chunk_tools(tools):
        return f"""
{common_guidance}
"""

    return f"""
FALLBACK STRATEGY: Look at the execution results and identify the chunk_id of the most relevant scene found (even if not perfect match), then use ChunkReader to visually analyze that scene to answer the question.
MULTI-CHUNK IMAGEQNA STRATEGY: If ChunkReader was used on one chunk but didn't find the answer, try it on additional chunks from your search results:
   - Review all chunk_ids mentioned in your previous search results
   - Select 2-3 additional chunk_ids that could potentially contain the scene/person/object
   - Use ChunkReader with the exact same question on these additional chunks
   - **ALSO TRY ADJACENT CHUNKS**: Use chunks immediately before/after your main results
   - This increases your chances of finding the specific content described in the query
{common_guidance}
3. Try ChunkReader on different chunk_ids from your results (ESPECIALLY if first ChunkReader attempt failed)
4. If ChunkReader has returned "No images found for the specified chunk ID." for previous calls then *do not use* ChunkReader tool for next iterations.
5. **FOR TEMPORAL QUESTIONS**: When you get multiple relevant results, compare their timestamps and choose based on temporal logic:
   - "First" questions: Choose the EARLIEST timestamp
   - "Last" questions: Choose the LATEST timestamp
   - "After X" questions: Choose results with timestamps AFTER the reference event
   - "Before X" questions: Choose results with timestamps BEFORE the reference event
6. For open-ended questions without options: State "No relevant information found" only if no evidence is found
"""


def create_evaluation_prompt(
    tools: List[Any] = None,
) -> str:
    """Generate an evaluation guidance prompt."""
    sections = [
        _get_evaluation_base_guidance(tools),
        _get_evaluation_tool_specific_guidance(tools),
    ]

    return "".join(section for section in sections if section)
