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
        prompt_name = getattr(cls, "prompt_name", "") or cls.__name__
        template = PromptTemplate(
            input_variables=["tool_number"] + list(kwargs.keys()),
            template=f"""
### {{tool_number}}. {prompt_name}

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

    # Add visual details warning for ChunkReader
    if _has_chunk_reader(tools):
        content += """\n\n🔥 **BEFORE USING ANY TOOL - READ THIS:**
If your question asks about VISUAL DETAILS (colors, appearance, objects, actions, spatial relationships, etc.):
1. Find the relevant occurrence with ChunkFilter/ChunkSearch
2. If text doesn't specify the visual information you need → IMMEDIATELY use ChunkReader on that time period
3. Do NOT search other content until you've visually analyzed the relevant occurrence

"""

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


def _has_subtitle_tools(tools: List[Any]) -> bool:
    """Check if any tools are subtitle-related."""
    return tools and any("SubtitleSearch" in tool.name for tool in tools)


def _has_chunk_reader(tools: List[Any]) -> bool:
    """Check if ChunkReader tool is present."""
    return tools and any("ChunkReader" in tool.name for tool in tools)


def _has_chunk_search(tools: List[Any]) -> bool:
    """Check if ChunkSearch tool is present."""
    return tools and any("ChunkSearch" in tool.name for tool in tools)


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
```<thinking>Final reasoning with comprehensive analysis of all evidence found</thinking><answer>"""


def _get_answer_format(multi_choice: bool) -> str:
    """Get the answer format configuration."""
    if multi_choice:
        return "(only the letter (A, B, C, D, ...))"
    else:
        return "Final answer with timestamps, locations, visual descriptions, and supporting evidence"


def _get_subtitle_strategy(tools: List[Any]) -> str:
    """Get subtitle-first strategy if subtitle tools are available."""
    if not _has_subtitle_tools(tools):
        return ""

    return """🔥 **SUBTITLE-FIRST STRATEGY**: This agent is specifically optimized for subtitle questions. For ANY question involving spoken content, dialogue, conversations, opinions, or audio - **ALWAYS use subtitle_search as your FIRST tool call** before any other tools.

🚨 **SUBTITLE QUESTION PATTERNS - ALWAYS USE SUBTITLE_SEARCH FIRST:**
- "which subtitles appear..."
- "what subtitles are displayed..."
- Questions with subtitle text as answer choices (A, B, C, D with text content)
- "what is said during..." or "what is mentioned when..."
- Any question asking about text/words appearing on screen with timing

**CRITICAL EXAMPLE:** If the question is "which subtitles appear at the same time as [visual description]?" →
✅ CORRECT: Use subtitle_search FIRST to find all subtitle options and their timestamps, then use chunk_reader to verify visual content at those specific times
❌ WRONG: Use chunk_search to find the visual description first

"""


def _get_critical_assumptions(tools: List[Any]) -> str:
    """Get subtitle + visual verification workflow if both tools are available."""
    if _has_chunk_reader(tools) and _has_subtitle_tools(tools):
        return """
**SUBTITLE + VISUAL VERIFICATION WORKFLOW:**
When you have subtitle timestamps and need to verify visual content:
1. subtitle_search to find subtitle options and timestamps ✓
3. chunk_reader to visually check what's happening at each time
4. DO NOT use chunk_search when you already have specific timestamps

CRITICAL ASSUMPTIONS:
1. ALL queries describe scenes from video content that you must search for using your tools. NEVER treat queries as logic puzzles or general knowledge questions - they are ALWAYS about finding specific video content.
2. 🚨 **SUBTITLE PRIORITY RULE**: If the question involves any subtitle related content OR asks "which subtitles..." - use subtitle_search FIRST, then use other tools as needed. Do NOT start with visual search for subtitle questions.
3. 🚨 **VISUAL ANALYSIS MANDATORY:** For ANY question about visual details (colors, appearance, objects, actions, spatial relationships, etc.) - if you find relevant content but text doesn't specify the visual information needed, you MUST use ChunkReader immediately. Do NOT skip to other content without visual examination first!
"""
    elif _has_chunk_reader(tools) and not _has_subtitle_tools(tools):
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


def _get_suggestions_header() -> str:
    """Get suggestions header."""
    return """
## SUGGESTIONS"""


def _get_suggestions() -> str:
    """Get general suggestions for tool usage."""
    return """
- Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries).
- For counting problems, consider using a higher top-k and more diverse queries to ensure no missing items.
- For counting problems, your question should follow this format: Is there xxx occur in this video? (A chunk should be considered correct as long as the queried event occurs in more than one frame, even if the chunk also includes other content or is primarily focused on something else. coarsely matched chunks should be taken into account (e.g., watering flowers vs. watering toy flowers))
- For counting problems, you should carefully examine each chunk to avoid any omissions!!!
- For ordering, you can either use the chunk_id or the timestamps to determine the order.
"""


def _get_chunk_reader_search_logic(multi_choice: bool, tools: List[Any]) -> str:
    """Get chunk reader and search specific logic."""
    prompt = ""
    if _has_chunk_reader(tools) and _has_chunk_search(tools):
        prompt += """
- To save the calling budget, it is suggested that you include as many tool calls as possible in the same response, but you can only concatenate video chunks that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of two at a time!!
- Suppose the `<chunk_search>event</chunk_search>` returns a list of segments [a,b,c,d,e]. If the `chunk_reader` checks each chunk in turn and finds that none contain the event, but you still need to locate the chunk where the event occurs, then by default, assume the event occurs in the top-1 chunk a.**
- Use the chunk_reader tool only when the retrieved chunk metadata is insufficient to answer the question confidently, or when visual confirmation is specifically required.
- When the question explicitly refers to a specific scene for answering (e.g., "What did Player 10 do after scoring the first goal?\"), first use chunk_search to locate relevant chunks. Only use chunk_reader if the chunk metadata doesn't provide sufficient detail to identify the scene with confidence. Once the key scene is identified—e.g., the moment of Player 10's first goal in chunk N—you should then generate follow-up questions based only on that chunk and its adjacent chunks. For example, to answer what happened after the first goal, you should ask questions targeting chunk N and chunk N+1.
- SEQUENTIAL EXECUTION: When using ChunkFilter or ChunkSearch followed by ChunkReader, you MUST wait for the first tool's results to get actual chunk_ids before calling ChunkReader. Never use placeholder values like 'chunk_N' - always use the real chunk_ids returned from the previous tool.
- The chunk_search may make mistakes, and the chunk_reader is more accurate than the chunk_search. If the chunk_search retrieves a chunk but the chunk_reader indicates that the chunk is irrelevant to the current query, the result from the chunk_reader should be trusted.
- Each time the user returns retrieval results, you should query all the retrieved chunks in the next round. If clips retrieved by different queries overlap, you can merge all the queries into a single question and access the overlapping chunk only once using chunk_reader
"""

    if multi_choice and _has_chunk_reader(tools) and not _has_subtitle_tools(tools):
        prompt += """
⚠️ **MANDATORY VISUAL ANALYSIS FOR VISUAL DETAIL QUESTIONS** ⚠️:
   When ANY question asks about visual characteristics:
   - If you find relevant content but text doesn't specify the visual details needed
   - You MUST immediately use ChunkReader on that exact time period
   - NEVER skip to other content without visual analysis first
   - Example: Subject mentioned at Time X but missing visual details → MUST use ChunkReader(Time X range, "[Ask about specific visual details needed]")
After getting a confident answer, verify the answer again with the ChunkReader tool on the most relevant chunks.
"""

    return prompt


def _get_subtitle_workflow_examples(tools: List[Any]) -> str:
    """Get subtitle workflow examples if subtitle tools are available."""
    if not _has_subtitle_tools(tools):
        return ""

    return """
- **SUBTITLE WORKFLOW EXAMPLE:** For questions like "What did the person say about the topic?":
    - Step 1: IMMEDIATELY use subtitle_search to find relevant spoken content
    - Step 2: Use chunk_reader with the time ranges from subtitles for visual verification if needed
    - Step 3: Do NOT start with chunk_search for speech-based questions

- **SUBTITLE + VISUAL VERIFICATION EXAMPLE:** For questions like "Which subtitles appear together with a red rectangle on the PPT?":
    - Step 1: Use subtitle_search to find all subtitle options A, B, C, D with their timestamps
    - Step 2: Use chunk_reader on each chunk to visually verify what appears on screen at that time
    - Step 3: ❌ DO NOT use chunk_search when you already have specific timestamps from subtitles
- **VISUAL DETAIL WORKFLOW EXAMPLE:** For questions like "What color clothing is the first person wearing?":
    - Step 1: Use ChunkFilter to find the FIRST person appearance (e.g., 0:10 seconds)
    - Step 2: IMMEDIATELY use ChunkReader with that time range asking "What color clothing is the person wearing?"
    - Step 3: Do NOT search other time periods until you've visually examined the first occurrence
    - Step 4: Only if no person is visible in the first period, then check subsequent periods
"""


def _get_core_rules() -> str:
    """Get core rules for the thinking agent."""
    return """\n\n## Strict Rules

1. Response of each round should provide thinking process in <thinking></thinking> at the beginning!! Never output anything after [Pause]!!
2. You can only concatenate video chunks that are TEMPORALLY ADJACENT to each other (n;n+1), with a maximum of TWO at a time!!!
3. If you are unable to give a precise answer or you are not sure, continue calling tools for more information; if the maximum number of attempts has been reached and you are still unsure, choose the most likely one.
4. **DO NOT CONCLUDE PREMATURELY**: For complex queries (especially cross-camera tracking), you MUST make multiple tool calls and exhaust all search strategies before providing a final answer. One tool call is rarely sufficient for comprehensive analysis.
"""


def _get_tool_specific_rules(tools: List[Any]) -> str:
    """Get rules specific to available tools."""
    if not (tools and _has_chunk_reader(tools) and _has_chunk_search(tools)):
        return ""

    return """
5. **TOOL DEPENDENCY RULE**: When one tool's output is required as input for another tool (e.g., ChunkFilter → ChunkReader, ChunkSearch → ChunkReader), execute them sequentially, not in parallel. Wait for the first tool's results before calling the dependent tool.
6. Analyze each chunk returned by the chunk search carefully. Only use chunk_reader for verification when you are genuinely uncertain about the answer based on the chunk metadata alone.
"""


def _get_multi_choice_rules(multi_choice: bool) -> str:
    """Get rules specific to multi-choice questions."""
    if not multi_choice:
        return ""

    return """
7. Never guess the answer, question about every choice, question about every chunk retrieved by the chunk_search!!!!
8. **VISUAL ANALYSIS PRIORITY:** For questions about visual details, you MUST use ChunkReader to visually examine relevant time periods. Do NOT rely solely on text descriptions that lack the visual information needed.
   📋 **GENERAL SCENARIO EXAMPLE:** Question asks about visual characteristics of something
   - Tool finds: "Time X: [Subject mentioned]" (but visual details missing from text)
   - Tool finds: "Time Y: [Subject with some visual details in text]"
   - ❌ WRONG: Answer based on Time Y without checking Time X first
   - ✅ CORRECT: Use ChunkReader(Time X range, "[Ask about the specific visual details needed]") to analyze the relevant occurrence first
"""


def _get_subtitle_specific_rules(tools: List[Any]) -> str:
    """Get rules specific to subtitle tools."""
    if not _has_subtitle_tools(tools):
        return ""

    return """
9. 🚨 **SUBTITLE SEARCH FIRST RULE**: For ANY question involving spoken content, dialogue, conversations, opinions, arguments, or verbal interactions - you MUST use subtitle_search as your FIRST tool call. **ESPECIALLY** if the question asks "which subtitles appear..." or has subtitle text as answer choices. This is NON-NEGOTIABLE for subtitle-related questions.
10. 🚨 **SUBTITLE + VISUAL VERIFICATION RULE**: When you have subtitle timestamps and need to verify visual content at those times:
    - Use chunk_reader to visually analyze what's happening at each timestamp
    - DO NOT use chunk_search when you already have specific timestamps from subtitle_search
"""


def _get_subtitle_search_first_rule(tools: List[Any]) -> str:
    """Get rules specific to subtitle search first rule."""
    if not _has_subtitle_tools(tools):
        return ""

    return """
- 🚨 **SUBTITLE-FIRST MANDATORY**: For questions about spoken content, dialogue, conversations, opinions, or audio - ALWAYS start with subtitle_search before any other tool. This is your PRIMARY strategy for subtitle-related questions.
"""


def _get_final_rule(tools: List[Any]) -> str:
    """Get rules specific to final rule."""
    if not _has_subtitle_tools(tools):
        return ""

    return """
- Don't output anything after [Pause] !!!!!!
"""


def _get_num_cameras_and_video_length_info(multi_choice: bool) -> str:
    """Get num cameras and video length info."""
    if multi_choice:
        return ""

    return """
{num_cameras_info}
{video_length_info}
"""


def create_thinking_prompt(
    tools: List[Any] = None,
    multi_channel: bool = False,
    multi_choice: bool = False,
) -> str:
    """
    Create a thinking agent prompt with specified configuration.

    Args:
        tools: List of tools to include in the prompt
        multi_channel: Enable cross-camera entity tracking
        multi_choice: Format for multiple choice questions

    Returns:
        Complete thinking agent prompt
    """
    # Build the prompt by combining all sections
    sections = [
        # Base workflow and capabilities
        _get_base_agent_workflow(),
        _get_answer_format(multi_choice),
        "</answer>\n```\n\n",
        _get_num_cameras_and_video_length_info(multi_choice),
        # Context and assumptions
        _get_subtitle_strategy(tools),
        _get_critical_assumptions(tools),
        # Tools and features
        _get_tool_section(tools),
        _get_cross_camera_tracking(multi_channel),
        # Guidelines and examples
        _get_suggestions_header(),
        _get_subtitle_search_first_rule(tools),
        _get_suggestions(),
        _get_chunk_reader_search_logic(multi_choice, tools),
        _get_subtitle_workflow_examples(tools),
        _get_final_rule(tools),
        # Rules
        _get_core_rules(),
        _get_tool_specific_rules(tools),
        _get_multi_choice_rules(multi_choice),
        _get_subtitle_specific_rules(tools),
    ]

    # Filter out empty sections and join
    return "".join(section for section in sections if section)


def _get_response_base_guidelines(multi_choice: bool) -> str:
    """Get the base response agent guidelines."""
    if multi_choice:
        return ""

    return """You are a response agent that provides comprehensive, informative answers to users based on analysis and tool results.

Your role is to:
1. Take the final analysis from the thinking agent
2. Provide a detailed, informative answer to the original user query
3. Include relevant context, evidence, and supporting details

RESPONSE GUIDELINES:
- **COMPREHENSIVE BY DEFAULT**: Provide detailed answers with extensive supporting evidence
- **PRESERVE ALL FINDINGS**: Include relevant observations, timestamps, locations, and descriptive information from the analysis
- **VISUAL DETAILS**: Preserve important identifying details such as physical appearance, clothing, objects, and visual characteristics
- **EVIDENCE-BASED**: Include specific evidence found during the analysis (e.g., timestamps, source locations, visual descriptions)
- **COMPLETE COVERAGE**: When multiple pieces of evidence support an answer, mention ALL relevant findings
- **CONTEXTUAL**: Provide context about the scene, actions, or events that led to the conclusion
- **CHRONOLOGICAL**: If patterns or sequences of events are relevant, describe them chronologically with timestamps
- **JUSTIFY CONCLUSIONS**: Always explain WHY you reached your conclusion based on the evidence gathered

FORMATTING:
- Do not add pleasantries, confirmations, or offers for further help
- Do not say things like "Certainly!", "Here is...", or "If you have any questions..."
- Focus on factual information and evidence-based conclusions
- If no relevant information was found, explain what was searched and why no evidence was located
- Remove any thinking process markers or formatting symbols
- **CRITICAL: NEVER include chunk IDs in your answer** - Do not output chunk IDs in any format including tables, lists, or references
- When creating tables or structured output, exclude chunk ID columns entirely
- Focus on descriptive content (time ranges, camera locations, visual descriptions) without internal system identifiers

FORMAT COMPLIANCE:
- **CRITICAL: FOLLOW EXACT FORMAT REQUIREMENTS**: If the user specifies a particular response format (e.g., 'yes/no only', 'one word', 'lowercase only'), you MUST follow that format exactly
- For yes/no questions: When asked to answer 'yes' or 'no', respond with ONLY 'yes' or 'no' - do not add explanations, confirmations, or additional text
- For case-specific requests: If asked for 'lowercase', provide answer in lowercase. If asked for 'uppercase', provide answer in uppercase
- For length constraints: If asked for 'one word', 'brief', or specific word limits, strictly adhere to those constraints
- For format specifications: If the user requests specific punctuation, structure, or formatting, follow it precisely
- When format is specified, prioritize format compliance over comprehensive explanations
"""


def _get_response_multi_choice_rules(multi_choice: bool) -> str:
    """Get multi-choice specific rules for response agent."""
    if not multi_choice:
        return ""

    return """
You are a response agent that provides clean, direct answers to users based on analysis and tool results.

Your role is to:
1. Take the final analysis from the thinking agent
2. Provide a clean, direct answer to the original user query
3. Be concise and factual

RESPONSE GUIDELINES:
- Be direct and to the point
- Provide ONLY the essential answer without extra commentary
- Do not add pleasantries, confirmations, or offers for further help
- Do not say things like "Certainly!", "Here is...", or "If you have any questions..."
- Just state the facts or answer directly
- If no relevant information was found, simply state "No relevant information found"
- Remove any thinking process markers or formatting symbols

MULTIPLE CHOICE QUESTIONS:
- If the original query contains multiple choice options (A, B, C, D, E), your answer MUST include the option letter
- Format: "D. [Option text]" or just "D" if the option text is obvious from context
- Look for phrases like "A.", "B.", "C.", "D.", "E." in the original query
- The thinking agent's analysis will identify which option matches - use that option letter in your response
- **ALWAYS provide an answer from the given options** - never respond with "No relevant information found" for multiple choice questions
- Select the option that best matches the evidence gathered by the thinking agent, even if not a perfect match
"""


def create_response_prompt(
    multi_choice: bool = False,
) -> str:
    """Create a response agent prompt."""
    sections = [
        _get_response_base_guidelines(multi_choice),
        _get_response_multi_choice_rules(multi_choice),
    ]

    return "".join(section for section in sections if section)


def _get_evaluation_header() -> str:
    """Get the evaluation header."""
    return """
EVALUATION GUIDANCE:
"""


def _get_evaluation_base_guidance(tools: List[Any], multi_choice: bool) -> str:
    """Get the base evaluation guidance."""
    if multi_choice:
        return """
1. If you're getting similar or repetitive results from ChunkSearch that don't match the exact scene described, consider using ChunkReader with the most relevant chunk_id from the available results instead of continuing to search unsuccessfully.
2. 🚨 **CRITICAL: NO INFINITE LOOPS** 🚨
   - If a tool returns no results or empty results, DO NOT repeat the exact same tool call
   - IMMEDIATELY switch to fallback strategies
   - For subtitle questions: if subtitle_search('word') fails, try searching each multiple choice option individually
   - For visual questions: if chunk_search fails, try different descriptions or keywords
"""
    elif tools and _has_chunk_reader(tools) and _has_chunk_search(tools):
        return """
1. If you're getting similar or repetitive results from ChunkSearch that don't match the exact scene described, consider using ChunkReader with the most relevant chunk_id from the available results instead of continuing to search unsuccessfully.
2. 🚨 **CRITICAL: NO INFINITE LOOPS** 🚨
   - If a tool returns no results or empty results, DO NOT repeat the exact same tool call
   - IMMEDIATELY switch to fallback strategies
   - For visual questions: if chunk_search fails, try different descriptions or keywords
"""
    else:
        return """
1. 🚨 **CRITICAL: NO INFINITE LOOPS** 🚨
   - If a tool returns no results or empty results, DO NOT repeat the exact same tool call
   - IMMEDIATELY switch to fallback strategies
"""


def _get_evaluation_tool_specific_guidance(tools: List[Any]) -> str:
    """Get tool-specific evaluation guidance."""
    if not (tools and _has_chunk_reader(tools) and _has_chunk_search(tools)):
        return """
CRITICAL: If multiple search attempts have failed to find the specific information requested, try:
1. ChunkSearch for specific people/objects mentioned in the query
2. Break down complex search terms into simpler components
"""

    return """
FALLBACK STRATEGY: Look at the execution results and identify the chunk_id of the most relevant scene found (even if not perfect match), then use ChunkReader to visually analyze that scene to answer the question.
MULTI-CHUNK IMAGEQNA STRATEGY: If ChunkReader was used on one chunk but didn't find the answer, try it on additional chunks from your search results:
   - Review all chunk_ids mentioned in your previous search results
   - Select 2-3 additional chunk_ids that could potentially contain the scene/person/object
   - Use ChunkReader with the exact same question on these additional chunks
   - **ALSO TRY ADJACENT CHUNKS**: Use chunks immediately before/after your main results
   - This increases your chances of finding the specific content described in the query
CRITICAL: If multiple search attempts have failed to find the specific information requested, try:
1. ChunkSearch for specific people/objects mentioned in the query
2. Break down complex search terms into simpler components
3. Try ChunkReader on different chunk_ids from your results (ESPECIALLY if first ChunkReader attempt failed)
4. If ChunkReader has returned "No images found for the specified chunk ID." for previous calls then *do not use* ChunkReader tool for next iterations.
5. **FOR MULTIPLE CHOICE QUESTIONS**: After exhausting all search strategies, select the closest matching option (A, B, C, D, E) based on the evidence you have gathered
6. **FOR TEMPORAL QUESTIONS**: When you get multiple relevant results, compare their timestamps and choose based on temporal logic:
   - "First" questions: Choose the EARLIEST timestamp
   - "Last" questions: Choose the LATEST timestamp
   - "After X" questions: Choose results with timestamps AFTER the reference event
   - "Before X" questions: Choose results with timestamps BEFORE the reference event
7. For open-ended questions without options: State "No relevant information found" only if no evidence is found
"""


def _get_evaluation_multi_choice_guidance(multi_choice: bool, tools: List[Any]) -> str:
    """Get multi-choice specific evaluation guidance."""
    if not multi_choice and (
        tools and _has_chunk_reader(tools) and _has_chunk_search(tools)
    ):
        return "- CONFIDENCE-BASED USAGE: If chunk search results provide clear, unambiguous information that directly answers the query, you can proceed with confidence without ChunkReader verification."
    elif multi_choice:
        return """
    - ABSOLUTELY NEVER GIVE UP ON MULTIPLE CHOICE QUESTIONS - always provide the most likely answer from the given options based on your search results.
    """
    else:
        return """
"""


def create_evaluation_prompt(
    tools: List[Any] = None, multi_choice: bool = False
) -> str:
    """Generate an evaluation guidance prompt."""
    sections = [
        _get_evaluation_header(),
        _get_evaluation_base_guidance(tools, multi_choice),
        _get_evaluation_tool_specific_guidance(tools),
        _get_evaluation_multi_choice_guidance(multi_choice, tools),
    ]

    return "".join(section for section in sections if section)
