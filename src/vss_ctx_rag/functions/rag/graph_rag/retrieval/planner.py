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


from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode
from typing import Annotated, TypedDict, Literal, Optional, Dict
from langgraph.graph.message import add_messages
import json
import xml.etree.ElementTree as ET
import uuid
from langchain_core.runnables import RunnableConfig

from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.functions.rag.graph_rag.tools.graph_search_tool import (
    ChunkSearch,
    ChunkFilter,
    ChunkReader,
    EntitySearch,
    SubtitleSearch,
    SubtitleFilter,
    BFS,
    NextChunk,
)
from vss_ctx_rag.functions.rag.graph_rag.prompt import (
    create_thinking_prompt,
    create_response_prompt,
    create_evaluation_prompt,
)
from vss_ctx_rag.tools.image.image_fetcher import ImageFetcher
from vss_ctx_rag.utils.globals import DEFAULT_NUM_FRAMES_PER_CHUNK


# Enhanced state for iterative thinking ↔ execution
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    original_query: str
    current_plan: str
    execution_results: str
    iteration_count: int
    max_iterations: int
    thinking_complete: bool
    num_cameras: int  # Add num_cameras to the state
    video_length: dict
    chunk_size: float
    is_subtitle: bool


def get_agent(
    llm,
    vlm,
    graph_db,
    uuids: str,
    multi_channel: bool,
    multi_choice: bool,
    top_k: int,
    tools: list[str] = None,
    image_fetcher: ImageFetcher = None,
    num_frames_per_chunk: int = DEFAULT_NUM_FRAMES_PER_CHUNK,
    prompt_config: Optional[Dict] = None,
):
    """
    Create agent with dynamic or static prompts based on available tools.

    Args:
        llm: Language model instance
        vlm: Vision language model instance
        graph_db: Graph database instance
        uuids: UUID string
        top_k: Top K results to retrieve
        multi_channel: Whether multi-channel is enabled
        multi_choice: Whether multi-choice is enabled
        tools: List of tools to use
    """

    if not tools:
        tools = ["chunk_search", "chunk_filter", "entity_search"]

    tool_list = []
    for tool in tools:
        if tool == "chunk_search":
            tool_list.append(ChunkSearch(graph_db, uuids, multi_channel, top_k))
        elif tool == "chunk_filter":
            tool_list.append(ChunkFilter(graph_db, multi_channel, uuids))
        elif tool == "entity_search":
            tool_list.append(EntitySearch(graph_db, uuids, multi_channel, top_k))
        elif tool == "chunk_reader":
            tool_list.append(
                ChunkReader(
                    graph_db,
                    vlm,
                    uuids,
                    multi_channel,
                    image_fetcher,
                    num_frames_per_chunk,
                )
            )
        elif tool == "subtitle_search":
            tool_list.append(SubtitleSearch(graph_db, uuids, multi_channel, top_k))
        elif tool == "subtitle_filter":
            tool_list.append(SubtitleFilter(graph_db))
        elif tool == "bfs":
            tool_list.append(BFS(graph_db))
        elif tool == "next_chunk":
            tool_list.append(NextChunk(graph_db))

    if tool_list and any(tool.name == "SubtitleSearch" for tool in tool_list):
        thinking_prompt_content = (
            create_thinking_prompt(
                tools=tool_list,
                multi_choice=multi_choice,
                multi_channel=multi_channel,
            )
            if not (prompt_config and prompt_config["thinking_sys_msg_subtitle_prompt"])
            else prompt_config["thinking_sys_msg_subtitle_prompt"]
        )
    else:
        thinking_prompt_content = (
            create_thinking_prompt(
                tools=tool_list,
                multi_choice=multi_choice,
                multi_channel=multi_channel,
            )
            if not (prompt_config and prompt_config["thinking_sys_msg_prompt"])
            else prompt_config["thinking_sys_msg_prompt"]
        )

    evaluation_prompt_content = (
        create_evaluation_prompt(
            tools=tool_list,
            multi_choice=multi_choice,
        )
        if not (prompt_config and prompt_config["evaluation_guidance_prompt"])
        else prompt_config["evaluation_guidance_prompt"]
    )
    response_prompt_content = (
        create_response_prompt(
            multi_choice=multi_choice,
        )
        if not (prompt_config and prompt_config["response_sys_msg_prompt"])
        else prompt_config["response_sys_msg_prompt"]
    )

    logger.debug(f"Thinking prompt content: {thinking_prompt_content}")
    logger.debug(f"Evaluation prompt content: {evaluation_prompt_content}")
    logger.debug(f"Response prompt content: {response_prompt_content}")
    # RESPONSE AGENT - Clean user response formatting with correct config
    response_sys_msg = SystemMessage(content=response_prompt_content)

    # THINKING AGENT NODE
    def thinking_agent(state: AgentState):
        # THINKING AGENT - Pure reasoning and planning
        # Format the prompt with runtime num_cameras info
        if not multi_choice:
            num_cameras_info = (
                f"Note: There are {state['num_cameras']} cameras in the video."
                if multi_channel and state["num_cameras"] > 0
                else ""
            )

            video_length_info = (
                f"Note: The video length of each camera is as follows: {state['video_length']}"
                if multi_channel
                else f"Note: The video length is {state['video_length']} seconds."
            )
        else:
            video_length_info = ""
            num_cameras_info = ""

        formatted_prompt = thinking_prompt_content.format(
            num_cameras_info=num_cameras_info, video_length_info=video_length_info
        )
        thinking_sys_msg = SystemMessage(content=formatted_prompt)
        # First iteration - create initial plan
        if state["iteration_count"] == 0:
            messages = [
                thinking_sys_msg,
                HumanMessage(
                    content=f"User Query: {state['original_query']}\n\nCreate an execution plan to answer this query."
                ),
            ]
        else:
            # Subsequent iterations - evaluate results and plan next steps
            # Add strategic guidance for repetitive results
            evaluation_guidance = ""
            if state["iteration_count"] >= 2:
                guidance = evaluation_prompt_content
                evaluation_guidance = f"""IMPORTANT: This is iteration {state['iteration_count']}. {guidance}"""

            messages = [
                thinking_sys_msg,
                HumanMessage(
                    content=f"""User Query: {state['original_query']}

Previous Plan: {state['current_plan']}

Execution Results: {state['execution_results']}{evaluation_guidance}

Evaluate these results and determine if you need more information or if you can provide a complete answer. If more information is needed, create a new execution plan. If complete, respond with **COMPLETE**.
"""
                ),
            ]

        response = llm.invoke(messages)

        # Log thinking process
        logger.info(
            f"🤔 Thinking Agent (Iteration {state['iteration_count']}): {response.content}"
        )

        # Check if thinking is complete
        thinking_complete = "<answer>" in response.content

        return {
            "messages": [response],
            "current_plan": response.content,
            "thinking_complete": thinking_complete,
        }

    # TOOL EXECUTION NODE
    async def tool_node_with_logging(state: AgentState):
        last_message = state["messages"][-1]

        if hasattr(last_message, "tool_calls") and last_message.tool_calls:
            tool_node = ToolNode(tool_list)
            result = await tool_node.ainvoke(
                state,
                config=RunnableConfig(
                    configurable={
                        "chunk_size": state["chunk_size"],
                        "is_subtitle": state["is_subtitle"],
                    }
                ),
            )

            # Log and collect results
            execution_results = []
            for msg in result["messages"]:
                if hasattr(msg, "type") and msg.type == "tool":
                    logger.info(f"📋 Tool Result: {msg.content}")
                    execution_results.append(f"Tool: {msg.name}\nResult: {msg.content}")

            # Update execution results for thinking agent
            execution_results_str = "\n\n".join(execution_results)

            return {
                "messages": result["messages"],
                "execution_results": execution_results_str,
                "iteration_count": state["iteration_count"] + 1,
            }

        return {"messages": []}

    # EXECUTION AGENT NODE
    def execution_node(state: AgentState):
        thoughts = state["current_plan"]
        tool_calls = []

        def escape_xml_content(text):
            """Escape XML special characters in text content while preserving XML structure.

            Based on XML specification, these characters must be escaped in text content:
            & -> &amp;
            < -> &lt;
            > -> &gt;
            " -> &quot;
            ' -> &apos;
            """
            # First escape & to avoid double-escaping
            text = text.replace("&", "&amp;")
            # Then escape other special characters
            text = text.replace("<", "&lt;")
            text = text.replace(">", "&gt;")
            text = text.replace('"', "&quot;")
            text = text.replace("'", "&apos;")
            return text

        def unescape_xml_tags(text):
            """Unescape specific XML tags that we want to parse."""
            # Unescape execute block tags
            text = text.replace("&lt;execute&gt;", "<execute>")
            text = text.replace("&lt;/execute&gt;", "</execute>")

            # Unescape step tags
            text = text.replace("&lt;step&gt;", "<step>")
            text = text.replace("&lt;/step&gt;", "</step>")

            # Unescape tool tags
            text = text.replace("&lt;tool&gt;", "<tool>")
            text = text.replace("&lt;/tool&gt;", "</tool>")

            # Unescape input tags
            text = text.replace("&lt;input&gt;", "<input>")
            text = text.replace("&lt;/input&gt;", "</input>")

            # Unescape common input field tags
            text = text.replace("&lt;chunk_id&gt;", "<chunk_id>")
            text = text.replace("&lt;/chunk_id&gt;", "</chunk_id>")
            text = text.replace("&lt;query&gt;", "<query>")
            text = text.replace("&lt;/query&gt;", "</query>")
            text = text.replace("&lt;keywords&gt;", "<keywords>")
            text = text.replace("&lt;/keywords&gt;", "</keywords>")
            text = text.replace("&lt;start_time&gt;", "<start_time>")
            text = text.replace("&lt;/start_time&gt;", "</start_time>")
            text = text.replace("&lt;end_time&gt;", "<end_time>")
            text = text.replace("&lt;/end_time&gt;", "</end_time>")
            text = text.replace("&lt;range&gt;", "<range>")
            text = text.replace("&lt;/range&gt;", "</range>")
            text = text.replace("&lt;topk&gt;", "<topk>")
            text = text.replace("&lt;/topk&gt;", "</topk>")
            text = text.replace("&lt;event_description&gt;", "<event_description>")
            text = text.replace("&lt;/event_description&gt;", "</event_description>")
            text = text.replace("&lt;max_search_results&gt;", "<max_search_results>")
            text = text.replace("&lt;/max_search_results&gt;", "</max_search_results>")
            text = text.replace("&lt;camera_id&gt;", "<camera_id>")
            text = text.replace("&lt;/camera_id&gt;", "</camera_id>")

            return text

        # Tool name to class name mapping
        TOOL_NAME_MAPPING = {
            "chunk_search": "ChunkSearch",
            "entity_search": "EntitySearch",
            "chunk_filter": "ChunkFilter",
            "chunk_reader": "ChunkReader",
            "subtitle_search": "SubtitleSearch",
            "subtitle_filter": "SubtitleFilter",
            "bfs": "BFS",
            "next_chunk": "NextChunk",
        }

        def create_tool_call(tool_name: str, input_data: dict) -> dict:
            """Create a standardized tool call structure."""
            return {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    "name": TOOL_NAME_MAPPING[tool_name],
                    "arguments": json.dumps(input_data),
                },
            }

        def handle_chunk_reader_special_case(
            input_data: dict, input_ids: list, original_query: str
        ) -> list:
            """Handle the special case where chunk_reader needs multiple calls for multiple IDs."""
            calls = []
            for chunk_id in input_ids:
                chunk_data = input_data.copy()
                chunk_data["chunk_id"] = chunk_id
                chunk_data["query"] = original_query
                calls.append(create_tool_call("chunk_reader", chunk_data))
            return calls

        def handle_query_special_case(
            tool_name: str, input_data: dict, queries: list
        ) -> list:
            """Handle the special case where chunk_search needs multiple calls for multiple queries."""
            calls = []
            for query in queries:
                chunk_data = input_data.copy()
                chunk_data["query"] = query
                calls.append(create_tool_call(tool_name, chunk_data))
            return calls

        try:
            # Step 1: Escape all XML special characters in the content
            escaped_thoughts = escape_xml_content(thoughts)

            # Step 2: Unescape the specific XML tags we want to parse
            xml_ready_thoughts = unescape_xml_tags(escaped_thoughts)

            # Step 3: Parse the XML safely
            root = ET.fromstring(f"<root>{xml_ready_thoughts}</root>")

            for block in root.findall("execute"):
                tool_name = block.findtext("tool").strip().lower()
                input_elem = block.find("input")
                input_ids = None
                input_data = {child.tag: child.text for child in input_elem}
                logger.info(f"Parsed tool: {tool_name}, input_data: {input_data}")
                if "chunk_id" in input_data:
                    input_ids = input_data["chunk_id"].split(";")
                queries = []
                if "query" in input_data:
                    queries = input_data["query"].split(";")

                # Check if tool is supported
                if tool_name not in TOOL_NAME_MAPPING:
                    logger.warning(f"Unknown tool: {tool_name}")
                    continue

                # Handle special case for chunk_reader with multiple IDs
                if tool_name == "chunk_reader":
                    if input_ids:
                        if multi_choice:
                            query = state["original_query"]
                        else:
                            query = input_data["query"]
                        tool_calls.extend(
                            handle_chunk_reader_special_case(
                                input_data, input_ids, query
                            )
                        )
                    else:
                        if multi_choice:
                            query = state["original_query"]
                        else:
                            query = input_data["query"]
                        input_data["query"] = query
                        tool_calls.append(create_tool_call(tool_name, input_data))
                elif (
                    (tool_name == "chunk_search" and queries)
                    or (tool_name == "entity_search" and queries)
                    or (tool_name == "subtitle_search" and queries)
                ):
                    tool_calls.extend(
                        handle_query_special_case(tool_name, input_data, queries)
                    )
                else:
                    # Standard case - create single tool call
                    tool_calls.append(create_tool_call(tool_name, input_data))

        except ET.ParseError as e:
            logger.error(f"Error in tool parsing after XML escaping: {str(e)}")
            logger.error(f"Original thoughts (first 500 chars): {thoughts[:500]}")
            logger.error(
                f"XML-ready thoughts (first 500 chars): {xml_ready_thoughts[:500] if 'xml_ready_thoughts' in locals() else 'Not available'}"
            )
            # Return an error message that the thinking agent can process
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": f"XML parsing error: {str(e)}. Please reformat your execution plan with valid XML.",
                    }
                ]
            }
        except Exception as e:
            logger.error(f"Unexpected error in tool parsing: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            return {
                "messages": [
                    {
                        "role": "system",
                        "content": f"Unexpected error in parsing: {str(e)}. Please retry with a simpler execution plan.",
                    }
                ]
            }

        messages = []
        if tool_calls:
            logger.info(f"Tool calls {tool_calls}")
            messages.append(
                {"role": "assistant", "content": None, "tool_calls": tool_calls}
            )
        else:
            logger.info("Tools calls are empty")
        return {"messages": messages}

    # RESPONSE AGENT NODE
    def response_agent(state: AgentState):
        messages = [
            response_sys_msg,
            HumanMessage(
                content=f"""Original Query: {state['original_query']}

Final Analysis and Results: {state['current_plan']}

All Execution Results: {state['execution_results']}

Provide a clean, direct answer to the user's question based on this information.
"""
            ),
        ]

        response = llm.invoke(messages)
        logger.info(f"📤 Final Response: {response.content}")

        return {"messages": [response]}

    # ROUTING LOGIC
    def should_continue_thinking(
        state: AgentState,
    ) -> Literal["execute", "response", "max_iterations"]:
        # Check iteration limit
        if state["iteration_count"] >= state["max_iterations"]:
            logger.warning(f"Reached maximum iterations ({state['max_iterations']})")
            return "max_iterations"

        # Check if thinking is complete
        if state.get("thinking_complete", False):
            return "response"

        return "execute"

    def should_execute_tools(state: AgentState) -> Literal["tools", "think"]:
        last_message = state["messages"][-1] if state["messages"] else None

        # If execution agent returned tool calls, go to tools
        if (
            last_message
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
        ):
            return "tools"

        # Otherwise, return to thinking for evaluation
        return "think"

    # Initialize state
    def initialize_state(state: AgentState):
        # Set original query from first human message
        original_query = ""
        for msg in state["messages"]:
            if hasattr(msg, "type") and msg.type == "human":
                original_query = msg.content
                break

        return {
            "original_query": original_query,
            "current_plan": "",
            "execution_results": "",
            "iteration_count": 0,
            "max_iterations": 20,  # Prevent infinite loops
            "thinking_complete": False,
            "num_cameras": state.get(
                "num_cameras", 1
            ),  # Initialize num_cameras from incoming state
            "video_length": state.get(
                "video_length", {}
            ),  # Initialize video_length from incoming state
            "chunk_size": state.get(
                "chunk_size", {}
            ),  # Initialize chunk_size from incoming state
            "is_subtitle": state.get(
                "is_subtitle", False
            ),  # Initialize is_subtitle from incoming state
        }

    # BUILD GRAPH
    builder = StateGraph(AgentState)

    # Add nodes
    builder.add_node("initialize", initialize_state)
    builder.add_node("think", thinking_agent)
    builder.add_node("execute", execution_node)
    builder.add_node("tools", tool_node_with_logging)
    builder.add_node("response", response_agent)

    # Add edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "think")

    # Thinking agent decides next step
    builder.add_conditional_edges(
        "think",
        should_continue_thinking,
        {"execute": "execute", "response": "response", "max_iterations": "response"},
    )

    # Execution agent either calls tools or returns to thinking
    builder.add_conditional_edges(
        "execute", should_execute_tools, {"tools": "tools", "think": "think"}
    )

    # After tools, always return to thinking for evaluation
    builder.add_edge("tools", "think")
    builder.add_edge("response", "__end__")

    return builder.compile()
