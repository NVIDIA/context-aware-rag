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

import asyncio
from pydantic import BaseModel, Field
from typing import Any, Optional, Type, Dict, List, ClassVar
import math
from langchain_core.runnables import RunnableConfig
import os
from vss_ctx_rag.functions.rag.graph_rag.prompt import PromptCapableTool
from vss_ctx_rag.tools.storage.graph_storage_tool import GraphStorageTool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.tools.image.image_fetcher import ImageFetcher


def get_entities(
    graph_db: GraphStorageTool,
    query: str,
    uuid: str = "default",
    multi_channel: bool = False,
    top_k: int = 5,
):
    formatted_docs = []
    if query:
        formatted_docs, _ = graph_db.retrieve_documents(
            query,
            uuid=uuid,
            multi_channel=multi_channel,
            top_k=top_k,
            retriever="planner_entity",
        )
    return formatted_docs


def get_filtered_chunks(
    graph_db: GraphStorageTool,
    min_start_time: Optional[str] = None,
    max_end_time: Optional[str] = None,
    camera_id: Optional[str] = None,
    uuid: Optional[str] = None,
    chunk_size: Optional[Dict[str, float]] = None,
):
    if not chunk_size:
        chunk_size = 10.0  # default value
    elif camera_id and chunk_size and camera_id in chunk_size:
        chunk_size = chunk_size[camera_id]
    else:
        chunk_size = chunk_size.get("", 10.0) if chunk_size else 10.0
    min_start_val = (
        round_down_to_nearest_chunk_size(float(min_start_time), chunk_size)
        if min_start_time
        else None
    )
    max_end_val = (
        round_up_to_nearest_chunk_size(float(max_end_time), chunk_size)
        if max_end_time
        else None
    )

    values = graph_db.filter_chunks(
        min_start_time=min_start_val,
        max_end_time=max_end_val,
        camera_id=camera_id,
        uuid=uuid,
    )
    return values


def get_chunks(
    graph_db: GraphStorageTool,
    text_search: Optional[str] = None,
    uuid: str = "default",
    multi_channel: bool = False,
    top_k: int = 5,
):
    formatted_docs = []
    if text_search:
        formatted_docs, _ = graph_db.retrieve_documents(
            text_search,
            uuid=uuid,
            multi_channel=multi_channel,
            top_k=top_k,
            retriever="planner_chunk",
        )
    return formatted_docs


def get_bfs(
    graph_db: GraphStorageTool,
    node_id: str,
):
    connected_info = graph_db.get_neighbors(int(node_id))
    return connected_info


class ChunkFilterInput(BaseModel):
    range: str = Field(
        description="chunk start time and end time in seconds as a numeric value (e.g., '150.0:155.0')"
    )
    camera_id: Optional[str] = Field(
        None, description="The camera id to filter the chunks by"
    )


class ChunkFilter(PromptCapableTool):
    name: str = "ChunkFilter"
    description: str = "Use for retrieving chunks based on temporal range and camera id"
    args_schema: Type[BaseModel] = ChunkFilterInput
    graph_db: GraphStorageTool
    multi_channel: bool
    uuids: str

    def __init__(self, graph_db: GraphStorageTool, multi_channel: bool, uuids: str):
        super().__init__(graph_db=graph_db, multi_channel=multi_channel, uuids=uuids)

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for ChunkFilter."""
        return {
            "xml_format": """#### Query Formats:

Question about specific camera/video and time range:
```
<execute>
  <step>1</step>
  <tool>chunk_filter</tool>
  <input>
    <range>start_time:end_time</range>
    <camera_id>camera_X</camera_id>
  </input>
</execute>
```

Question about time range:
<execute>
  <step>1</step>
  <tool>chunk_filter</tool>
  <input>
    <range>start_time:end_time</range>
  </input>
</execute>
```""",
            "description": """- If the question mentions a specific timestamp or time, you must convert it to seconds as numeric values.
- **CRITICAL**: The range format must be <start_seconds>:<end_seconds> using ONLY numeric values in seconds.
- **DO NOT use time format like HH:MM:SS**. Convert all times to total seconds first.
- **IMPORTANT**: For camera_id, always use the format "camera_X" or "video_X" where X is the camera/video number (e.g., camera_1/video_1, camera_2/video_2, camera_3/video_3, camera_4/video_4, etc.) Mention the camera_id only when the question is about a specific camera/video.

**Time Conversion Examples:**
  - "What happens at 00:05?" (5 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>5:15</range></input></execute>`
  - "What happens at 2:15?" (2 minutes 15 seconds = 135 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>135:145</range></input></execute>`
  - "Describe the action in the first minute." (0 to 60 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>0:60</range></input></execute>`
  - "Events at 1:30:45" (1 hour 30 min 45 sec = 5445 seconds) -> Query `<execute><step>1</step><tool>chunk_filter</tool><input><range>5445:5455</range></input></execute>`""",
            "rules": "",
        }

    def _run(
        self,
        range: str,
        camera_id: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Use the tool."""
        chunk_size = config["configurable"].get("chunk_size")
        if range == "null":
            min_start_time = None
            max_end_time = None
        else:
            if ":" not in range:
                raise ValueError("Range must be in format 'start:end' or 'null'")
            min_start_time, max_end_time = range.split(":")

        return get_filtered_chunks(
            self.graph_db,
            min_start_time=min_start_time,
            max_end_time=max_end_time,
            camera_id=camera_id,
            uuid=self.uuids if not self.multi_channel else None,
            chunk_size=chunk_size,
        )


class ChunkSearchInput(BaseModel):
    query: Optional[str] = Field(
        None,
        description="Semantic search for chunks that semantically match the query.",
    )
    topk: Optional[int] = Field(5, description="Top k search results.")


class ChunkSearch(PromptCapableTool):
    name: str = "ChunkSearch"
    description: str = "Use for general information retrieval and finding events in the video based on what happened. "
    args_schema: Type[BaseModel] = ChunkSearchInput
    graph_db: GraphStorageTool
    uuid: str
    multi_channel: bool
    top_k: int

    def __init__(
        self, graph_db: GraphStorageTool, uuid: str, multi_channel: bool, top_k: int
    ):
        super().__init__(
            graph_db=graph_db, uuid=uuid, multi_channel=multi_channel, top_k=top_k
        )

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for ChunkSearch."""
        return {
            "xml_format": """#### Query Formats:

## Single Query
```
<execute>
  <step>1</step>
  <tool>chunk_search</tool>
  <input>
    <query>your_question</query>
    <topk>10</topk>
  </input>
</execute>
```

## Multiple Query
```
<execute>
  <step>1</step>
  <tool>chunk_search</tool>
  <input>
    <query>your_question;your_question;your_question</query>
    <topk>10</topk>
  </input>
</execute>
```""",
            "description": """- Returns a ranked list of chunks, with the most relevant results at the top. For example, given the list [d, g, a, e], chunk d is the most relevant, followed by g, and so on.
- Assign topk=15 for counting problem, assign lower topk=8 for other problem
- Try to provide diverse search queries to ensure comprehensive result(for example, you can add the options into queries).
- You must generate a question for **every chunk returned by the chunk search** — do not miss any one!!!!!
- The chunk search cannot handle queries related to the global video timeline, because the original temporal signal is lost after all video chunks are split. If a question involves specific video timing, you need to boldly hypothesize the possible time range and then carefully verify each candidate chunk to locate the correct answer.""",
            "rules": "",
        }

    @classmethod
    def get_dynamic_rules(cls, tools: List[Any], config=None) -> str:
        """Get dynamic rules for ChunkSearch based on configuration."""
        rules = []

        if config and config.multi_choice:
            rules.append("**Multi-Choice Rules:**")
            rules.append(
                "- Never guess the answer, question about every choice, question about every chunk retrieved by the chunk_search!!!!"
            )
            rules.append(
                "- The chunk_search may make mistakes, and the chunk_reader is more accurate than the chunk_search. If the chunk_search retrieves a chunk but the chunk_reader indicates that the chunk is irrelevant to the current query, the result from the chunk_reader should be trusted."
            )

        return "\n".join(rules) if rules else ""

    def _run(
        self,
        query: Optional[str] = None,
        topk: Optional[int] = 5,
    ) -> str:
        """Use the tool."""

        return get_chunks(
            self.graph_db,
            query,
            self.uuid,
            self.multi_channel,
            topk,
        )


class EntitySearchInput(BaseModel):
    query: str = Field(
        description="Search query to find entities similar to these keywords or concepts"
    )


class EntitySearch(PromptCapableTool):
    name: str = "EntitySearch"
    description: str = (
        "Use to find specific entities (people, objects, locations) in the video and the chunks where they appear. "
        "Best for questions about specific entities like 'show me scenes with worker X' or 'where is object Y?'"
    )
    args_schema: Type[BaseModel] = EntitySearchInput
    graph_db: GraphStorageTool
    uuid: str
    multi_channel: bool
    top_k: int

    def __init__(
        self, graph_db: GraphStorageTool, uuid: str, multi_channel: bool, top_k: int
    ):
        super().__init__(
            graph_db=graph_db, uuid=uuid, multi_channel=multi_channel, top_k=top_k
        )

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for EntitySearch."""
        return {
            "xml_format": """#### Query Formats:
```
<execute>
  <step>1</step>
  <tool>entity_search</tool>
  <input>
    <query>your_question</query>
  </input>
</execute>
```""",
            "description": """- Returns a ranked list of entities, with the most relevant results at the top. For example, given the list [a, b, c, d, e], entity a is the most relevant, followed by b, and so on.
- Best for finding specific people, objects, or locations in video content
- Use when you need to track or identify particular entities across video segments""",
            "rules": "",
        }

    def _run(
        self,
        query: str,
    ) -> str:
        """Use the tool."""
        return get_entities(
            self.graph_db,
            query,
            self.uuid,
            self.multi_channel,
            self.top_k,
        )


class BFSInput(BaseModel):
    node_id: int = Field(description="The id of the node to start the BFS from")


class BFS(PromptCapableTool):
    name: str = "BFS"
    description: str = "Use to find the 1 hop connections from a given node"
    args_schema: Type[BaseModel] = BFSInput
    graph_db: GraphStorageTool

    def __init__(self, graph_db: GraphStorageTool):
        super().__init__(graph_db=graph_db)

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for BFS."""
        return {
            "xml_format": """#### Query Formats:
```
<execute>
  <step>1</step>
  <tool>bfs</tool>
  <input>
    <node_id>node_id_number</node_id>
  </input>
</execute>
```""",
            "description": """- Use to explore graph connections from a specific node
- Returns all nodes connected to the given node (1-hop neighbors)
- Useful for understanding relationships between entities and chunks""",
            "rules": "",
        }

    def _run(
        self,
        node_id: int,
    ) -> str:
        """Use the tool."""
        return get_bfs(
            self.graph_db,
            node_id,
        )


def get_next_chunk(
    graph_db: GraphStorageTool,
    chunk_id: int,
    number_of_hops: int = 1,
):
    result = graph_db.get_next_chunks(chunk_id=chunk_id, number_of_hops=number_of_hops)
    return result.get("connected_chunk")


class NextChunkInput(BaseModel):
    chunk_id: int = Field(description="The id of the chunk to find the next chunk from")
    number_of_hops: int = Field(description="The number of next chunks to find")


class NextChunk(PromptCapableTool):
    name: str = "NextChunk"
    description: str = "Use to find the next chunk in the video"
    args_schema: Type[BaseModel] = NextChunkInput
    graph_db: GraphStorageTool

    def __init__(self, graph_db: GraphStorageTool):
        super().__init__(graph_db=graph_db)

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for NextChunk."""
        return {
            "xml_format": """#### Query Formats:
```
<execute>
  <step>1</step>
  <tool>next_chunk</tool>
  <input>
    <chunk_id>chunk_id_number</chunk_id>
    <number_of_hops>1</number_of_hops>
  </input>
</execute>
```""",
            "description": """- Use to find chunks that come sequentially after a given chunk
- Useful for temporal analysis and understanding video sequences
- number_of_hops determines how many chunks ahead to retrieve""",
            "rules": "",
        }

    def _run(
        self,
        chunk_id: int,
        number_of_hops: int = 1,
    ) -> str:
        """Use the tool."""
        return get_next_chunk(
            self.graph_db,
            chunk_id,
            number_of_hops,
        )


class ChunkReaderInput(BaseModel):
    query: str = Field(description="Question to ask about the image of the chunk")
    chunk_id: Optional[str] = Field(None, description="Chunk ID to fetch images from")
    start_time: Optional[str] = Field(
        None, description="Start time of the chunk to fetch images from"
    )
    end_time: Optional[str] = Field(
        None, description="End time of the chunk to fetch images from"
    )


def round_up_to_nearest_chunk_size(number, chunk_size):
    """
    Round a number up to the next multiple of chunk_size.

    Args:
        number (int or float): The number to round

    Returns:
        int: The number rounded up to the next multiple of 10
    """
    # Round up to the next multiple of 10
    # Special case: if number is already a multiple of 10, keep it
    if number % chunk_size == 0:
        return int(number)
    return math.ceil(number / chunk_size) * chunk_size


def round_down_to_nearest_chunk_size(number, chunk_size):
    """
    Round a number down to the previous multiple of chunk_size.

    Args:
        number (int or float): The number to round

    Returns:
        int: The number rounded down to the previous multiple of 10
    """
    # Round down to the previous multiple of 10
    # Special case: if number is already a multiple of 10, keep it
    if number % chunk_size == 0:
        return int(number)
    return math.floor(number / chunk_size) * chunk_size


class ChunkReader(PromptCapableTool):
    name: str = "ChunkReader"
    prompt_name: ClassVar[str] = "Chunk Reader ⭐ **CRITICAL FOR VISUAL QUESTIONS** ⭐"
    description: str = (
        "Use to ask questions about images associated with specific chunk IDs."
        "The tool will fetch images related to the chunk and use an LLM to answer questions about them. "
        "Best for questions about visual content like 'what is happening in this scene?' or 'what objects are visible?'"
    )
    args_schema: Type[BaseModel] = ChunkReaderInput
    graph_db: GraphStorageTool
    chat_llm: Any
    image_fetcher: ImageFetcher
    uuid: str
    multi_channel: bool
    num_frames_per_chunk: int

    def __init__(
        self,
        graph_db: GraphStorageTool,
        chat_llm: Any,
        uuid: str = "default",
        multi_channel: bool = False,
        image_fetcher: ImageFetcher = None,
        num_frames_per_chunk: int = 30,
    ):
        super().__init__(
            graph_db=graph_db,
            chat_llm=chat_llm,
            image_fetcher=image_fetcher,
            uuid=uuid,
            multi_channel=multi_channel,
            num_frames_per_chunk=num_frames_per_chunk,
        )

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for ChunkReader."""
        return {
            "xml_format": """

#### **VISUAL ANALYSIS PRIORITY RULE:**
🚨 **When questions ask about VISUAL DETAILS (colors, clothing, appearance, objects, faces, etc.) and text descriptions from other tools don't provide this information, IMMEDIATELY use ChunkReader to visually analyze the relevant time periods. Do NOT continue searching other time periods until you've visually examined the first relevant occurrence.**

#### **SPECIAL RULE FOR VISUAL DETAIL QUESTIONS:**
⚠️ If question asks about visual characteristics and you find relevant content but text doesn't provide the visual details:
- STOP immediately
- Use ChunkReader on that exact time period
- Do NOT search other content first
- Examples:
  • Person mentioned but no appearance details → ChunkReader(time_range, "Describe the person's appearance")
  • Object mentioned but no visual details → ChunkReader(time_range, "Describe the object's characteristics")
  • Action mentioned but details unclear → ChunkReader(time_range, "What exactly is happening in this scene?")

#### **Common Visual Detail Questions:**
- Colors of clothing, objects, or backgrounds
- Facial expressions or physical appearance
- Object types, shapes, or characteristics
- Actions or poses that require visual confirmation
- Spatial relationships between objects/people

#### Query Formats:
#### *Single Chunk Query*
```
<execute>
  <step>1</step>
  <tool>chunk_reader</tool>
  <input>
    <start_time>1</start_time>
    <end_time>10</end_time>
    <query>your_question</query>
  </input>
</execute>

<execute>
  <step>1</step>
  <tool>chunk_reader</tool>
  <input>
    <chunk_id>1</chunk_id>
    <query>your_question</query>
  </input>
</execute>
```

#### *Sequential Chunks Query*
```
<execute>
  <step>2</step>
  <tool>chunk_reader</tool>
  <input>
    <chunk_id>N;N+1</chunk_id>
    <query>your_question</query>
  </input>
</execute>
```""",
            "description": """- Allows asking questions about the video chunks returned by the ChunkSearch or ChunkFilter.
- If the question mentions a specific timestamp, use the Chunkfilter tool to get the chunk ids for the given timerange and then ask the target question on the corresponding video chunk returned by the ChunkFilter tool.
- If the question is about some entity or scene then use the ChunkSearch tool to get the chunk ids for the specified entity or scene and then ask the target question on the corresponding video chunk returned by the ChunkSearch or EntitySearch tool.""",
            "rules": """- Only *temporally adjacent chunks* supported, so you must first order all chunks FROM SMALLEST TO LARGEST and then concatenate those that are adjacent in time. (e.g. N;N+1 are temporally adjacent chunks, but N;N+2 are not.)\n
- Max 2 chunks per query* (split longer sequences into multiple 2-chunk batches).

- Important Notes:
  - 🚨 **CRITICAL RULE FOR VISUAL QUESTIONS:** If ANY question asks about visual details AND you find relevant content but the text description doesn't specify the visual information needed, you MUST immediately use ChunkReader on that time period. Do NOT continue to other content first!
  - You should read every retrieved chunk without any omission!!!!!
  - If the scene mentioned in the question has been successfully verified by the chunk reader and occurs in chunk N, and the question asks about events before or after that scene, you should scan accordingly targeting chunk N-1 and N (for "before"), or chunk N and N+1 (for "after").
  - For counting/order problems, the question should follow this format "For questions asking whether a specific action occurs, you should carefully examine each frame — if even a single frame contains the relevant action, it should be considered as having occurred. The question is: is there xxx?"
  - For anomaly detection, don't concate chunk and raise single chunk query.
  - For anomaly detection, provide all the candidate options in each question!!
  - For anomaly detection, you may concatenate up to 10 sequential video chunk, including the retrieved chunks and its neighboring chunks, to obtain a comprehensive overview of the event.
  - If ChunkReader has returned "No images found for the specified chunk ID." for previous calls then *do not use" use ChunkReader tool for next iterations.""",
        }

    @classmethod
    def get_dynamic_rules(cls, tools: List[Any], config=None) -> str:
        """Get dynamic rules for ChunkReader based on configuration."""
        rules = []

        if config and not config.multi_choice:
            rules.append(
                "- SMART USAGE: Before using ChunkReader, evaluate if the query can be answered with confidence from the retrieved chunk metadata alone. Only use ChunkReader for visual verification when necessary."
            )

        if tools and any("SubtitleSearch" in tool.name for tool in tools):
            rules.append(
                "The video is segmented into chunks, and you can query them by the start_time and end_time returned by SubtitleSearch tool."
            )
        elif tools and any("ChunkSearch" in tool.name for tool in tools):
            rules.append(
                "The video is segmented into chunks, and you can query them by the their chunk id returned by ChunkSearch tool."
                "You may also query multiple consecutive chunks by concatenating their numbers (e.g., `112:115;113:116`)."
            )

        return "\n".join(rules) if rules else ""

    def _run(
        self,
        query: str,
        chunk_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Synchronous version - delegates to async implementation."""
        import asyncio

        try:
            # Check if we're in an event loop
            asyncio.get_running_loop()
            # If we are, we can't use asyncio.run(), so we'll use the async version
            raise RuntimeError("Use _arun instead when in async context")
        except RuntimeError as e:
            if "Use _arun instead" in str(e):
                # This was our manually raised error - re-raise it
                raise
            # This was from get_running_loop() - no event loop exists, safe to use asyncio.run()
            return asyncio.run(
                self._arun(query, chunk_id, start_time, end_time, config)
            )

    async def _arun(
        self,
        query: str,
        chunk_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        config: RunnableConfig = None,
    ) -> str:
        """Use the tool to answer questions about images associated with a specific chunk ID."""
        camera_id = ""
        is_subtitle = config["configurable"].get("is_subtitle", False)
        subtitles = []
        pass_subtitle = os.environ.get("PASS_SUBTITLES_TO_VLM", "false").lower() in [
            "true",
            "1",
            "yes",
            "on",
        ]
        try:
            if chunk_id:
                asset_dir = self.graph_db.get_chunk_asset_dir(chunk_id)
                time_range = self.graph_db.get_chunk_time_range(chunk_id)
                camera_id = self.graph_db.get_chunk_camera_id(chunk_id)
                if time_range:
                    start_time, end_time = time_range
                else:
                    return "No images found for the specified chunk ID."
                result = [{"asset_dir": asset_dir}] if asset_dir else []
            elif start_time and end_time:
                chunk_size = config["configurable"].get("chunk_size", 10)
                if camera_id:
                    chunk_size = chunk_size[camera_id]
                else:
                    chunk_size = chunk_size[""] if chunk_size else 10.0
                start_time = round_down_to_nearest_chunk_size(
                    float(start_time), chunk_size
                )
                end_time = round_up_to_nearest_chunk_size(float(end_time), chunk_size)
                dirs = self.graph_db.get_asset_dirs_by_time_range(start_time, end_time)
                if is_subtitle and pass_subtitle:
                    subtitles = self.graph_db.filter_subtitles_by_time_range(
                        start_time, end_time
                    )
                result = [{"asset_dir": d} for d in dirs]
            else:
                result = []

            if not result:
                return "No images found for the specified chunk ID."

            responses: List[Any] = []
            tasks: List[Any] = []
            for r in result:
                asset_dir = r["asset_dir"]
                logger.debug(f"asset_dir: {asset_dir}")
                if not asset_dir:
                    logger.error("No image assets found for the specified chunk ID.")
                    continue
                # Get images from the asset directory
                image_list_base64 = self.image_fetcher.get_image_base64(
                    asset_dir, self.num_frames_per_chunk
                )
                if not image_list_base64:
                    logger.error(
                        "Failed to retrieve images for the specified chunk ID."
                    )
                    continue

                # Prepare image messages for the LLM
                image_message_list = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "chunk_id": chunk_id,
                        },
                    }
                    for img in image_list_base64
                ]
                logger.debug(f"image_message_list: {len(image_message_list)}")
                if image_message_list:
                    # Prepare the prompt for the LLM
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an AI assistant that answers questions about images. "
                                "Analyze the provided images carefully and answer the question based on what you can see. "
                                "If you cannot determine the answer from the images, say so clearly. "
                                "Be specific and descriptive in your answers."
                                f"Subtitles for the timerange are: {subtitles}"
                                if is_subtitle and pass_subtitle
                                else ""
                            ),
                        },
                        {
                            "role": "user",
                            "content": [
                                *image_message_list,
                                {"type": "text", "text": f"Question: {query}"},
                            ],
                        },
                    ]
                    tasks.append(
                        asyncio.create_task(
                            self.chat_llm.ainvoke(messages), name="ChunkReader"
                        )
                    )
                else:
                    responses.append("No images found for the specified chunk ID.")
            responses = await asyncio.gather(*tasks)
            return {
                "query": query,
                "response": [response.content for response in responses],
                "start_time": start_time,
                "end_time": end_time,
                "camera_id": camera_id,
            }

        except ValueError:
            return "Invalid chunk ID format. Please provide a valid chunk ID."
        except Exception as e:
            logger.error(f"Error in ImageQnA tool: {str(e)}")
            import traceback

            logger.error(f"Error in ImageQnA tool: {traceback.format_exc()}")
            return f"An error occurred while processing your request: {str(e)}"


def get_subtitles(
    graph_db: GraphStorageTool,
    query: str,
    uuid: str,
    multi_channel: bool,
    topk: int,
):
    formatted_docs, _ = graph_db.retrieve_documents(
        query, uuid=uuid, multi_channel=multi_channel, top_k=topk, retriever="subtitle"
    )
    return formatted_docs


class SubtitleSearchInput(BaseModel):
    query: str = Field(
        description="Search query to find subtitles that semantically match the query."
    )
    topk: int = Field(description="Top k search results.")


class SubtitleSearch(PromptCapableTool):
    name: str = "SubtitleSearch"
    prompt_name: ClassVar[str] = (
        "Subtitle Search ⭐ **ALWAYS USE FIRST FOR SUBTITLE QUESTIONS** ⭐"
    )
    description: str = "Use to find subtitles that semantically match the query."
    args_schema: Type[BaseModel] = SubtitleSearchInput
    graph_db: GraphStorageTool
    uuid: str
    multi_channel: bool
    top_k: int

    def __init__(
        self, graph_db: GraphStorageTool, uuid: str, multi_channel: bool, top_k: int
    ):
        super().__init__(
            graph_db=graph_db, uuid=uuid, multi_channel=multi_channel, top_k=top_k
        )

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for SubtitleSearch."""
        return {
            "xml_format": """
## Single Query
```
<execute>
  <step>1</step>
  <tool>subtitle_search</tool>
  <input>
    <query>search_query</query>
    <topk>5_to_10</topk>
  </input>
</execute>
```

## Multiple Query
```
<execute>
  <step>1</step>
  <tool>subtitle_search</tool>
  <input>
    <query>search_query;search_query;search_query</query>
    <topk>5_to_10</topk>
  </input>
</execute>
```""",
            "description": """Note: topk is the integer value between 1 and 10.
- Returns the semantic matching subtitles and corresponding time range.
- Optimal Use Cases:
    - Each subtitle have a corresponding time range, so you can use the subtitle_search to locate the time range and then use chunk_reader to get the visual information of corresponding video clips.
    - For the questions related to audio, such as someone's opinion on something or the content of an argument, you should use the subtitle retriever!!!!""",
            "rules": """- 🚨 **CRITICAL SUBTITLE-FIRST STRATEGY**: For ANY question involving spoken content, audio, dialogue, conversations, opinions, arguments, or verbal interactions - **ALWAYS start with subtitle_search as your FIRST tool call**.

- **Mandatory Use Cases (Use subtitle_search FIRST):**
    - Questions about what someone said, mentioned, or discussed
    - Questions about opinions, arguments, conversations, dialogue
    - Questions about audio content, speech, or verbal interactions
    - Questions about specific quotes, statements, or verbal responses
    - Questions asking "what did X say about Y?" or "what was mentioned about Z?"
    - Any question where the answer likely comes from spoken/audio content

- **How it Works:**
    - Returns semantic matching subtitles with corresponding time ranges
    - Each subtitle has a time range that you can then use with chunk_reader for visual context
    - Provides the most accurate retrieval for speech-based questions

- **Workflow for Subtitle Questions:**
    1. **FIRST**: Use subtitle_search to find relevant spoken content
    2. **THEN**: Use chunk_reader with the time ranges from subtitles for visual verification if needed
    3. **AVOID**: Starting with chunk_search for questions about spoken content
""",
        }

    def _run(
        self,
        query: str,
        topk: int = 5,
    ) -> str:
        """Use the tool."""
        return get_subtitles(self.graph_db, query, self.uuid, self.multi_channel, topk)


def get_filtered_subtitles(
    graph_db: GraphStorageTool,
    start_time: str = None,
    end_time: str = None,
    chunk_size: Dict[str, float] = None,
):
    chunk_size = chunk_size[""] if chunk_size else 10.0
    start_time = round_down_to_nearest_chunk_size(float(start_time), chunk_size)
    end_time = round_up_to_nearest_chunk_size(float(end_time), chunk_size)
    values = graph_db.filter_subtitles_by_time_range(start_time, end_time)
    return values


class SubtitleFilterInput(BaseModel):
    range: str = Field(
        description="subtitle start time and end time in seconds as a numeric value (e.g., '150.0:155.0')"
    )


class SubtitleFilter(PromptCapableTool):
    name: str = "SubtitleFilter"
    prompt_name: ClassVar[str] = (
        "Subtitle Filter ⭐ **CRITICAL FOR SUBTITLE QUESTIONS** ⭐"
    )
    description: str = "Use for retrieving subtitles based on temporal range"
    args_schema: Type[BaseModel] = SubtitleFilterInput
    graph_db: GraphStorageTool

    def __init__(self, graph_db: GraphStorageTool):
        super().__init__(graph_db=graph_db)

    @classmethod
    def get_prompt_template_info(cls) -> Dict[str, str]:
        """Return prompt template information for SubtitleFilter."""
        return {
            "xml_format": """```
#### Query Formats:

<execute>
    <step>1</step>
    <tool>subtitle_filter</tool>
    <input>
        <range>150.0:155.0</range>
    </input>
</execute>
            ```""",
            "description": """
- Returns the subtitles that are within the specified time range.""",
            "rules": """
- You should only use the subtitle_filter tool if the question is about a specific time range and subtitle search.""",
        }

    def _run(
        self,
        range: str,
        config: RunnableConfig = None,
    ) -> str:
        """Use the tool."""

        start_time, end_time = range.split(":")

        if not start_time or not end_time:
            raise ValueError(
                "Both start_time and end_time must be provided and non-empty."
            )

        return get_filtered_subtitles(
            self.graph_db,
            start_time,
            end_time,
            config["configurable"].get("chunk_size"),
        )
