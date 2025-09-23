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

from fastapi import FastAPI
import os
from pyaml_env import parse_config
import json
from vss_ctx_rag.context_manager import ContextManager
from vss_ctx_rag.utils.ctx_rag_logger import logger
import asyncio
from .models import (
    AddModel,
    CallModel,
    ConfigModel,
    ResetModel,
    DCFileModel,
    SummaryModel,
    CompleteIngestionModel,
    ChatModel,
    resolve_env_vars,
)
from fastapi import APIRouter
import traceback
from vss_ctx_rag.models.state_models import SourceDocs
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp import FastMCP
from typing import Any
from copy import deepcopy


common_router = APIRouter()
data_ingest_router = APIRouter()
data_retrieval_router = APIRouter()
dev_router = APIRouter()

DEFAULT_CONFIG_PATH = "/app/config/config.yaml"


class AppState:
    def __init__(self):
        self.ctx_managers = {}  # UUID -> ContextManager mapping
        self.configs = {}  # UUID -> config mapping


app_state = AppState()


@common_router.post("/init")
async def init_context_manager(init_config: ConfigModel):
    try:
        if not init_config.uuid:
            return {"status": "error", "message": "UUID is required for initialization"}

        if init_config.uuid in app_state.ctx_managers:
            logger.warning(
                f"UUID {init_config.uuid} already initialized, please use /update_config to update the config"
            )
            return {
                "status": "error",
                "message": "UUID already initialized, please use /update_config to update the config",
            }

        config = {}
        init_ret = "Using empty config as base"

        # Apply overrides if provided
        if init_config.config_path:
            config = parse_config(init_config.config_path).copy()
            init_ret += f" with overrides from config path {init_config.config_path}"
        elif init_config.context_config:
            config = init_config.context_config.copy()
            config = resolve_env_vars(config)
            init_ret += " with context config overrides"
        else:
            config = parse_config(DEFAULT_CONFIG_PATH).copy()
            init_ret += (
                f" with overrides from default config path {DEFAULT_CONFIG_PATH}"
            )

        logger.info(f"UUID {init_config.uuid}: {init_ret}")

        config["context_manager"]["uuid"] = init_config.uuid

        app_state.ctx_managers[init_config.uuid] = ContextManager(config=config)
        app_state.configs[init_config.uuid] = config

        context_manager = app_state.ctx_managers[init_config.uuid]
        context_manager.reset(
            {
                "summarization": {"uuid": init_config.uuid},
                "retriever_function": {"uuid": init_config.uuid},
                "ingestion_function": {"uuid": init_config.uuid},
            }
        )

        return {
            "status": "success",
            "message": f"ContextManager initialized for UUID {init_config.uuid}: {init_ret}",
        }
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


def get_context_manager(uuid: str):
    if not uuid:
        return None, {"status": "error", "message": "UUID is required"}
    if uuid not in app_state.ctx_managers:
        return None, {
            "status": "error",
            "message": f"Context manager for UUID {uuid} not initialized, please call init",
        }
    return app_state.ctx_managers[uuid], None


@data_ingest_router.post("/add_doc")
async def add_doc(doc: AddModel):
    logger.info(f"UUID {doc.uuid}: add_doc called")
    ctx_mgr, error = get_context_manager(doc.uuid)
    if error:
        return error

    try:
        await asyncio.sleep(0.001)
        if "uuid" not in doc.doc_metadata:
            doc.doc_metadata["uuid"] = doc.uuid
        ctx_mgr.add_doc(doc.document, doc.doc_index, doc.doc_metadata)
        response = f"Added document {doc.doc_index} for UUID {doc.uuid}"
        logger.info(f"UUID {doc.uuid}: Successfully added document {doc.doc_index}")
        return {"status": "success", "result": response}
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@dev_router.post("/call")
async def call_endpoint(call_model: CallModel):
    logger.info(f"UUID {call_model.uuid}: call endpoint called")
    ctx_mgr, error = get_context_manager(call_model.uuid)
    if error:
        return error

    try:
        if "uuid" not in call_model.state:
            call_model.state["uuid"] = call_model.uuid
        result = ctx_mgr.call(call_model.state)
        if "error" in result:
            logger.error(
                f"UUID {call_model.uuid}: Call failed with error: {result['error']}"
            )
            return {"status": "error", "result": result}

        logger.info(f"UUID {call_model.uuid}: Call completed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@data_retrieval_router.post("/chat/completions")
async def chat_completions_endpoint(chat_model: ChatModel):
    logger.info(f"UUID {chat_model.uuid}: chat completions endpoint called")
    ctx_mgr, error = get_context_manager(chat_model.uuid)
    if error:
        return error

    try:
        question = "\n".join(
            [f"{msg.role}: {msg.content}" for msg in chat_model.messages]
        )

        current_config = deepcopy(app_state.configs.get(chat_model.uuid, {}))
        if (
            "functions" in current_config
            and "retriever_function" in current_config["functions"]
        ):
            llm_name = current_config["functions"]["retriever_function"]["tools"]["llm"]
            llm_config = deepcopy(current_config["tools"][llm_name])

            # Update LLM config with ChatModel parameters
            if chat_model.model:
                llm_config["params"]["model"] = chat_model.model
            if chat_model.base_url:
                llm_config["params"]["base_url"] = chat_model.base_url
            if chat_model.temperature is not None:
                llm_config["params"]["temperature"] = chat_model.temperature
            if chat_model.top_p is not None:
                llm_config["params"]["top_p"] = chat_model.top_p
            if chat_model.max_tokens is not None:
                llm_config["params"]["max_tokens"] = chat_model.max_tokens

            # Save the updated LLM config back to current_config
            current_config["tools"][llm_name] = llm_config

            ctx_mgr.configure(config=current_config)

            result = ctx_mgr.call(
                {"retriever_function": {"question": question, "uuid": chat_model.uuid}}
            )

        else:
            result = {"retriever_function": {"error": "Retriever function not found"}}

        base_response = {
            "id": f"chatcmpl-{chat_model.uuid}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": chat_model.model,
        }

        if "error" in result:
            logger.error(
                f"UUID {chat_model.uuid}: Call failed with error: {result['retriever_function']['error']}"
            )
            base_response["error"] = {
                "message": str(result["retriever_function"]["error"]),
                "type": "retriever_error",
                "code": "retriever_failed",
            }
            return base_response

        retriever_result = result.get("retriever_function", {})
        response_text = retriever_result.get("response", "")

        base_response["choices"] = [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ]

        logger.info(f"UUID {chat_model.uuid}: Chat completion completed successfully")
        return base_response

    except Exception as e:
        traceback.print_exc()
        print(e)
        return {
            "id": f"chatcmpl-{chat_model.uuid}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": chat_model.model,
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "exception_occurred",
            },
        }


@data_ingest_router.post("/complete_ingestion")
async def complete_ingestion(ingestion_model: CompleteIngestionModel):
    logger.info(f"UUID {ingestion_model.uuid}: complete_ingestion called")
    ctx_mgr, error = get_context_manager(ingestion_model.uuid)
    if error:
        return error

    try:
        ctx_mgr.call({"ingestion_function": {"uuid": ingestion_model.uuid}})
        logger.info(f"UUID {ingestion_model.uuid}: Ingestion completed successfully")
        return {
            "status": "success",
            "result": f"Ingestion complete for UUID {ingestion_model.uuid}",
        }
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@data_retrieval_router.post("/summary")
async def summary_endpoint(summary_model: SummaryModel):
    logger.info(f"UUID {summary_model.uuid}: summary endpoint called")
    ctx_mgr, error = get_context_manager(summary_model.uuid)
    if error:
        return error

    try:
        logger.debug(
            f"UUID {summary_model.uuid}: Summary request: {summary_model.model_dump()}"
        )
        result = ctx_mgr.call(summary_model.model_dump(exclude={"uuid"}))
        logger.info(f"UUID {summary_model.uuid}: Summary completed successfully")
        return {"status": "success", "result": result["summarization"]["result"]}
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@common_router.post("/update_config")
async def update_config(config_model: ConfigModel):
    logger.info(f"UUID {config_model.uuid}: update_config called")
    ctx_mgr, error = get_context_manager(config_model.uuid)
    if error:
        return error

    try:
        # Start with current config as base (or default if not exists)
        current_config = app_state.configs.get(config_model.uuid, {})
        config = current_config.copy()
        update_ret = "Using current config as base"

        # Apply overrides if provided
        if config_model.config_path:
            config = parse_config(config_model.config_path).copy()
            update_ret += f" with overrides from config path {config_model.config_path}"
        elif config_model.context_config:
            config = config_model.context_config.copy()
            config = resolve_env_vars(config)
            update_ret += " with context config overrides"
        else:
            update_ret += " (no overrides provided)"
            return {
                "status": "error",
                "message": "No overrides provided, please provide either config_path or context_config",
            }

        logger.info(f"UUID {config_model.uuid}: {update_ret}")

        config["context_manager"]["uuid"] = config_model.uuid
        ctx_mgr.configure(config=config)
        app_state.configs[config_model.uuid] = config
        return {
            "status": "success",
            "message": f"Configuration updated for UUID {config_model.uuid}: {update_ret}",
        }
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@common_router.post("/reset")
async def reset_context(reset_model: ResetModel):
    logger.info(f"UUID {reset_model.uuid}: reset called")
    ctx_mgr, error = get_context_manager(reset_model.uuid)
    if error:
        return error

    try:
        result = ctx_mgr.reset(reset_model.state)
        logger.info(f"UUID {reset_model.uuid}: Reset completed successfully")
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


@common_router.get("/health/live")
async def liveness_probe():
    """
    Liveness probe endpoint for Kubernetes.
    Returns whether the service is running and responding.
    """
    return {"status": "success", "message": "Service is alive"}


@common_router.get("/health/ready")
async def readiness_probe():
    """
    Readiness probe endpoint for Kubernetes.
    Returns whether the service is ready to handle requests.
    """
    try:
        if not app_state.ctx_managers:
            return {"status": "error", "message": "Context manager map is empty"}
        if not app_state.configs:
            return {"status": "error", "message": "Config map is empty"}
        return {"status": "success", "message": "Service is ready"}
    except Exception as e:
        logger.error(f"Readiness probe failed: {str(e)}")
        return {"status": "error", "message": "Service not ready"}


@dev_router.post("/add_doc_from_dc")
async def add_doc_from_dc(dc_file_model: DCFileModel):
    logger.info(f"UUID {dc_file_model.uuid}: add_doc_from_dc called")
    ctx_mgr, error = get_context_manager(dc_file_model.uuid)
    if error:
        return error

    try:
        data_list = []
        file_path = dc_file_model.dc_file_path
        with open(file_path, "r") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    formatted_dict = {
                        "vlm_response": data.get("vlm_response", ""),
                        "frame_times": data.get("frame_times", []),
                        "chunk": data.get("chunk", {}),
                        "streamId": data.get("chunk", {}).get("streamId", ""),
                        "chunkIdx": data.get("chunk", {}).get("chunkIdx", None),
                        "file": data.get("chunk", {}).get("file", ""),
                        "pts_offset_ns": 0,
                        "start_pts": data.get("chunk", {}).get("start_pts", None),
                        "end_pts": data.get("chunk", {}).get("end_pts", None),
                        "start_ntp": data.get("chunk", {}).get("start_ntp", ""),
                        "end_ntp": data.get("chunk", {}).get("end_ntp", ""),
                        "start_ntp_float": data.get("chunk", {}).get(
                            "start_ntp_float", None
                        ),
                        "end_ntp_float": data.get("chunk", {}).get(
                            "end_ntp_float", None
                        ),
                        "is_first": data.get("chunk", {}).get("is_first", False),
                        "is_last": data.get("chunk", {}).get("is_last", False),
                        "uuid": "",
                        "cv_meta": "[]",
                    }
                    data_list.append(formatted_dict)
                except json.JSONDecodeError as e:
                    logger.info(
                        f"Skipping invalid JSON line: {line[:100]}... Error: {e}"
                    )
        for vlm_chunk in data_list:
            doc_meta = {
                _key: _val
                for _key, _val in vlm_chunk.items()
                if _key != "vlm_response" and _key != "frame_times" and _key != "chunk"
            }
            ctx_mgr.add_doc(
                vlm_chunk["vlm_response"],
                doc_i=vlm_chunk["chunkIdx"],
                doc_meta=doc_meta,
            )

        ctx_mgr.call({"ingestion_function": {"post_process": True}})
        logger.info(
            f"UUID {dc_file_model.uuid}: Documents from DC file added successfully"
        )
        return {"status": "success", "message": "Documents added"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        return {"status": "error", "message": str(e)}


# ------------------------------------------------------------
# MCP server and tools
# ------------------------------------------------------------


mcp_server = FastMCP(name="CA-RAG retrieval MCP Server")


class CheckContextManagerMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        # Check if UUID is provided in the arguments
        arguments = context.request.params.get("arguments", {})
        uuid = arguments.get("uuid")

        if not uuid:
            return {
                "status": "error",
                "message": "UUID parameter is required for MCP tools",
            }

        if uuid not in app_state.ctx_managers:
            return {
                "status": "error",
                "message": f"Context manager for UUID {uuid} not initialized, please call /init",
            }

        return await call_next(context)


@mcp_server.tool()
def query(question: str, uuid: str) -> dict[str, Any]:
    """
    Query to ask complex questions about the videos to context manager which can internally use graphRAG or vectorRAG to answer.
    Args:
        question: The question to ask the context manager.
        uuid: The UUID of the context manager to use.
    Returns:
        A dictionary containing the response from the context manager with response, errors and other keys.
    """
    logger.info(
        f"UUID {uuid}: MCP query tool called with question: {question[:100]}..."
        if len(question) > 100
        else f"UUID {uuid}: MCP query tool called with question: {question}"
    )
    ctx_mgr = app_state.ctx_managers[uuid]  # Middleware already validates UUID exists
    call_params = {"question": question}

    result = ctx_mgr.call({"retriever_function": call_params})
    response = result["retriever_function"].get("response", None)
    error = result["retriever_function"].get("error", None)

    if error:
        logger.error(f"UUID {uuid}: MCP query tool failed with error: {error}")
    else:
        logger.info(f"UUID {uuid}: MCP query tool completed successfully")

    return {"response": response, "error": error}


@mcp_server.tool()
def find_event(keywords: str, uuid: str) -> list[SourceDocs]:
    """
    Find events in the videos based on the keywords. Useful for filtering events later based on the returned source docs as they contain metadata.
    Args:
        keywords: The keywords to search for in the videos.
        uuid: The UUID of the context manager to use.
    Returns:
        A list of source docs containing the events found in the videos.
    """
    logger.info(f"UUID {uuid}: MCP find_event tool called with keywords: {keywords}")
    ctx_mgr = app_state.ctx_managers[uuid]  # Middleware already validates UUID exists
    call_params = {"question": keywords}
    result = ctx_mgr.call({"retriever_function": call_params})
    source_docs = result["retriever_function"].get("source_docs", [])
    logger.info(
        f"UUID {uuid}: MCP find_event tool found {len(source_docs)} source docs"
    )
    return source_docs


@mcp_server.tool()
def find_object(object_name: str) -> list[SourceDocs]:
    """
    Returns events and the metadata of the event in the videos similar to the given object name. Useful for filtering events for the presence of a particular object.
    For example, used for finding particular occurances like "worker with yellow vest" or "red forklift".
    Args:
        object_name: The object name to search for in the videos.
    Returns:
        A list of source docs containing the events found in the videos. The SourceDocs contains the metadata of the event where the object was found in the videos.
    """
    call_params = {"question": object_name, "retriever_type": "entity"}
    result = app_state.ctx_mgr.call({"retriever_function": call_params})
    return result["retriever_function"].get("source_docs", [])


@mcp_server.tool()
def find_event_formatted(keywords: str, uuid: str) -> list[str]:
    """
    Find events in the videos based on the keywords. Useful for getting a list of the most relevant docs as formatted strings.
    Args:
        keywords: The keywords to search for in the videos.
        uuid: The UUID of the context manager to use.
    Returns:
        A list of formatted docs as strings containing the events found in the videos.
    """
    logger.info(
        f"UUID {uuid}: MCP find_event_formatted tool called with keywords: {keywords}"
    )
    ctx_mgr = app_state.ctx_managers[uuid]  # Middleware already validates UUID exists
    call_params = {"question": keywords}
    result = ctx_mgr.call({"retriever_function": call_params})
    formatted_docs = result["retriever_function"].get("formatted_docs", [])
    logger.info(
        f"UUID {uuid}: MCP find_event_formatted tool found {len(formatted_docs)} formatted docs"
    )
    return formatted_docs


@mcp_server.tool()
def summary_retriever(
    start_time: float | None = None,
    end_time: float | None = None,
    uuid: str | None = None,
) -> str:
    """
    Summary retriever to get a summary of the events in the videos between given start_time(in seconds) and end_time(in seconds) and uuid of the video.
    Provide UUID to summarize a particular stream.
    If you dont provide start_time and end_time, it will summarize the entire video.
    Args:
        start_time: The start time of the events to summarize.
        end_time: The end time of the events to summarize.
        uuid: The uuid of the video to summarize.
    Returns:
        A summary of the events in the videos.
    """
    call_params = {"start_time": start_time, "end_time": end_time, "uuid": uuid}
    result = app_state.ctx_mgr.call({"summary_retriever": call_params})
    return result["summary_retriever"].get("summary", "")


# ------------------------------------------------------------
# FastMCP + FastAPI servers config
# ------------------------------------------------------------

mcp_server.add_middleware(CheckContextManagerMiddleware())
mcp_app = mcp_server.http_app(path="/mcp")
app = FastAPI(
    title="Context Aware RAG Service", lifespan=mcp_app.router.lifespan_context
)


app.include_router(common_router)

if str(os.environ.get("VSS_CTX_RAG_ENABLE_RET")).lower() in ["true", "1"]:
    app.include_router(data_retrieval_router)
    app.mount("/", mcp_app)
else:
    app.include_router(data_ingest_router)
    if str(os.environ.get("VIA_CTX_RAG_ENABLE_DEV")).lower() in ["true", "1"]:
        app.include_router(dev_router)
