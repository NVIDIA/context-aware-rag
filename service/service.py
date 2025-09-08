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

from fastapi import FastAPI, HTTPException
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
)
from fastapi import APIRouter
import traceback
from .globals import DEFAULT_CONFIG_PATH
from vss_ctx_rag.models.state_models import SourceDocs
from fastmcp.server.middleware import Middleware, MiddlewareContext
from fastmcp import FastMCP
from typing import Any


common_router = APIRouter()
data_ingest_router = APIRouter()
data_retrieval_router = APIRouter()
dev_router = APIRouter()


class AppState:
    def __init__(self):
        self.ctx_mgr = None


app_state = AppState()


@common_router.post("/init")
async def init_context_manager(init_config: ConfigModel):
    try:
        init_ret = ""
        if init_config.config_path:
            init_ret = f"Using config path {init_config.config_path}"
            config = parse_config(init_config.config_path)
        elif init_config.context_config:
            init_ret = "Using context config"
            config = init_config.context_config
        else:
            init_ret = f"Using default config path {DEFAULT_CONFIG_PATH}"
            config = parse_config(DEFAULT_CONFIG_PATH)
        logger.info(init_ret)

        app_state.ctx_mgr = ContextManager(config=config)
        app_state.config = config

        return {
            "status": "success",
            "message": f"ContextManager initialized: {init_ret}",
        }
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


def check_context_manager():
    if app_state.ctx_mgr is None:
        raise HTTPException(
            status_code=400, detail="Context manager not initialized, please call init"
        )


@data_ingest_router.post("/add_doc")
async def add_doc(doc: AddModel):
    check_context_manager()
    try:
        await asyncio.sleep(0.001)
        app_state.ctx_mgr.add_doc(doc.document, doc.doc_index, doc.doc_metadata)
        response = f"Added document {doc.doc_index}"
        return {"status": "success", "result": response}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@data_retrieval_router.post("/call")
async def call_endpoint(call_model: CallModel):
    check_context_manager()
    try:
        result = app_state.ctx_mgr.call(call_model.state)
        if "error" in result["retriever_function"]:
            return {"status": "error", "result": result["retriever_function"]["error"]}

        return {"status": "success", "result": result["retriever_function"]["response"]}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@data_ingest_router.post("/complete_ingestion")
async def complete_ingestion():
    check_context_manager()
    try:
        app_state.ctx_mgr.call({"ingestion_function": {}})
        return {"status": "success", "result": "Ingestion complete"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@data_retrieval_router.post("/summary")
async def summary_endpoint(summary_model: SummaryModel):
    check_context_manager()
    try:
        logger.debug(f"Summary request: {summary_model.model_dump()}")
        result = app_state.ctx_mgr.call(summary_model.model_dump())
        return {"status": "success", "result": result["summarization"]["result"]}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.post("/update_config")
async def update_config(config_model: ConfigModel):
    check_context_manager()
    update_ret = ""
    try:
        if config_model.config_path:
            update_ret = f"Using config path {config_model.config_path}"
            config = parse_config(config_model.config_path)
        elif config_model.context_config:
            config = config_model.context_config
        else:
            update_ret = f"Using default config path {DEFAULT_CONFIG_PATH}"
            config = parse_config(DEFAULT_CONFIG_PATH)
        logger.info(update_ret)

        app_state.ctx_mgr.configure(config=config)
        app_state.config = config
        return {"status": "success", "message": f"Configuration updated: {update_ret}"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.post("/reset")
async def reset_context(reset_model: ResetModel):
    check_context_manager()
    try:
        result = app_state.ctx_mgr.reset(reset_model.state)
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.get("/health")
async def health_check():
    return {"status": "success", "message": "Service is healthy"}


@dev_router.get("/add_doc_from_dc")
async def add_doc_from_dc(dc_file_model: DCFileModel):
    check_context_manager()
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
            app_state.ctx_mgr.add_doc(
                vlm_chunk["vlm_response"],
                doc_i=vlm_chunk["chunkIdx"],
                doc_meta=doc_meta,
            )

        app_state.ctx_mgr.call({"ingestion_function": {"post_process": True}})
        return {"status": "success", "message": "Documents added"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


# ------------------------------------------------------------
# MCP server and tools
# ------------------------------------------------------------


mcp_server = FastMCP(name="CA-RAG retrieval MCP Server")


class CheckContextManagerMiddleware(Middleware):
    async def __call__(self, context: MiddlewareContext, call_next):
        if app_state.ctx_mgr is None:
            return {
                "status": "error",
                "message": "Context manager not initialized, please call /init",
            }

        return await call_next(context)


@mcp_server.tool()
def query(question: str) -> dict[str, Any]:
    """
    Query to ask complex questions about the videos to context manager which can internally use graphRAG or vectorRAG to answer.
    Args:
        question: The question to ask the context manager.
    Returns:
        A dictionary containing the response from the context manager with response, errors and other keys.
    """
    call_params = {"question": question}

    result = app_state.ctx_mgr.call({"retriever_function": call_params})
    response = result["retriever_function"].get("response", None)
    error = result["retriever_function"].get("error", None)
    return {"response": response, "error": error}


@mcp_server.tool()
def find_event(keywords: str) -> list[SourceDocs]:
    """
    Find events in the videos based on the keywords. Useful for filtering events later based on the returned source docs as they contain metadata.
    Args:
        keywords: The keywords to search for in the videos.
    Returns:
        A list of source docs containing the events found in the videos.
    """
    call_params = {"question": keywords}
    result = app_state.ctx_mgr.call({"retriever_function": call_params})
    return result["retriever_function"].get("source_docs", [])


@mcp_server.tool()
def find_event_formatted(keywords: str) -> list[str]:
    """
    Find events in the videos based on the keywords. Useful for getting a list of the most relevant docs as formatted strings.
    Args:
        keywords: The keywords to search for in the videos.
    Returns:
        A list of formatted docs as strings containing the events found in the videos.
    """
    call_params = {"question": keywords}
    result = app_state.ctx_mgr.call({"retriever_function": call_params})
    return result["retriever_function"].get("formatted_docs", [])


# ------------------------------------------------------------
# FastMCP + FastAPI servers config
# ------------------------------------------------------------

# mcp_server.add_middleware(CheckContextManagerMiddleware()) # TODO: Uncomment this to enable MCP only for retrieval
mcp_app = mcp_server.http_app(path="/mcp")
app = FastAPI(
    title="Context Aware RAG Service", lifespan=mcp_app.router.lifespan_context
)


app.include_router(common_router)

if str(os.environ.get("VIA_CTX_RAG_ENABLE_RET")).lower() in ["true", "1"]:
    app.include_router(data_retrieval_router)
    app.mount("/", mcp_app)
else:
    app.include_router(data_ingest_router)
    if str(os.environ.get("VIA_CTX_RAG_ENABLE_DEV")).lower() in ["true", "1"]:
        app.include_router(dev_router)
