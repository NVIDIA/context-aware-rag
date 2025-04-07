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
import yaml
import json
from vss_ctx_rag.context_manager import ContextManager
from vss_ctx_rag.utils.ctx_rag_logger import logger
from .models import (
    AddRequest,
    RequestInfo,
    CallRequest,
    InitRequest,
    UpdateConfigRequest,
    ResetRequest,
    DCFileRequest,
)
from fastapi import APIRouter
import traceback
import random
from .globals import DEFAULT_CONFIG_PATH

app = FastAPI()

common_router = APIRouter()
data_ingest_router = APIRouter()
data_retrieval_router = APIRouter()
dev_router = APIRouter()


class AppState:
    def __init__(self):
        self.ctx_mgr = None
        self.req_info = RequestInfo()


app_state = AppState()


@common_router.post("/init")
async def init_context_manager(init_request: InitRequest):
    try:
        init_ret = ""
        if init_request.config_path:
            init_ret = f"Using config path {init_request.config_path}"
            with open(init_request.config_path, "r") as file:
                config = yaml.safe_load(file)
        elif init_request.context_config:
            init_ret = "Using context config"
            config = init_request.context_config
        else:
            init_ret = f"Using default config path {DEFAULT_CONFIG_PATH}"
            config = yaml.safe_load(open(DEFAULT_CONFIG_PATH))
        logger.info(init_ret)
        config["milvus_db_host"] = os.environ["MILVUS_HOST"]
        config["milvus_db_port"] = os.environ["MILVUS_PORT"]
        config["api_key"] = os.environ["NVIDIA_API_KEY"]
        app_state.req_info.summarize_max_tokens = config["summarization"]["llm"].get(
            "max_tokens"
        )
        app_state.req_info.summarize_temperature = config["summarization"]["llm"].get(
            "temperature"
        )
        app_state.req_info.summarize_top_p = config["summarization"]["llm"].get("top_p")

        app_state.req_info.chat_max_tokens = config["chat"]["llm"].get("max_tokens")
        app_state.req_info.chat_temperature = config["chat"]["llm"].get("temperature")
        app_state.req_info.chat_top_p = config["chat"]["llm"].get("top_p")

        app_state.req_info.rag_type = config["chat"].get("rag", "vector-rag")

        app_state.req_info.uuid = (
            init_request.uuid if init_request.uuid else str(random.randint(0, 1000000))
        )

        app_state.ctx_mgr = ContextManager(config=config, req_info=app_state.req_info)

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
async def add_doc(doc: AddRequest):
    check_context_manager()
    try:
        app_state.ctx_mgr.add_doc(doc.document, doc.doc_index, doc.doc_metadata)
        response = f"Added document {doc.doc_index}"
        if "is_last" in doc.doc_metadata:
            response += " and calling post process"
            app_state.ctx_mgr.call({"chat": {"post_process": True}})
        return {"status": "success", "result": response}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@data_retrieval_router.post("/call")
async def call_endpoint(call_request: CallRequest):
    check_context_manager()
    try:
        result = app_state.ctx_mgr.call(call_request.state)
        return {"status": "success", "result": result["chat"]["response"]}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.post("/update_config")
async def update_config(config_request: UpdateConfigRequest):
    check_context_manager()
    update_ret = ""
    try:
        if config_request.config_path:
            update_ret = f"Using config path {config_request.config_path}"
            with open(config_request.config_path, "r") as file:
                config = yaml.safe_load(file)
        elif config_request.context_config:
            config = config_request.context_config
        else:
            update_ret = f"Using default config path {DEFAULT_CONFIG_PATH}"
            config = yaml.safe_load(open(DEFAULT_CONFIG_PATH))
        logger.info(update_ret)

        if config_request.request_info:
            app_state.req_info = config_request.request_info

        app_state.ctx_mgr.configure_update(config=config, req_info=app_state.req_info)
        return {"status": "success", "message": f"Configuration updated: {update_ret}"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.post("/reset")
async def reset_context(reset_request: ResetRequest):
    check_context_manager()
    try:
        result = app_state.ctx_mgr.reset(reset_request.state)
        return {"status": "success", "result": result}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@common_router.get("/health")
async def health_check():
    return {"status": "success", "message": "Service is healthy"}


@dev_router.get("/add_doc_from_dc")
async def add_doc_from_dc(dc_file_path: DCFileRequest):
    check_context_manager()
    try:
        data_list = []
        file_path = dc_file_path.dc_file_path
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

        app_state.ctx_mgr.call({"chat": {"post_process": True}})
        return {"status": "success", "message": "Documents added"}
    except Exception as e:
        traceback.print_exc()
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


app.include_router(common_router)

if str(os.environ.get("VIA_CTX_RAG_ENABLE_RET")).lower() in ["true", "1"]:
    app.include_router(data_retrieval_router)
else:
    app.include_router(data_ingest_router)
    if str(os.environ.get("VIA_CTX_RAG_ENABLE_DEV")).lower() in ["true", "1"]:
        app.include_router(dev_router)
