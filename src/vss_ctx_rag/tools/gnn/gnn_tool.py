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

import requests
import json
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.tool_models import (
    register_tool_config,
    register_tool,
    ToolBaseModel,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger


@register_tool_config("gnn")
class GnnConfig(ToolBaseModel):
    base_url: str = "http://localhost:8000"
    timeout: int = 30


@register_tool(config=GnnConfig)
class GnnTool(Tool):
    """GNN Tool that wraps GNN for use as a proper Tool."""

    def __init__(
        self,
        name: str,
        tools=None,
        config=None,
    ):
        super().__init__(name, config, tools)
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        # Tools are instantiated with a ToolsConfig wrapper; actual fields live under .params
        self.base_url = self.config.params.base_url
        self.timeout = self.config.params.timeout

        logger.info(f"Initialized GnnTool with base_url: {self.base_url}")

    async def acall_inference(self, payload_data: dict, headers: dict) -> dict:
        def _do_request():
            logger.debug(f"GNN NIM Base URL: {self.base_url}")
            logger.debug(f"GNN NIM Request: {payload_data}")
            logger.debug(f"GNN NIM Headers: {headers}")
            response = requests.post(
                f"{self.base_url}/v1/inference",
                headers=headers,
                data=json.dumps(payload_data),
                timeout=self.timeout,
            )
            response.raise_for_status()  # Raise exception for HTTP errors
            json_response = response.json()
            logger.debug(f"GNN NIM Response: {json_response}")
            if (
                "result" not in json_response
                or not json_response["result"]
                or len(json_response["result"]) == 0
            ):
                raise ValueError("Invalid GNN response: missing or empty result")
            answer = json_response["result"][0]
            return answer

        try:
            answer = await asyncio.to_thread(_do_request)
            return answer
        except requests.exceptions.Timeout:
            logger.error(f"GNN request timed out after {self.timeout} seconds")
            raise
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Failed to connect to GNN service at {self.base_url}: {e}")
            raise
        except requests.exceptions.HTTPError as e:
            logger.error(f"GNN service returned HTTP error: {e}")
            raise
        except requests.exceptions.RequestException as e:
            logger.error(f"GNN request failed: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse GNN response as JSON: {e}")
            raise
        except ValueError as e:
            logger.error(f"Invalid GNN response format: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during GNN inference: {e}")
            raise
