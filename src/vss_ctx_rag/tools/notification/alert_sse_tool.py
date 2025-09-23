# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import aiohttp

from vss_ctx_rag.tools.notification.notification_tool import NotificationTool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.models.tool_models import register_tool_config, register_tool
from vss_ctx_rag.models.tool_models import ToolBaseModel


@register_tool_config("alert_sse_notifier")
class AlertSSEConfig(ToolBaseModel):
    endpoint: str


@register_tool(config=AlertSSEConfig)
class AlertSSETool(NotificationTool):
    """Tool for sending an alert as a post request to the endpoint.
    Implements NotificationTool class
    """

    def __init__(self, name="alert_sse_notifier", tools=None, config=None) -> None:
        super().__init__(name, config, tools)
        self.update_tool(self.config, tools)

    def update_tool(self, config, tools=None):
        self.config = config
        self.alert_endpoint = config.params.endpoint

    async def notify(self, title: str, message: str, metadata: dict):
        try:
            headers = {}
            body = {
                "title": title,
                "message": message,
                "metadata": metadata,
            }
            async with aiohttp.ClientSession() as session:
                response = await session.post(
                    self.alert_endpoint, json=body, headers=headers
                )
                response.raise_for_status()
        except Exception as ex:
            events_detected = metadata.get("events_detected", [])
            logger.error(
                "Alert callback failed for event(s) '%s' - %s",
                ", ".join(events_detected),
                str(ex),
            )
