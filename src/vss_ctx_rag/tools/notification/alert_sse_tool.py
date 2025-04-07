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
from vss_ctx_rag.tools.notification import NotificationTool
from vss_ctx_rag.utils.ctx_rag_logger import logger


class AlertSSETool(NotificationTool):
    """Tool for sending an alert as a post request to the endpoint.
    Implements NotificationTool class
    """

    def __init__(self, endpoint: str, name="alert_sse_notifier") -> None:
        super().__init__(name)
        self.alert_endpoint = endpoint

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
