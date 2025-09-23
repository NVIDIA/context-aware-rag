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

"""summarization.py: File contains Function class"""

import asyncio
import json

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.tools.notification.notification_tool import NotificationTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import LLM_TOOL_NAME
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from pydantic import BaseModel, Field
from typing import Optional, Union, List
from vss_ctx_rag.tools.llm.llm_handler import LLMConfig


class Event(BaseModel):
    event_id: str
    event_list: List[str]


class NotificationParams(BaseModel):
    events: Optional[List[Event]] = Field(default_factory=list)


@register_function_config("notification")
class NotificationConfig(BaseModel):
    params: NotificationParams
    llm: Optional[Union[str, LLMConfig]] = None


@register_function(config=NotificationConfig)
class Notifier(Function):
    """
    Notifier Function
    """

    prompt_str: str = 'You are a Event Detection System. From the following\
                        text, detect if any of the following events are present {events}. \
                        The output should be a json in the format:\
                        {{\
                            "result" : [\
                                {{\
                                    "event": "$event_name_1",\
                                    "is_detected": $is_detected\
                                }},\
                                {{\
                                    "event": "$event_name_2",\
                                    "is_detected": $is_detected\
                                }}\
                                {{\
                                    "event": "$event_name_n",\
                                    "is_detected": $is_detected\
                                }}\
                            ]\
                        }}\
                        where "$event_name_i" is the ith event to be detected and\
                        "$is_detected" is either true or false.\
                        Output should only be the said json; nothing else!'
    output_parser = StrOutputParser()
    pipeline: RunnableSequence
    notification_tool: NotificationTool
    events: list[dict]

    def setup(self):
        self.notification_tool = self.get_tool("notification_tool")
        self.events = self.get_param("events", default=[])
        self.prompt = ChatPromptTemplate.from_messages(
            [("system", self.prompt_str), ("user", "{input}")]
        )
        self.output_parser = StrOutputParser()
        self.pipeline = self.prompt | self.get_tool(LLM_TOOL_NAME) | self.output_parser

    async def acall(self, state: dict):
        return await asyncio.sleep(0.001)

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        # TODO: Ensure the output is proper. Ensure json load works.
        notifications = []
        tasks = []

        with Metrics("notifier/llm_call", span_kind=Metrics.SPAN_KIND["LLM"]) as tm:
            tm.input({"events": self.events})

            for event_item in self.events:
                event = event_item["event_list"]
                event_id = event_item["event_id"]

                tasks.append(
                    self.pipeline.ainvoke(
                        {
                            "events": str(event),
                            "input": doc,
                        }
                    )
                )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            tm.output(results)

        with Metrics("notifier/notify_call", span_kind=Metrics.SPAN_KIND["TOOL"]):
            for result, event_item in zip(results, self.events):
                event_id = event_item["event_id"]

                if isinstance(result, Exception):
                    logger.warning(
                        f"Pipeline invocation failed for event_id {event_id}: {result}"
                    )
                    continue

                result = result.strip("`json")
                try:
                    result = json.loads(result)
                except Exception:
                    logger.warning(
                        f"Notification failed due to incorrect json generation:\n{result}"
                    )
                    continue

                events_detected = [
                    item["event"] for item in result["result"] if item["is_detected"]
                ]

                if events_detected:
                    events_detected_str = " ".join(events_detected)
                    notifications.append(
                        self.notification_tool.notify(
                            title=f"{events_detected_str} detected!",
                            message=f"{events_detected_str} detected in '{doc}'",
                            metadata=doc_meta
                            | {
                                "doc": doc,
                                "events_detected": events_detected,
                                "event_id": event_id,
                            },
                        )
                    )
            if notifications:
                await asyncio.gather(*notifications)
