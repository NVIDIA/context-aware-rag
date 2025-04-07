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

"""utils.py: File contains utility functions"""

import jsonschema
import json
import re
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.context_manager.context_manager_models import (
    ContextManagerConfig,
    AlertConfig,
)
from typing import Dict, Any, Optional


def validate_config_json(parsed_yaml, schema_json_filepath):
    try:
        with open(schema_json_filepath) as f:
            spec_schema = json.load(f)
        jsonschema.validate(parsed_yaml, spec_schema)
    except jsonschema.ValidationError as e:
        raise ValueError(
            f"Invalid config file: {'.'.join([str(p) for p in e.absolute_path])}: {e.message}"
        )


def remove_think_tags(text_in):
    text_out = re.sub(r"<think>.*?</think>", "", text_in, flags=re.DOTALL)
    return text_out


def remove_lucene_chars(text: str) -> str:
    """
    Remove Lucene special characters from the given text.

    This function takes a string as input and removes any special characters
    that are used in Lucene query syntax. The characters removed are:
    +, -, &, |, !, (, ), {, }, [, ], ^, ", ~, *, ?, :, \ and /.

    Args:
        text (str): The input string from which to remove Lucene special characters.

    Returns:
        str: The cleaned string with Lucene special characters replaced by spaces.
    """
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


class RequestInfo:
    def __init__(
        self,
        summarize: bool = True,
        enable_chat: bool = True,
        is_live: bool = False,
        uuid: str = "0",
        caption_summarization_prompt: str = "Return input as is",
        summary_aggregation_prompt: str = "Return input as is",
        chunk_size: int = 0,
        summary_duration: Optional[int] = None,
        summarize_top_p: Optional[float] = None,
        summarize_temperature: Optional[float] = None,
        summarize_max_tokens: Optional[int] = None,
        chat_top_p: Optional[float] = None,
        chat_temperature: Optional[float] = None,
        chat_max_tokens: Optional[int] = None,
        notification_top_p: Optional[float] = None,
        notification_temperature: Optional[float] = None,
        notification_max_tokens: Optional[int] = None,
        rag_type: Optional[str] = None,
    ):
        self.summarize = summarize
        self.enable_chat = enable_chat
        self.is_live = is_live
        self.uuid = uuid
        self.caption_summarization_prompt = caption_summarization_prompt
        self.summary_aggregation_prompt = summary_aggregation_prompt
        self.chunk_size = chunk_size
        self.summary_duration = summary_duration
        self.summarize_top_p = summarize_top_p
        self.summarize_temperature = summarize_temperature
        self.summarize_max_tokens = summarize_max_tokens
        self.chat_top_p = chat_top_p
        self.chat_temperature = chat_temperature
        self.chat_max_tokens = chat_max_tokens
        self.notification_top_p = notification_top_p
        self.notification_temperature = notification_temperature
        self.notification_max_tokens = notification_max_tokens
        self.rag_type = rag_type


def validate_config(config: Dict[str, Any]) -> None:
    if config.get("summarization", {}).get("enable") is False:
        logger.warning(
            "Summarization disabling not supported, setting summarization enable to True"
        )
        config["summarization"]["enable"] = True

    try:
        _ = ContextManagerConfig(**config)
        return
    except Exception as e_context:
        context_error = e_context

    try:
        _ = AlertConfig(**config)
        return
    except Exception as e_alert:
        alert_error = e_alert

    error_msg = (
        f"Invalid Config: failed as ContextManagerConfig ({context_error}) "
        f"and AlertConfig ({alert_error}). "
        "Config must be either ContextManagerConfig or AlertConfig."
    )
    logger.warn(error_msg)
    raise ValueError(error_msg)
