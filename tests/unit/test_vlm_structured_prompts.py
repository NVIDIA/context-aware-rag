# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Unit tests for VLM structured summarization prompt selection.

Verifies that ``aggregation_prompt`` and ``description_merge_prompt`` use
configured overrides when set, and fall back to built-in defaults when
unset or empty. ``description_merge_prompt`` is only applied when LLM
description merging is enabled.
"""

from __future__ import annotations

from typing import Any, Optional

import pytest
from langchain_core.language_models import FakeListLLM
from langchain_core.prompts import ChatPromptTemplate

from vss_ctx_rag.functions.summarization.vlm_structured import (
    VlmStructuredSummarization,
)
from vss_ctx_rag.functions.summarization.vlm_structured_base import (
    DEFAULT_AGGREGATION_PROMPT,
    DEFAULT_DESCRIPTION_MERGE_PROMPT,
)


OVERRIDE_AGGREGATION_PROMPT = (
    "CUSTOM AGGREGATION: write a concise chronological summary of the events."
)
OVERRIDE_DESCRIPTION_MERGE_PROMPT = (
    "CUSTOM MERGE: combine the descriptions into one coherent sentence."
)


def _system_prompt(prompt: ChatPromptTemplate) -> str:
    """Return the system message template text from a chat prompt."""
    return prompt.messages[0].prompt.template


def _user_prompt(prompt: ChatPromptTemplate) -> str:
    """Return the user message template text from a chat prompt."""
    return prompt.messages[1].prompt.template


def create_vlm_structured_for_prompts(
    *,
    aggregation_prompt: Optional[str] = None,
    description_merge_prompt: Optional[str] = None,
    enable_llm_merging: bool = False,
    include_aggregation_prompt_key: bool = True,
    include_description_merge_prompt_key: bool = True,
) -> VlmStructuredSummarization:
    """Build a minimal instance and run ``_setup_aggregation_pipeline``.

    Mirrors the lightweight construction pattern used in
    ``ci-carag-oss/tests/unit/test_vlm_structured_event_merging.py``.
    """
    params: dict[str, Any] = {
        "enable_llm_merging": enable_llm_merging,
    }
    if include_aggregation_prompt_key:
        params["aggregation_prompt"] = aggregation_prompt
    if include_description_merge_prompt_key:
        params["description_merge_prompt"] = description_merge_prompt

    instance = VlmStructuredSummarization.__new__(VlmStructuredSummarization)
    instance.enable_llm_merging = enable_llm_merging
    instance.llm = FakeListLLM(responses=["ok"])
    instance.get_param = lambda key, default=None: params.get(key, default)
    instance._setup_aggregation_pipeline()
    return instance


class TestAggregationPromptSelection:
    """Tests for aggregation system-prompt override vs default."""

    @pytest.mark.parametrize(
        "aggregation_prompt,include_key",
        [
            (None, True),
            ("", True),
            ("   \n\t  ", True),
            (None, False),
        ],
        ids=["none", "empty", "whitespace", "unset"],
    )
    def test_uses_default_when_override_missing_or_blank(
        self, aggregation_prompt, include_key
    ):
        instance = create_vlm_structured_for_prompts(
            aggregation_prompt=aggregation_prompt,
            include_aggregation_prompt_key=include_key,
        )

        prompt = instance.aggregation_pipeline.first
        assert isinstance(prompt, ChatPromptTemplate)
        assert _system_prompt(prompt) == DEFAULT_AGGREGATION_PROMPT
        assert "{input}" in _user_prompt(prompt)

    def test_uses_override_when_configured(self):
        instance = create_vlm_structured_for_prompts(
            aggregation_prompt=OVERRIDE_AGGREGATION_PROMPT,
        )

        prompt = instance.aggregation_pipeline.first
        assert _system_prompt(prompt) == OVERRIDE_AGGREGATION_PROMPT
        assert _system_prompt(prompt) != DEFAULT_AGGREGATION_PROMPT
        assert "{input}" in _user_prompt(prompt)

    def test_override_is_stripped(self):
        padded = f"  {OVERRIDE_AGGREGATION_PROMPT}  \n"
        instance = create_vlm_structured_for_prompts(aggregation_prompt=padded)

        assert (
            _system_prompt(instance.aggregation_pipeline.first)
            == OVERRIDE_AGGREGATION_PROMPT
        )


class TestDescriptionMergePromptSelection:
    """Tests for description-merge system-prompt override vs default."""

    def test_pipeline_not_built_when_llm_merging_disabled(self):
        instance = create_vlm_structured_for_prompts(
            description_merge_prompt=OVERRIDE_DESCRIPTION_MERGE_PROMPT,
            enable_llm_merging=False,
        )

        assert instance.description_merge_pipeline is None

    @pytest.mark.parametrize(
        "description_merge_prompt,include_key",
        [
            (None, True),
            ("", True),
            ("   \n\t  ", True),
            (None, False),
        ],
        ids=["none", "empty", "whitespace", "unset"],
    )
    def test_uses_default_when_override_missing_or_blank(
        self, description_merge_prompt, include_key
    ):
        instance = create_vlm_structured_for_prompts(
            description_merge_prompt=description_merge_prompt,
            enable_llm_merging=True,
            include_description_merge_prompt_key=include_key,
        )

        assert instance.description_merge_pipeline is not None
        prompt = instance.description_merge_pipeline.first
        assert isinstance(prompt, ChatPromptTemplate)
        assert _system_prompt(prompt) == DEFAULT_DESCRIPTION_MERGE_PROMPT
        user = _user_prompt(prompt)
        assert "{event_type}" in user
        assert "{descriptions}" in user

    def test_uses_override_when_configured(self):
        instance = create_vlm_structured_for_prompts(
            description_merge_prompt=OVERRIDE_DESCRIPTION_MERGE_PROMPT,
            enable_llm_merging=True,
        )

        prompt = instance.description_merge_pipeline.first
        assert _system_prompt(prompt) == OVERRIDE_DESCRIPTION_MERGE_PROMPT
        assert _system_prompt(prompt) != DEFAULT_DESCRIPTION_MERGE_PROMPT
        user = _user_prompt(prompt)
        assert "{event_type}" in user
        assert "{descriptions}" in user

    def test_override_is_stripped(self):
        padded = f"\n  {OVERRIDE_DESCRIPTION_MERGE_PROMPT}  "
        instance = create_vlm_structured_for_prompts(
            description_merge_prompt=padded,
            enable_llm_merging=True,
        )

        assert (
            _system_prompt(instance.description_merge_pipeline.first)
            == OVERRIDE_DESCRIPTION_MERGE_PROMPT
        )


class TestBothPromptOverridesTogether:
    """Ensure aggregation and description-merge overrides can be set independently."""

    def test_both_overrides_applied_when_llm_merging_enabled(self):
        instance = create_vlm_structured_for_prompts(
            aggregation_prompt=OVERRIDE_AGGREGATION_PROMPT,
            description_merge_prompt=OVERRIDE_DESCRIPTION_MERGE_PROMPT,
            enable_llm_merging=True,
        )

        assert (
            _system_prompt(instance.aggregation_pipeline.first)
            == OVERRIDE_AGGREGATION_PROMPT
        )
        assert (
            _system_prompt(instance.description_merge_pipeline.first)
            == OVERRIDE_DESCRIPTION_MERGE_PROMPT
        )

    def test_aggregation_override_with_default_description_merge(self):
        instance = create_vlm_structured_for_prompts(
            aggregation_prompt=OVERRIDE_AGGREGATION_PROMPT,
            description_merge_prompt=None,
            enable_llm_merging=True,
        )

        assert (
            _system_prompt(instance.aggregation_pipeline.first)
            == OVERRIDE_AGGREGATION_PROMPT
        )
        assert (
            _system_prompt(instance.description_merge_pipeline.first)
            == DEFAULT_DESCRIPTION_MERGE_PROMPT
        )
