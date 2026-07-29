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

import pytest
from vss_ctx_rag.functions.rag.graph_rag.prompt import (
    _get_critical_assumptions,
    _get_chunk_reader_search_logic,
)


# Create mock tool objects with a 'name' attribute to simulate real tools
class MockTool:
    def __init__(self, name):
        self.name = name


chunk_reader_tool = MockTool(name="ChunkReader")
other_tool = MockTool(name="OtherTool")
chunk_search_tool = MockTool(name="ChunkSearch")


@pytest.mark.parametrize(
    "tools, expected_keywords, unexpected_keywords",
    [
        (
            [chunk_reader_tool],
            ["VISUAL ANALYSIS MANDATORY"],
            ["SUBTITLE + VISUAL VERIFICATION WORKFLOW"],
        ),
        (
            [other_tool],
            ["NEVER treat queries as logic puzzles"],
            ["VISUAL ANALYSIS MANDATORY"],
        ),
        ([], ["NEVER treat queries as logic puzzles"], ["VISUAL ANALYSIS MANDATORY"]),
    ],
)
def test_get_critical_assumptions(tools, expected_keywords, unexpected_keywords):
    """
    Test the different logic paths for returning critical assumptions based on available tools.
    """
    result = _get_critical_assumptions(tools)
    for keyword in expected_keywords:
        assert keyword in result
    for keyword in unexpected_keywords:
        assert keyword not in result


@pytest.mark.parametrize(
    "tools, should_be_present",
    [
        ([chunk_reader_tool, chunk_search_tool], True),
        ([other_tool], False),
    ],
)
def test_get_chunk_reader_search_logic(tools, should_be_present):
    """
    Test that the chunk reader search logic is added only for the specific case of
    a chunk reader tool.
    """
    result = _get_chunk_reader_search_logic(tools=tools)
    keyword = "To save the calling budget"
    if should_be_present:
        assert keyword in result
    else:
        assert keyword not in result
