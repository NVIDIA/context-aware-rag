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


from typing import Any, TypedDict


class Metadata(TypedDict, total=False):
    """
    Metadata for documents and chunks in the RAG system.

    .. code-block:: python

        metadata = Metadata(
            asset_dirs=["path/to/assets"],
            length=1024,
            streamId="stream123",
            file="document.txt",
            start_pts=100,
            end_pts=200,
            is_first=True,
            is_last=False,
            uuid="unique-id-123",
            linked_summary_chunks="chunk1,chunk2"
        )
    """

    asset_dirs: list[str]
    length: int
    streamId: str
    file: str
    start_pts: int
    end_pts: int
    is_first: bool
    is_last: bool
    uuid: str
    linked_summary_chunks: str


class SourceDocs(TypedDict, total=False):
    """
    Source docs for the retriever function.

    .. code-block:: python

        doc = SourceDocs(
            page_content="Hello world",
            metadata={"source": "file.txt"}
        )
    """

    page_content: str
    metadata: dict[str, Any]


class RetrieverFunctionState(TypedDict, total=False):
    """
    State of the retriever function used for both input and output for RAG types: vector-rag, graph-rag, foundation-rag

    .. code-block:: python

        state = {
            "question": "What is the capital of France?",
            "response_method": "text",
            "response_schema": None,
            "response": "Paris",
            "error": None,
            "source_docs": [
                {
                    "page_content": "Paris is the capital and most populous city of France.",
                    "metadata": {"source": "geography.txt", "page": 1}
                }
            ]
        }
    """

    question: str
    response_method: str | None
    response_schema: dict[str, Any] | None
    response: str | dict | None
    error: str | None
    source_docs: list[SourceDocs] | None
    formatted_docs: list[str] | None
