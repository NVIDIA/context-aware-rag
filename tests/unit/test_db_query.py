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

"""Unit tests for the db_query function."""

import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from vss_ctx_rag.functions.storage.db_query import (
    DbQueryConfig,
    DbQueryFunc,
    _make_serializable,
)
from vss_ctx_rag.models.function_models import _FUNCTION_IMPLEMENTATION_REGISTRY
from vss_ctx_rag.tools.storage.storage_tool import StorageTool


class _FakeDB(StorageTool):
    """Minimal StorageTool stub that records query calls."""

    def __init__(self, name="fake_db", result=None, raise_error: Exception = None):
        # Bypass Tool.__init__ wiring; only need query + get_tool contract pieces.
        self.name = name
        self._result = result if result is not None else [{"id": 1}]
        self._raise = raise_error
        self.calls: List[tuple] = []

    def add_summary(self, summary, metadata):
        pass

    def reset(self, state: dict = {}):
        pass

    async def aget_text_data(self, start_batch_index, end_batch_index, uuid):
        return []

    def filter_chunks(
        self,
        min_start_time: Optional[float] = None,
        max_start_time: Optional[float] = None,
        min_end_time: Optional[float] = None,
        max_end_time: Optional[float] = None,
        camera_id: Optional[str] = None,
        chunk_id: Optional[int] = None,
        uuid: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        return []

    def query(self, query, params: dict = {}):
        self.calls.append((query, params))
        if self._raise:
            raise self._raise
        return self._result

    async def aget_max_batch_index(self, uuid: str) -> int:
        return 0

    def retrieve_docs(self, uuid: str, doc_type: str = "raw_events"):
        return []

    def as_retriever(self, search_kwargs: dict = None):
        return MagicMock()

    def update_tool(self, config, tools=None):
        return self


class _AsyncFakeDB(_FakeDB):
    async def query(self, query, params: dict = {}):
        self.calls.append((query, params))
        if self._raise:
            raise self._raise
        return self._result


def _build_func(db: StorageTool) -> DbQueryFunc:
    fn = DbQueryFunc("db_query")
    fn.add_tool("db", db)
    fn.config(DbQueryConfig())
    fn.done()
    return fn


def test_db_query_is_registered():
    assert "db_query" in _FUNCTION_IMPLEMENTATION_REGISTRY
    info = _FUNCTION_IMPLEMENTATION_REGISTRY["db_query"]
    assert info["class"] == "DbQueryFunc"


def test_db_query_executes_sync_backend():
    db = _FakeDB(result=[{"n": 42}])
    fn = _build_func(db)

    out = asyncio.run(fn({"query": "MATCH (n) RETURN n", "params": {"limit": 1}}))

    assert out == {"result": [{"n": 42}]}
    assert db.calls == [("MATCH (n) RETURN n", {"limit": 1})]


def test_db_query_defaults_empty_params():
    db = _FakeDB()
    fn = _build_func(db)

    out = asyncio.run(fn({"query": "pk > 0"}))

    assert "result" in out
    assert db.calls == [("pk > 0", {})]


def test_db_query_supports_async_backend():
    db = _AsyncFakeDB(result=[{"hit": True}])
    fn = _build_func(db)

    out = asyncio.run(fn({"query": {"query": {"match_all": {}}}, "params": {}}))

    assert out == {"result": [{"hit": True}]}
    assert len(db.calls) == 1


def test_db_query_missing_query_returns_error():
    fn = _build_func(_FakeDB())

    out = asyncio.run(fn({}))

    assert "error" in out
    assert "query" in out["error"]


def test_db_query_backend_exception_returns_error():
    fn = _build_func(_FakeDB(raise_error=RuntimeError("boom")))

    out = asyncio.run(fn({"query": "bad"}))

    assert out == {"error": "boom"}


def test_db_query_setup_requires_db_tool():
    fn = DbQueryFunc("db_query")
    fn.config(DbQueryConfig())
    with pytest.raises(RuntimeError, match="requires a 'db' tool"):
        fn.done()


def test_make_serializable_handles_nested_and_temporal_like():
    class _Temporal:
        def iso_format(self):
            return "2024-01-01T00:00:00"

    assert _make_serializable({"a": [1, _Temporal()]}) == {
        "a": [1, "2024-01-01T00:00:00"]
    }


def test_allowed_tool_types_cover_all_storage_backends():
    assert set(DbQueryConfig.ALLOWED_TOOL_TYPES.keys()) == {
        "neo4j",
        "arango",
        "milvus",
        "elasticsearch",
    }
    for keywords in DbQueryConfig.ALLOWED_TOOL_TYPES.values():
        assert keywords == ["db"]
