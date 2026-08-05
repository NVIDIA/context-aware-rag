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

"""db_query.py: Function that executes a query against a configured storage DB tool."""

import inspect
from typing import Any, ClassVar, Dict, List, Optional

from pydantic import BaseModel, Field

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.function_models import (
    FunctionModel,
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger


@register_function_config("db_query")
class DbQueryConfig(FunctionModel):
    """Config for the db_query function.

    Accepts any storage backend that implements ``StorageTool.query``:
    neo4j, arango, milvus, and elasticsearch.
    """

    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "neo4j": ["db"],
        "arango": ["db"],
        "milvus": ["db"],
        "elasticsearch": ["db"],
    }

    class DbQueryParams(BaseModel):
        pass

    params: DbQueryParams = Field(default_factory=DbQueryParams)


@register_function(config=DbQueryConfig)
class DbQueryFunc(Function):
    """Execute a backend-native query via the attached storage tool.

    Call payload (passed as ``call_params`` for this function)::

        {
            "query": <str | dict>,   # required — Cypher / AQL / Milvus expr / ES body
            "params": { ... },       # optional bind / kwargs for the backend
        }

    Returns::

        {"result": <backend results>}   # on success
        {"error": "<message>"}          # on failure
    """

    db: StorageTool

    def setup(self) -> None:
        self.db = self.get_tool("db")
        if self.db is None:
            raise RuntimeError(
                f"Function '{self.name}' requires a 'db' tool "
                "(neo4j, arango, milvus, or elasticsearch)"
            )

    async def acall(self, state: dict) -> dict:
        with Metrics("db_query/acall", "blue"):
            try:
                if not isinstance(state, dict):
                    raise ValueError(
                        "db_query state must be a dict with 'query' "
                        "(and optional 'params')"
                    )

                query = state.get("query")
                if query is None:
                    raise ValueError("db_query requires a 'query' field in the call state")

                params: Optional[dict] = state.get("params")
                if params is None:
                    params = {}
                if not isinstance(params, dict):
                    raise ValueError("db_query 'params' must be a dict when provided")

                logger.info(
                    "db_query calling %s.query (params keys=%s)",
                    type(self.db).__name__,
                    list(params.keys()),
                )
                result = self.db.query(query, params)
                if inspect.isawaitable(result):
                    result = await result

                return {"result": _make_serializable(result)}
            except Exception as e:
                logger.error(f"Error in db_query: {e}")
                return {"error": str(e)}

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, state: dict):
        pass


def _make_serializable(value: Any) -> Any:
    """Best-effort conversion so results survive multiprocess queues / JSON."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _make_serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_serializable(v) for v in value]
    if hasattr(value, "iso_format"):
        # neo4j temporal types
        try:
            return value.iso_format()
        except Exception:
            pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    try:
        return str(value)
    except Exception:
        return repr(value)
