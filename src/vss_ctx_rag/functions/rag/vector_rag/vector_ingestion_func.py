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

"""vector_ingestion_func.py: File contains Function class"""

import asyncio
from typing import Optional, ClassVar, Dict, List
from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.functions.rag.config import IngestionConfig


@register_function_config("vector_ingestion")
class VectorIngestionConfig(IngestionConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "milvus": ["db"],
        "elasticsearch": ["db"],
    }

    class VectorIngestionParams(IngestionConfig.IngestionParams):
        custom_metadata: Optional[dict] = None
        is_user_specified_collection: Optional[bool] = False
        uuid: Optional[str] = "default"

    params: VectorIngestionParams


@register_function(config=VectorIngestionConfig)
class VectorIngestionFunc(Function):
    """Vector Ingestion Function"""

    db: StorageTool

    def setup(self):
        self.db = self.get_tool("db")
        self.uuid = self.get_param("uuid", default="default")

    async def acall(self, state: dict):
        """batch summarization function call"""
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        try:
            logger.info("Adding doc %d", doc_i)
            doc_meta.setdefault("is_first", False)
            doc_meta.setdefault("is_last", False)

            self.db.add_summary(
                summary=doc,
                metadata={
                    **doc_meta,
                    "doc_type": "caption",
                    "batch_i": -1,
                    "uuid": self.uuid,
                    "linked_summary_chunks": [],
                },
            )
        except Exception as e:
            logger.error(e)

    async def areset(self, state: dict):
        self.db.reset(state)
        await asyncio.sleep(0.001)
