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

"""summary_retriever.py: File contains Function class"""

from vss_ctx_rag.base.function import Function
from schema import Schema, Or, Optional
from pydantic import BaseModel, Field
from vss_ctx_rag.base.tool import logger
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.utils.ctx_rag_logger import Metrics
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from langchain_core.prompts import ChatPromptTemplate
from vss_ctx_rag.utils.globals import LLM_TOOL_NAME
from langchain.chains.combine_documents import create_stuff_documents_chain
from vss_ctx_rag.models.function_models import (
    FunctionModel,
)
from langchain_core.documents import Document


@register_function_config("summary_retriever")
class SummaryRetrieverConfig(FunctionModel):
    class SummaryRetrieverParams(BaseModel):
        summarization_prompt: str = Field(default="Summarize the following:  {context}")

    params: SummaryRetrieverParams = Field(default=SummaryRetrieverParams())


@register_function(config=SummaryRetrieverConfig)
class SummaryRetriever(Function):
    """Summary Retriever Function"""

    db: StorageTool
    batch_prompt: str
    aggregation_prompt: str
    class_schema = Schema(
        {
            Optional("start_time"): float,
            Optional("end_time"): float,
            Optional("camera_id"): str,
            Optional("uuid"): Or(str, None),
        },
        ignore_extra_keys=True,
    )

    def setup(self):
        self.db = self.get_tool("db")
        self.llm = self.get_tool(LLM_TOOL_NAME)
        self.summarization_prompt = ChatPromptTemplate.from_messages(
            [("system", self.get_param("summarization_prompt"))]
        )
        self.summarization_chain = create_stuff_documents_chain(
            self.llm, self.summarization_prompt
        )

    async def acall(self, state: dict):
        with Metrics("summary_retriever/acall", "blue"):
            try:
                logger.info(f"Summary retriever called with state: {state}")
                self.class_schema.validate(state)
                logger.debug(f"State validated: {state}")

                chunks = self.db.filter_chunks(
                    min_start_time=state.get("start_time"),
                    max_end_time=state.get("end_time"),
                    camera_id=state.get("camera_id"),
                    uuid=state.get("uuid"),
                )
                logger.info(f"Retrieved {len(chunks)} chunk data from DB: {self.db}")
                docs = [
                    Document(
                        page_content=chunk["text"],
                        metadata={**{k: v for k, v in chunk.items() if k != "text"}},
                    )
                    for chunk in chunks
                ]
                logger.debug(
                    f"Docs: { [doc.page_content[:min(len(doc.page_content), 100)] for doc in docs]}"
                )
                logger.info(f"Creating summary with {len(docs)} docs")
                summary = self.summarization_chain.invoke({"context": docs})
                logger.debug(f"Summary: {summary}")

                state["summary"] = summary

            except Exception as e:
                logger.error(f"Error in summary_retriever: {e}")
                state["error"] = str(e)
            return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, state: dict):
        pass
