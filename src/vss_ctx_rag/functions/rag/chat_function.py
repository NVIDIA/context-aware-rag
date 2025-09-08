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

"""function.py: File contains Function class"""

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.utils.ctx_rag_logger import logger


class ChatFunction(Function):
    def setup(self) -> dict:
        logger.info(f"=== DEBUG: ChatFunction.setup() ===")
        
        self.rag = self.get_param("rag")
        self.chat_config = self._params
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool("llm")
        self.vector_db = self.get_tool("vector_db")
        self.extraction_function = self.get_function("extraction_function")
        self.retrieval_function = self.get_function("retrieval_function")
        
        logger.info(f"RAG type: {self.rag}")
        logger.info(f"Extraction function: {self.extraction_function}")
        logger.info(f"Retrieval function: {self.retrieval_function}")
        logger.info(f"Retrieval function type: {type(self.retrieval_function)}")
        logger.info("=== END DEBUG: ChatFunction.setup() ===")

    async def acall(self, state: dict) -> dict:
        logger.info(f"=== DEBUG: ChatFunction.acall() ===")
        logger.info(f"State: {state}")
        logger.info(f"Has extraction_function: {self.extraction_function is not None}")
        logger.info(f"Has retrieval_function: {self.retrieval_function is not None}")
        
        if (
            self.extraction_function
            and "post_process" in state
            and state["post_process"]
        ):
            logger.info("Calling extraction function for post_process")
            state = await self.extraction_function.acall(state)
        if "question" in state:
            logger.info(f"Calling retrieval function with question: {state.get('question', 'NO_QUESTION')}")
            state = await self.retrieval_function.acall(state)
            logger.info(f"Retrieval function returned: {state.get('response', 'NO_RESPONSE')[:100]}...")
        
        logger.info("=== END DEBUG: ChatFunction.acall() ===")
        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        if self.extraction_function:
            await self.extraction_function.aprocess_doc(doc, doc_i, doc_meta)

    async def areset(self, state: dict):
        if self.extraction_function:
            await self.extraction_function.areset(state)
        await self.retrieval_function.areset(state)
