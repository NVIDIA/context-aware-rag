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

from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import logger
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_community.graphs import Neo4jGraph
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from neo4j.time import DateTime
from typing import List, Dict
import asyncio


class Neo4jGraphDB(StorageTool):
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        name="neo4j_db",
        embedding_model_name="nvidia/nv-embedqa-e5-v5",
        embedding_base_url="https://integrate.api.nvidia.com/v1",
    ) -> None:
        super().__init__(name)

        if bool(os.getenv("NVIDIA_API_KEY")) is True:
            api_key = os.getenv("NVIDIA_API_KEY")
        else:
            api_key = "NOAPIKEYSET"

        self.graph_db = Neo4jGraph(
            url=url,
            username=username,
            password=password,
            sanitize=True,
            refresh_schema=False,
        )
        self.embeddings = NVIDIAEmbeddings(
            model=embedding_model_name,
            truncate="NONE",
            api_key=api_key,
            base_url=embedding_base_url,
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "\n-", ".", ";", ",", " ", ""],
        )

    def extract_cypher(self, text: str) -> str:
        """Extract Cypher code from a text.

        Args:
            text: Text to extract Cypher code from.

        Returns:
            Cypher code extracted from the text.
        """
        # The pattern to find Cypher code enclosed in triple backticks
        # pattern = r"```cypher(.*?)```"

        # # Find all matches in the input text
        # matches = re.findall(pattern, text, re.DOTALL)
        def find_between(s, first, last):
            try:
                start = s.index(first) + len(first)
                end = s.index(last, start)
                return s[start:end]
            except ValueError:
                return ""

        logger.debug("Generated Query: %s", text)
        start = "CYPHER_START"
        end = "CYPHER_END"
        result = find_between(text, start, end)
        logger.debug("Extracted Query: %s", result)

        return result if result else text

    def run_cypher_query(self, query, params: dict = {}):
        # query = correct_query(graph.get_structured_schema, query)
        # query = self.extract_cypher(query)
        logger.debug(f"Query: {query}")
        try:
            result = self.graph_db.query(query, params)
            logger.debug(f"Query exec result: {result}")
            return result
        except Exception as e:
            logger.error("Neo4j Query failed %s", str(e))
            return None

    # async def aadd_texts(self, texts):
    #     logger.debug("Adding text to vector db %s", texts)
    #     texts_chunks=[]
    #     for text in texts:
    #         text_chunks = self.text_splitter.split_text(text)
    #         texts_chunks.append(text_chunks)
    #     return await self.vector_db.aadd_texts(texts_chunks)

    # def get_fresh_schema(self, _):
    #     self.graph_db.refresh_schema()
    #     return self.graph_db.schema

    def datetime_encoder(self, obj):
        if isinstance(obj, DateTime):
            return obj.to_native().isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def arun_cypher_query(self, query: str) -> List[Dict]:
        """Async wrapper around run_cypher_query"""
        return await asyncio.to_thread(self.run_cypher_query, query)
