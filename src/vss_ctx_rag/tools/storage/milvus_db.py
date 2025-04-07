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

import asyncio
import os
from langchain_milvus import Milvus
from langchain.docstore.document import Document
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, NVIDIARerank
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vss_ctx_rag.tools.storage import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from pymilvus import MilvusException


class MilvusDBTool(StorageTool):
    """Handler for Milvus DB which stores the video embeddings mapped using
    the summary text embeddings which can be used for retrieval.

    Implements StorageHandler class
    """

    def __init__(
        self,
        collection_name,
        host="127.0.0.1",
        port="19530",
        embedding_model_name="nvidia/llama-3.2-nv-embedqa-1b-v2",
        embedding_base_url="https://integrate.api.nvidia.com/v1",
        reranker_model_name="nvidia/llama-3.2-nv-rerankqa-1b-v2",
        reranker_base_url="https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-3_2-nv-rerankqa-1b-v2/reranking",
        name="milvus_db",
    ) -> None:
        super().__init__(name)

        if bool(os.getenv("NVIDIA_API_KEY")) is True:
            api_key = os.getenv("NVIDIA_API_KEY")
        else:
            api_key = "NOAPIKEYSET"
        self.connection = {"host": host, "port": port}
        self.collection_name = collection_name

        self.embedding = NVIDIAEmbeddings(
            model=embedding_model_name,
            truncate="END",
            api_key=api_key,
            base_url=embedding_base_url,
        )
        self.vector_db = Milvus(
            embedding_function=self.embedding,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=os.getenv("VIA_CTX_RAG_ENABLE_RET", "True").lower()
            not in ["true", "1"],
        )
        self.reranker = NVIDIARerank(
            model=reranker_model_name, api_key=api_key, base_url=reranker_base_url
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", ";", ",", " ", ""],
        )

    def add_summary(self, summary: str, metadata: dict):
        with TimeMeasure("milvusdb/add caption", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
        try:
            return self.vector_db.add_documents([doc])
        except MilvusException as e:
            logger.error(
                f"Invalid metadata while adding documents to Milvus: {metadata}"
            )
            raise e

    async def aadd_summary(self, summary: str, metadata: dict):
        with TimeMeasure("milvusdb/add caption", "blue"):
            doc = Document(page_content=summary, metadata=metadata)
            return await self.vector_db.aadd_documents([doc])

    def add_summaries(self, batch_summary: list[str], batch_metadata: list[dict]):
        with TimeMeasure("Milvus/AddSummries", "yellow"):
            if len(batch_summary) != len(batch_metadata):
                raise ValueError(
                    "Incorrect param. The length of batch_summary batch and\
                    metadata batch should match."
                )
            docs = []
            for i in range(len(batch_summary)):
                docs.append(
                    Document(page_content=batch_summary[i], metadata=batch_metadata[i])
                )
            document_chunks = self.text_splitter.split_documents(docs)
            self.vector_db.add_documents(document_chunks)

    async def aget_text_data(self, fields=["*"], filter="pk > 0"):
        # TODO(sl): make this truly async
        if self.vector_db.col:
            await asyncio.sleep(0.001)
            results = self.vector_db.col.query(expr=filter, output_fields=fields)
            # pks = self.vector_db.get_pks(expr=filter)
            # results = self.vector_db.get_by_ids(pks)
            return [
                {k: v for k, v in result.items() if k != "pk"} for result in results
            ]
        else:
            return []

    def search(self, search_query, top_k=1):
        search_results = self.vector_db.similarity_search(search_query, k=top_k)
        return [result.metadata for result in search_results]

    def query(self, search_query, top_k=1):
        search_results = self.vector_db.col.query(search_query, output_fields=["*"])
        return search_results

    def drop_data(self, expr="pk > 0"):
        if self.vector_db.col:
            self.vector_db.col.delete(expr=expr)
            self.vector_db.col.flush()

    def drop_data_filtered(self, filter):
        if self.vector_db.col:
            result = self.vector_db.col.delete(expr=filter)
            self.vector_db.col.flush()
            return result.delete_count
        return 0

    def drop_collection(self):
        self.vector_db = Milvus(
            embedding_function=self.embedding,
            connection_args=self.connection,
            collection_name=self.collection_name,
            auto_id=True,
            drop_old=True,
        )


if __name__ == "__main__":
    summarydb = MilvusDBTool("test")
    summarydb.drop_data()
    summarydb.add_summary(
        "A man wearing a yellow vest and a hard hat stands in front of a \
        conveyor belt, moves boxes around, and talks to the camera.",
        {"chunkIdx": 1, "start_time": 0, "end_time": 30},
    )
    summarydb.add_summary(
        "Two people in yellow hard hats are working in a warehouse. One is \
        stacking boxes while the other sorts items. ",
        {"chunkIdx": 2, "start_time": 30, "end_time": 60},
    )
    summarydb.add_summary(
        "An athletic man is seen running with a long pole and using it to vault \
        over a high beam. He then lands on a mat, celebrates his success, and the crowd cheers.",
        {"chunkIdx": 3, "start_time": 60, "end_time": 90},
    )

    chunks = summarydb.search("people in yellow vest and jumping?")
    for chunk in chunks:
        print(chunk["chunkIdx"])
    print(
        summarydb.get_text_data(
            fields=["text", "start_time", "end_time"],
            filter="0 <= start_time \
                                    and end_time <= 70",
        )
    )
