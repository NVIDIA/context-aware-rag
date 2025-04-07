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

import os
import traceback
import threading
from functools import partial

from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever

from langchain.retrievers.document_compressors import (
    EmbeddingsFilter,
    DocumentCompressorPipeline,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter

from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.utils.utils import remove_think_tags, remove_lucene_chars
from vss_ctx_rag.functions.rag.graph_rag.constants import (
    CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
    QUESTION_TRANSFORM_TEMPLATE,
    CHAT_SYSTEM_TEMPLATE,
    VECTOR_GRAPH_SEARCH_QUERY,
    VECTOR_SEARCH_TOP_K,
    CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
)
from vss_ctx_rag.tools.storage.neo4j_db import Neo4jGraphDB


class GraphRetrieval:
    def __init__(
        self,
        llm,
        graph: Neo4jGraphDB,
        multi_channel=False,
        uuid="default",
        top_k=None,
    ):
        self.chat_llm = llm
        self.graph_db = graph
        self.chat_history = ChatMessageHistory()
        self.top_k = top_k
        self.uuid = uuid
        self.multi_channel = multi_channel
        summarization_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "system",
                    "Summarize the above chat messages into a concise message, \
                    focusing on key points and relevant details that could be useful for future conversations. \
                    Exclude all introductions and extraneous information.",
                ),
            ]
        )
        self.chat_history_summarization_chain = summarization_prompt | llm
        question_answering_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CHAT_SYSTEM_TEMPLATE),
                MessagesPlaceholder(variable_name="messages"),
                ("human", "User question: {input}"),
            ]
        )

        self.question_answering_chain = question_answering_prompt | self.chat_llm
        neo_4j_retriever = self.get_neo4j_retriever()
        self.doc_retriever = self.create_document_retriever_chain(neo_4j_retriever)

    def create_neo4j_vector(self, index_name: str):
        try:
            retrieval_query = VECTOR_GRAPH_SEARCH_QUERY
            keyword_index = "keyword"
            node_label = "Chunk"
            embedding_node_property = "embedding"
            text_node_properties = ["text"]

            vector_db = Neo4jVector.from_existing_graph(
                embedding=self.graph_db.embeddings,
                index_name=index_name,
                retrieval_query=retrieval_query,
                graph=self.graph_db.graph_db,
                search_type="hybrid",
                node_label=node_label,
                embedding_node_property=embedding_node_property,
                text_node_properties=text_node_properties,
                keyword_index_name=keyword_index,
            )
            logger.info(
                f"Successfully retrieved Neo4jVector Fulltext index '{index_name}' and keyword index '{keyword_index}'"
            )
        except Exception as e:
            logger.error(f"Error retrieving Neo4jVector index {index_name} : {e}")
            raise
        return vector_db

    def debug_logs(self, state, message: str = "Debug logs"):
        logger.debug(f"{message}: {state}")
        return state

    def get_neo4j_retriever(self):
        with TimeMeasure("GraphRetrieval/Neo4jRetriever", "blue"):
            try:
                index_name = "vector"
                vector_db = self.create_neo4j_vector(index_name)
                search_k = self.top_k or VECTOR_SEARCH_TOP_K
                retriever = vector_db.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={
                        "k": search_k,
                        "score_threshold": CHAT_SEARCH_KWARG_SCORE_THRESHOLD,
                        "params": {"uuid": self.uuid}
                        if not self.multi_channel
                        else {"uuid": None},
                    },
                )
                return retriever
            except Exception as e:
                logger.error(
                    f"Error retrieving Neo4jVector index  {index_name} or creating retriever: {e}"
                )
                raise Exception(
                    f"An error occurred while retrieving the Neo4jVector index or creating the retriever. \
                    Please drop and create a new vector index '{index_name}': {e}"
                ) from e

    def create_document_retriever_chain(self, neo_4j_retriever):
        with TimeMeasure("GraphRetrieval/CreateDocRetChain", "blue"):
            try:
                logger.info("Starting to create document retriever chain")

                query_transform_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", QUESTION_TRANSFORM_TEMPLATE),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                output_parser = StrOutputParser()
                embeddings_dimension = int(
                    os.environ.get("CA_RAG_EMBEDDINGS_DIMENSION", 1024)
                )
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=embeddings_dimension,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", "\n-", ".", ";", ",", " ", ""],
                )
                embeddings_filter = EmbeddingsFilter(
                    embeddings=self.graph_db.embeddings,
                    similarity_threshold=CHAT_EMBEDDING_FILTER_SCORE_THRESHOLD,
                )
                pipeline_compressor = DocumentCompressorPipeline(
                    transformers=[splitter, embeddings_filter]
                )
                compression_retriever = ContextualCompressionRetriever(
                    base_compressor=pipeline_compressor, base_retriever=neo_4j_retriever
                )
                query_transforming_retriever_chain = RunnableBranch(
                    (
                        lambda x: len(x.get("messages", [])) == 1,
                        (lambda x: x["messages"][-1].content)
                        | output_parser
                        | remove_think_tags
                        | remove_lucene_chars
                        | partial(
                            self.debug_logs,
                            message="After removing think tags and lucene chars:",
                        )
                        | compression_retriever,
                    ),
                    query_transform_prompt
                    | partial(self.debug_logs, message="Query Transform Input")
                    | self.chat_llm
                    | partial(self.debug_logs, message="Query Transform Output")
                    | output_parser
                    | remove_think_tags
                    | remove_lucene_chars
                    | partial(
                        self.debug_logs,
                        message="After removing think tags and lucene chars:",
                    )
                    | compression_retriever,
                ).with_config(run_name="chat_retriever_chain")

                logger.info("Successfully created document retriever chain")
                return query_transforming_retriever_chain

            except Exception as e:
                logger.error(
                    f"Error creating document retriever chain: {e}", exc_info=True
                )
                raise

    def retrieve_documents(self):
        with TimeMeasure("Retrive documents", "green"):
            try:
                docs = self.doc_retriever.invoke(
                    {"messages": self.chat_history.messages}
                )

            except Exception as e:
                logger.error(traceback.format_exc())
                error_message = f"Error retrieving documents: {str(e)}"
                logger.error(error_message)
                raise RuntimeError(error_message)
            return docs

    def fetch_documents(self, filter):
        """Fetch documents from graph database using a cypher query filter.

        Args:
            filter (str): Cypher query filter to apply

        Returns:
            list: Retrieved documents matching the filter

        Raises:
            Exception: If there is an error retrieving documents
        """
        with TimeMeasure("Retrieve documents with filter", "green"):
            try:
                docs = self.graph_db.run_cypher_query(filter)
                return docs
            except Exception as e:
                logger.error(f"Error retrieving documents: {e}")
                logger.error(traceback.format_exc())
                raise

    def format_documents(self, documents):
        prompt_token_cutoff = 28
        sorted_documents = sorted(
            documents,
            key=lambda doc: doc.state.get("query_similarity_score", 0),
            reverse=True,
        )
        documents = sorted_documents[:prompt_token_cutoff]

        formatted_docs = list()
        sources = set()
        entities = dict()

        for doc in documents:
            try:
                source = doc.metadata.get("source", "unknown")
                sources.add(source)

                entities = (
                    doc.metadata["entities"]
                    if "entities" in doc.metadata.keys()
                    else entities
                )

                formatted_doc = (
                    f"Document start\nContent: {doc.page_content}\nDocument end\n"
                )
                formatted_docs.append(formatted_doc)

            except Exception as e:
                logger.error(f"Error formatting document: {e}")

        return "\n\n".join(formatted_docs), sources, entities

    def get_sources_and_chunks(self, sources_used, docs):
        chunkdetails_list = []
        sources_used_set = set(sources_used)
        seen_ids_and_scores = set()

        for doc in docs:
            try:
                source = doc.metadata.get("source")
                chunkdetails = doc.metadata.get("chunkdetails", [])

                if source in sources_used_set:
                    for chunkdetail in chunkdetails:
                        id = chunkdetail.get("id")
                        score = round(chunkdetail.get("score", 0), 4)

                        id_and_score = (id, score)

                        if id_and_score not in seen_ids_and_scores:
                            seen_ids_and_scores.add(id_and_score)
                            chunkdetails_list.append({**chunkdetail, "score": score})

            except Exception as e:
                logger.error(f"Error processing document: {e}")

        result = {
            "sources": sources_used,
            "chunkdetails": chunkdetails_list,
        }
        return result

    def process_documents(self, docs):
        with TimeMeasure("chat/process documents", "green"):
            try:
                formatted_docs, sources, entitydetails = self.format_documents(docs)
                result = {"sources": list(), "nodedetails": dict(), "entities": dict()}
                node_details = {"chunkdetails": list(), "entitydetails": list()}
                entities = {"entityids": list(), "relationshipids": list()}

                sources_and_chunks = self.get_sources_and_chunks(sources, docs)
                result["sources"] = sources_and_chunks["sources"]
                node_details["chunkdetails"] = sources_and_chunks["chunkdetails"]
                entities.update(entitydetails)

            except Exception as e:
                logger.error(f"Error processing documents: {e}")
                raise
            return formatted_docs

    def add_message(self, message):
        self.chat_history.add_message(message)

    def clear_chat_history(self):
        self.chat_history.clear()

    def summarize_chat_history(self):
        summarization_thread = threading.Thread(
            target=self.summarize_chat_history_and_log,
            args=(self.chat_history.messages,),
        )
        summarization_thread.start()

    def get_response(self, question, formatted_docs):
        return self.question_answering_chain.invoke(
            {
                "messages": self.chat_history.messages[:-1],
                "context": formatted_docs,
                "input": question,
            }
        )

    def summarize_chat_history_and_log(self, stored_messages):
        logger.info("Starting summarizing chat history in a separate thread.")
        if not stored_messages:
            logger.info("No messages to summarize.")
            return False

        try:
            with TimeMeasure("GraphRetrieval/SummarizeChat", "yellow"):
                summary_message = self.chat_history_summarization_chain.invoke(
                    {"chat_history": stored_messages}
                )
                summary_message.content = remove_think_tags(summary_message.content)

                with threading.Lock():
                    self.chat_history.clear()
                    self.chat_history.add_user_message(
                        "Our current conversation summary till now: "
                        + summary_message.content
                    )
                    logger.debug(
                        f"after summarization chat history: {self.chat_history.messages}"
                    )
                return True

        except Exception as e:
            logger.error(
                f"An error occurred while summarizing messages: {e}", exc_info=True
            )
            return False
