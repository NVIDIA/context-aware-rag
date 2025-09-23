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

import re
import threading
from functools import partial
from typing import Any, List

from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder
from langchain_core.runnables import RunnableBranch

from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.prompts import (
    CHAT_SYSTEM_IMAGE_PREFIX,
    CHAT_SYSTEM_TEMPLATE_PREFIX,
    CHAT_SYSTEM_TEMPLATE_SUFFIX,
    QUESTION_TRANSFORM_TEMPLATE,
)
from vss_ctx_rag.utils.utils import remove_lucene_chars, remove_think_tags


class GraphRetrieval:
    """
    Helper base class for graph retrieval.

    Steps for retrieval:
    1. Transform the question using chat history
    2. Retrieve the documents using the transformed question
        a. First run a similarity search on the chunk nodes using the transformed question
        b. Run graph walk on the top k results from the similarity search
        c. Retrieve related entities from the graph
    3. Format the documents
    4. Get the response using the chat history and the formatted documents
    """

    def __init__(
        self,
        llm,
    ) -> None:
        """
        Initialize the GraphRetrieval class.

        Args:
            llm: Language model.

        Instance variables:
            self.chat_llm: LLM Tool.
            self.chat_history: Chat history.

        Returns:
            None
        """
        self.chat_llm = llm
        self.chat_history = ChatMessageHistory()
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
        self.chat_history_summarization_chain = summarization_prompt | self.chat_llm
        self.question_transform_chain = self.create_question_transform_chain()
        self.regex_object = re.compile(r"<(\d+[.]\d+)>")

    def create_question_transform_chain(self) -> RunnableBranch:
        """
        Create the question transform chain.
        This chain is used to transform the question using the chat history.

        Returns:
            RunnableBranch: The question transform chain.
        """
        with Metrics(
            "GraphRetrieval/CreateQuestionTransformChain",
            "blue",
            span_kind=Metrics.SPAN_KIND["CHAIN"],
        ) as tm:
            try:
                logger.info("Starting to create question transform chain")

                query_transform_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", QUESTION_TRANSFORM_TEMPLATE),
                        MessagesPlaceholder(variable_name="messages"),
                    ]
                )

                query_transforming_retriever_chain = RunnableBranch(
                    (
                        lambda x: len(x.get("messages", [])) == 1,
                        # If only one message, return it directly after cleaning
                        (lambda x: x["messages"][-1].content)
                        | StrOutputParser()
                        | remove_think_tags
                        | remove_lucene_chars
                        | partial(
                            self.debug_logs,
                            message="Cleaned single message:",
                        )
                        | StrOutputParser(),
                    ),
                    # If multiple messages, transform the query using the history
                    query_transform_prompt
                    | partial(self.debug_logs, message="Query Transform Input")
                    | self.chat_llm
                    | partial(self.debug_logs, message="Query Transform Output")
                    | StrOutputParser()
                    | remove_think_tags
                    | remove_lucene_chars
                    | partial(
                        self.debug_logs,
                        message="Cleaned transformed query:",
                    )
                    | StrOutputParser(),
                ).with_config(run_name="chat_retriever_chain")

                logger.info("Successfully created question transform chain")
                return query_transforming_retriever_chain

            except Exception as e:
                tm.error(e)
                logger.error(
                    f"Error creating question transform chain: {e}", exc_info=True
                )
                raise

    def add_message(self, message: BaseMessage) -> None:
        """
        Add a message to the chat history.

        Args:
            message: The message to add.

        Returns:
            None
        """
        self.chat_history.add_message(message)

    def clear_chat_history(self) -> None:
        """
        Clear the chat history.

        Args:
            None

        Returns:
            None
        """
        self.chat_history.clear()

    def summarize_chat_history(self) -> None:
        """
        Summarize the chat history in a separate thread.

        Args:
            None

        Returns:
            None
        """

        summarization_thread = threading.Thread(
            target=self.summarize_chat_history_and_log,
            args=(self.chat_history.messages,),
        )
        summarization_thread.start()

    async def get_response(
        self,
        question: str,
        formatted_docs: List[str],
        image_list_base64: List[str] = [],
        response_method: str | None = None,
        response_schema: dict | None = None,
    ) -> str | dict:
        """
        Get the response from the chat history.

        Args:
            question: The question to answer.
            formatted_docs: The formatted documents.

        Returns:
            str: The response.
        """
        with Metrics(
            "GraphRetrieval/GetResponse", "blue", span_kind=Metrics.SPAN_KIND["CHAIN"]
        ):
            if image_list_base64:
                image_message_list = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": ("data:image/jpeg;base64," + image_list_base64[j]),
                        },
                    }
                    for j in range(len(image_list_base64))
                ]

                messages = [
                    {
                        "role": "system",
                        "content": CHAT_SYSTEM_IMAGE_PREFIX
                        + CHAT_SYSTEM_TEMPLATE_SUFFIX,
                    },
                    {
                        "role": "user",
                        "content": [
                            *image_message_list,
                            {
                                "type": "text",
                                "text": f"Question: {question}\nVideo Summary: {formatted_docs}",
                            },
                        ],
                    },
                ]
            else:
                messages = [
                    {
                        "role": "system",
                        "content": CHAT_SYSTEM_TEMPLATE_PREFIX
                        + CHAT_SYSTEM_TEMPLATE_SUFFIX,
                    },
                    {
                        "role": "user",
                        "content": f"Question: {question}\nVideo Summary: {formatted_docs}",
                    },
                ]

            if response_method is not None and response_method not in [
                "json_mode",
                "text",
                "function_calling",
            ]:
                raise ValueError(
                    f"Invalid response_method: {response_method}, has to be one of json_mode, text, or function_calling"
                )

            if response_method is not None and response_method != "text":
                if response_method == "json_mode" and "json" not in question.lower():
                    raise ValueError("JSON mode requires 'json' in the question")

                logger.info(
                    f"AI response with structured output: {response_method}, {response_schema}"
                )
                llm = self.chat_llm.with_structured_output(
                    method=response_method,
                    schema=response_schema,
                )
                response = await llm.ainvoke(messages)
            else:
                llm = self.chat_llm
                response = await llm.ainvoke(messages)
                response = remove_think_tags(response.content)
                response = self.regex_object.sub(r"\g<1>", response)

            logger.info(f"AI response: {response}")
            return response

    def summarize_chat_history_and_log(
        self, stored_messages: List[BaseMessage]
    ) -> bool:
        """
        Summarize the chat history and log the result.

        Args:
            stored_messages: The chat history.

        Returns:
            bool: Whether the summarization was successful.
        """

        logger.info("Starting summarizing chat history in a separate thread.")
        if not stored_messages:
            logger.info("No messages to summarize.")
            return False

        try:
            with Metrics("GraphRetrieval/SummarizeChat", "yellow"):
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

    def debug_logs(self, state, message: str = "Debug logs") -> Any:
        """
        Log the state inbetween the chain.

        Args:
            state: The state to log.
            message: The message to log.
        """
        logger.debug(f"{message}: {state}")
        return state
