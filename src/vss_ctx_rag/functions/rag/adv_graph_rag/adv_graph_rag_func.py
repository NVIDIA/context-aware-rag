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

"""adv_graph_rag_func.py: File contains AdvGraphRAGFunc class"""

from typing import Optional
from copy import deepcopy
import asyncio
import json

from vss_ctx_rag.base import Function
from vss_ctx_rag.utils.ctx_rag_logger import TimeMeasure, logger
from vss_ctx_rag.utils.globals import LLM_TOOL_NAME
from .adv_graph_retrieval import AdvGraphRetrieval
from langchain_core.prompts import ChatPromptTemplate
from vss_ctx_rag.utils.utils import remove_think_tags


from pydantic import BaseModel, Field


class AnswerFormat(BaseModel):
    answer: str = Field(description="The answer to the question")
    updated_question: Optional[str] = Field(
        description="A reformulated question to get better database results"
    )
    confidence: float = Field(description="A confidence score between 0 and 1")


class AdvGraphRAGFunc(Function):
    """Advanced Graph RAG Function with iterative retrieval"""

    def setup(self):
        logger.info("Setting up AdvGraphRAGFunc")
        self.graph_db = self.get_tool("graph_db")
        self.chat_llm = self.get_tool(LLM_TOOL_NAME)
        self.top_k = self.get_param("params", "top_k", required=False)
        self.top_k = self.top_k if self.top_k else 10
        self.max_iterations = self.get_param("params", "max_iterations", required=False)
        self.max_iterations = self.max_iterations if self.max_iterations else 3
        self.max_ret_retries = self.get_param(
            "params", "max_ret_retries", required=False
        )
        self.max_ret_retries = self.max_ret_retries if self.max_ret_retries else 3
        self.confidence_threshold = self.get_param(
            "params", "confidence_threshold", required=False
        )
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.confidence_threshold = (
            self.confidence_threshold if self.confidence_threshold is not None else 0.7
        )
        self.chat_history = []

        self.retriever = AdvGraphRetrieval(
            llm=self.chat_llm,
            graph=self.graph_db,
            top_k=self.top_k,
            max_retries=self.max_ret_retries,
        )
        logger.info(f"Initialized retriever with top_k={self.top_k}")

        # Setup prompts with examples
        self.qa_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an AI assistant that answers questions based on the provided context.

             The context includes both retrieved information and relevant chat history.
             Use both the retrieved context and chat history to provide more accurate and contextual answers.
             If the fetched context is insufficient, formulate a better question to
             fetch more relevant information.

             You must respond in the following JSON format:
             {{\
                 "description": "A description of the answer",\
                 "answer": "your answer here or null if more info needed",\
                 "updated_question": "reformulated question to get better database results" or null,\
                 "confidence": 0.95 // number between 0-1\
             }}\

             Example 1 (when you have enough info):
             {{\
                 "description": "A description of the answer",\
                 "answer": "The worker dropped a box at timestamp 78.0 and it took 39 seconds to remove it",\
                 "updated_question": null,\
                 "confidence": 0.95\
             }}\

             Example 2 (when you need more info):
             {{\
                 "description": "A description of the answer",\
                 "answer": null,\
                 "updated_question": "What events occurred between timestamp 75 and 80?",\
                 "confidence": 0\
             }}\

             Only respond with valid JSON. Do not include any other text.
             """,
                ),
                ("human", "Question: {question}\nContext: {context}"),
            ]
        )
        logger.info("Initialized QA prompt template")

    async def acall(self, state: dict) -> dict:
        """Main QA function with iterative retrieval"""
        with TimeMeasure("AdvGraphRAG/call", "blue"):
            question = state.get("question", "").strip()
            if not question:
                logger.error("No question provided in state")
                state["response"] = "Please provide a question"
                return state

            if question.lower() == "/clear":
                logger.debug("Clearing chat history...")
                self.chat_history = []
                state["response"] = "Cleared chat history"
                return state

            logger.info(f"Processing question: {question}")
            logger.debug(f"Chat history length: {len(self.chat_history)}")

            # Initial retrieval
            context = await self.retriever.retrieve_relevant_context(question)
            retrieved_context = deepcopy(context)

            # If no context is found, assume that we did a temporal retrieval and
            # nothing turned up
            if not context:
                logger.info("No context found, assuming temporal retrieval")
                state["response"] = "I apologize, but I have no stored records that are either relevant or in the requested time range."
                return state

            # Add relevant chat history to context
            if self.chat_history:
                # Format chat history as additional context
                history_context = (
                    "\nRelevant chat history is included below. "
                    "Only use the chat history if it is relevant to the question and if "
                    "it seems like there is some context there that the user is referring to "
                    "in the current question.\n"
                )
                for entry in self.chat_history[-3:]:  # Consider last 3 interactions
                    q = entry.get("question", "")
                    a = entry.get("response", "")
                    if q and a:
                        history_context += f"User: {q}\nAssistant: {a}\n"
                context.append(history_context)

            logger.debug(f"Context: {context}")

            logger.info("Retrieved initial context with chat history")

            # Iterative retrieval and answering
            for i in range(self.max_iterations):
                logger.info(f"Starting iteration {i + 1}/{self.max_iterations}")
                # Get answer attempt
                response = await self.chat_llm.ainvoke(
                    self.qa_prompt.format(question=question, context=context)
                )

                logger.info(f"Response: {response.content}")
                retry_count = 0
                try:
                    while retry_count < self.max_iterations:
                        # Extract just the JSON content from the response
                        response_text = (
                            remove_think_tags(response.content)
                            if hasattr(response, "content")
                            else str(response)
                        )
                        # Remove any non-JSON text before or after
                        json_start = response_text.find("{")
                        json_end = response_text.rfind("}") + 1
                        if json_start >= 0 and json_end > json_start:
                            json_str = response_text[json_start:json_end]
                            logger.info(f"JSON string: {json_str}")
                            result = json.loads(json_str)
                            logger.info("Successfully parsed LLM response as JSON")
                            break
                        else:
                            logger.error(f"No JSON found in response: {response_text}")
                            retry_count += 1
                            if retry_count < self.max_iterations:
                                response = await self.chat_llm.ainvoke(
                                    self.qa_prompt.format(
                                        question=question, context=context
                                    )
                                )
                                continue
                            state["response"] = (
                                "I apologize, but I cannot provide a confident answer based on the available information."
                            )
                            return state
                except Exception as e:
                    logger.error(f"Failed to parse LLM response as JSON: {response}")
                    logger.error(f"Parse error: {str(e)}")
                    continue

                logger.debug(f"Result: {result}")

                # If we have a confident answer, return it
                if (
                    result.get("answer")
                    and result.get("confidence", 0) > self.confidence_threshold
                ):
                    logger.info(
                        f"Found confident answer with confidence {result['confidence']}"
                    )
                    state["response"] = result["answer"]
                    state["confidence"] = result["confidence"]

                    # Store the current Q&A pair in history before returning
                    current_interaction = {
                        "question": question,
                        "response": result["answer"],
                        "confidence": result["confidence"],
                    }
                    self.chat_history.append(current_interaction)

                    # Append relevant documents to the top of the answer
                    if len(retrieved_context) > 0:
                        snippet_len = 200
                        citation_text = "**Sources:**\n"
                        for doc in retrieved_context:
                            sid = doc.metadata.get("stream_id", "unknown-source")
                            snippet = doc.page_content[:snippet_len]
                            if len(doc.page_content) > snippet_len:
                                snippet += "..."
                            citation_text += f"\n - **{sid}** - *\"{snippet}\"*\n"
                        citation_text += "\n - - - - - \n"
                        state["response"] = f"{citation_text}{state['response']}"
                    return state

                # If we need more info, try to retrieve it
                if result.get("updated_question"):
                    logger.info(f"Need more info: {result['updated_question']}")
                    new_context = []
                    for info_need in [result["updated_question"]]:
                        # Use the retriever to get additional context
                        additional_docs = (
                            await self.retriever.retrieve_relevant_context(info_need)
                        )
                        if additional_docs is not None:
                            new_context.extend(additional_docs)
                            logger.info(f"Retrieved additional context for: {info_need}")

                    # Add new context and continue
                    context.extend(new_context)
                    continue

            # If we get here, we couldn't get a confident answer
            logger.info("Could not find confident answer after max iterations")
            state["response"] = (
                "I apologize, but I cannot provide a confident answer based on the available information."
            )
            state["confidence"] = 0.0
            # Store the current Q&A pair in history before returning
            current_interaction = {
                "question": question,
                "response": state["response"],
                "confidence": state.get("confidence", 0.0),
            }
            self.chat_history.append(current_interaction)
            return state

    async def areset(self, state: dict):
        """Reset the function state"""
        logger.info("Resetting AdvGraphRAGFunc state")
        self.chat_history = []
        await asyncio.sleep(0.01)
