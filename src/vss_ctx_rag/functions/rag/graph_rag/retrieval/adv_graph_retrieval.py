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

import asyncio
import json
import traceback
from copy import deepcopy
from typing import Optional, ClassVar, Dict, List

from pydantic import BaseModel, Field
from langchain_core.documents import Document
import yaml

from vss_ctx_rag.functions.rag.graph_rag.constants import get_adv_chat_template_image
from vss_ctx_rag.functions.rag.graph_rag.retrieval.adv_base import AdvGraphRetrieval
from vss_ctx_rag.functions.rag.graph_rag.retrieval.graph_retrieval_base import (
    GraphRetrievalBaseFunc,
)
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.utils import remove_think_tags


class AnswerFormat(BaseModel):
    answer: str = Field(description="The answer to the question")
    updated_question: Optional[str] = Field(
        description="A reformulated question to get better database results"
    )
    confidence: float = Field(description="A confidence score between 0 and 1")
    need_image_data: bool = Field(
        default=False,
        description="Whether visual data is needed to answer the question",
    )


@register_function_config("cot_retrieval")
class AdvGraphRAGConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "neo4j": ["db"],
        "gnn": ["gnn"],
    }

    params: RetrieverConfig.RetrieverParams


@register_function(config=AdvGraphRAGConfig)
class AdvGraphRAGFunc(GraphRetrievalBaseFunc):
    """Advanced Image Graph RAG Function with iterative retrieval"""

    def setup(self):
        super().setup()
        self.max_iterations = self.get_param("max_iterations", default=3)
        self.max_ret_retries = self.get_param("max_ret_retries", default=3)
        self.confidence_threshold = self.get_param("confidence_threshold", default=0.7)

        self.chat_history = []

        self.image = self.get_param("image", default=False)
        self.prompt_config_path = self.get_param("prompt_config_path", default=None)
        self.prompt_config = None
        if self.prompt_config_path:
            try:
                with open(self.prompt_config_path, "r") as f:
                    self.prompt_config = yaml.safe_load(f)
            except Exception as e:
                logger.error(f"Error loading prompt config: {str(e)}")
                self.prompt_config = None

        self.retriever = AdvGraphRetrieval(
            llm=self.chat_llm,
            graph=self.graph_db,
            top_k=self.top_k,
            max_retries=self.max_ret_retries,
            multi_channel=self.multi_channel,
            uuid=self.uuid,
            prompt_config=self.prompt_config,
        )
        logger.info(f"Initialized retriever with top_k={self.top_k}")
        # Simplified citation configuration
        logger.info("Attempting to load citations configuration...")
        self.citations_config = self.get_param("citations", default={})
        logger.info(f"Raw citations_config retrieved: {self.citations_config}")
        self.citations_enabled = self.citations_config.get("enabled", False)
        self.snippet_length = self.citations_config.get("snippet_length", 200)

        self.qa_system_prompt = get_adv_chat_template_image(
            self.prompt_config, self.image
        )
        logger.info("Initialized unified QA prompt template")

    async def acall(self, state: dict) -> dict:
        """Main QA function with iterative retrieval"""
        with Metrics(
            "AdvImgGraphRAG/call", "blue", span_kind=Metrics.SPAN_KIND["AGENT"]
        ) as tm:
            tm.input({"state": state})
            try:
                question = state.get("question", "").strip()
                if not question:
                    logger.debug("No question provided in state")
                    state["response"] = "Please provide a question"
                    return state

                if question.lower() == "/clear":
                    logger.debug("Clearing chat history...")
                    self.chat_history = []
                    state["response"] = "Cleared chat history"
                    return state

                if (
                    state.get("response_method") is not None
                    or state.get("response_schema") is not None
                ):
                    logger.warning(
                        "Advanced Graph RAG does not support structured mode, ignoring response_method and response_schema"
                    )

                logger.info(f"Processing question: {question}")
                logger.debug(f"Chat history length: {len(self.chat_history)}")

                # Initial retrieval
                context = await self.retriever.retrieve_relevant_context(question)
                retrieved_context = deepcopy(context)
                # If no context is found, assume that we did a temporal retrieval and
                # nothing turned up
                if not context:
                    logger.info("No context found assuming temporal retrieval")
                    state["response"] = (
                        "I apologize, but I have no stored records that are either relevant or in the requested time range."
                    )
                    return state

                # Add relevant chat history to text context
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
                    # Convert history context to Document object to maintain consistent format
                    history_doc = Document(
                        page_content=history_context,
                        metadata={"source": "chat_history", "type": "context"},
                    )
                    context.append(history_doc)

                logger.debug(f"Context: {context}")
                logger.info("Retrieved initial context with chat history")

                # Initial empty image context
                image_message_list = []
                image_list_base64 = []

                # Iterative retrieval and answering
                for i in range(self.max_iterations):
                    logger.info(f"Starting iteration {i + 1}/{self.max_iterations}")

                    if image_message_list:
                        messages = [
                            {"role": "system", "content": self.qa_system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    *image_message_list,
                                    {
                                        "type": "text",
                                        "text": f"Question: {question}\nContext: {context} ",
                                    },
                                ],
                            },
                        ]
                    else:
                        messages = [
                            {"role": "system", "content": self.qa_system_prompt},
                            {
                                "role": "user",
                                "content": f"Question: {question}\nContext: {context} ",
                            },
                        ]
                    response = await self.chat_llm.ainvoke(messages)

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
                                logger.error(
                                    f"No JSON found in response: {response_text}"
                                )
                                retry_count += 1
                                if retry_count < self.max_iterations:
                                    if image_message_list:
                                        messages = [
                                            {
                                                "role": "system",
                                                "content": self.qa_system_prompt,
                                            },
                                            {
                                                "role": "user",
                                                "content": [
                                                    *image_message_list,
                                                    {
                                                        "type": "text",
                                                        "text": f"Question: {question}\nContext: {context} ",
                                                    },
                                                ],
                                            },
                                        ]
                                    else:
                                        messages = [
                                            {
                                                "role": "system",
                                                "content": self.qa_system_prompt,
                                            },
                                            {
                                                "role": "user",
                                                "content": f"Question: {question}\nContext: {context} ",
                                            },
                                        ]
                                    response = await self.chat_llm.ainvoke(messages)
                                    continue

                                state["response"] = (
                                    "I apologize, but I cannot provide a confident answer based on the available information."
                                )
                                return state
                    except Exception as e:
                        logger.error(
                            f"Failed to parse LLM response as JSON: {response}"
                        )
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
                        state["source_docs"] = [
                            {
                                "metadata": getattr(doc, "metadata", {}),
                                "page_content": getattr(doc, "page_content", str(doc)),
                            }
                            for doc in context
                        ]
                        state["formatted_docs"] = [
                            getattr(i, "page_content", str(i)) for i in context
                        ]

                        current_interaction = {
                            "question": question,
                            "response": result["answer"],
                            "confidence": result["confidence"],
                        }
                        self.chat_history.append(current_interaction)

                        # Append relevant documents to the top of the answer
                        if self.citations_enabled and len(retrieved_context) > 0:
                            snippet_len = self.snippet_length
                            citation_text = "**Sources:**\n"
                            for doc in retrieved_context:
                                page_content = getattr(doc, "page_content", str(doc))
                                snippet = page_content[:snippet_len]
                                if len(page_content) > snippet_len:
                                    snippet += "..."
                                citation_text += f'\n --- *"{snippet}"*\n'
                            citation_text += "\n - - - - - \n"
                            state["citations"] = citation_text
                            state["response"] = f"{citation_text}{state['response']}"
                        return state

                    # Check if visual data is needed (only when image features are enabled)
                    if self.image and result.get(
                        "need_image_data", "false"
                    ).lower() in [
                        "true",
                        "yes",
                        "1",
                    ]:
                        logger.info("Visual data needed, processing images")

                        # Process image data
                        image_list_base64 = await self.extract_images(context)

                        # Now that we have image data, process a unified response using both text and image contexts
                        image_message_list = [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": (
                                        "data:image/jpeg;base64," + image_list_base64[j]
                                    ),
                                },
                            }
                            for j in range(len(image_list_base64))
                        ]
                        continue

                    # If we need more info, try to retrieve it
                    if result.get("updated_question"):
                        logger.info(f"Need more info: {result['updated_question']}")
                        new_context = []
                        for info_need in [result["updated_question"]]:
                            # Use the retriever to get additional context
                            additional_docs = (
                                await self.retriever.retrieve_relevant_context(
                                    info_need
                                )
                            )
                            if additional_docs:
                                new_context.extend(additional_docs)
                                logger.info(
                                    f"Retrieved additional context for: {info_need}"
                                )

                        context.extend(new_context)

                # If we get here, we couldn't get a confident answer
                logger.info("Could not find confident answer after max iterations")
                state["response"] = (
                    "I apologize, but I cannot provide a confident answer based on the available information."
                )
                state["confidence"] = 0.0
                state["source_docs"] = [
                    {
                        "metadata": getattr(doc, "metadata", {}),
                        "page_content": getattr(doc, "page_content", str(doc)),
                    }
                    for doc in context
                ]
                state["formatted_docs"] = [
                    getattr(i, "page_content", str(i)) for i in context
                ]
                # Store the current Q&A pair in history before returning
                current_interaction = {
                    "question": question,
                    "response": state["response"],
                    "confidence": state.get("confidence", 0.0),
                }
                self.chat_history.append(current_interaction)

            except Exception as e:
                tm.error(e)
                logger.error(traceback.format_exc())
                logger.error("Error in AdvGraphRAGFunc %s", str(e))
                state["error"] = str(e)
            finally:
                tm.output({"state": state})
            return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        pass

    async def areset(self, state: dict):
        """Reset the function state"""
        logger.info("Resetting AdvGraphRAGFunc state")
        self.chat_history = []
        await asyncio.sleep(0.01)
