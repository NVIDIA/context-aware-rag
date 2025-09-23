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

"""vlm_retrieval_func.py: File contains VLM-only retrieval function"""

import asyncio
from re import compile
import traceback
from typing import Optional, List, Dict, Any, ClassVar

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from vss_ctx_rag.functions.rag.prompt import IMAGE_QA_PROMPT
from vss_ctx_rag.tools.image.image_fetcher import ImageFetcher

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.models.function_models import (
    register_function,
    register_function_config,
)
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.utils.globals import (
    DEFAULT_RAG_TOP_K,
    DEFAULT_MULTI_CHANNEL,
    DEFAULT_NUM_FRAMES_PER_CHUNK,
)
from vss_ctx_rag.functions.rag.config import RetrieverConfig
from vss_ctx_rag.utils.utils import format_docs
from vss_ctx_rag.utils.prompts import BASIC_QA_PROMPT
from langchain_core.prompts import PromptTemplate


@register_function_config("vlm_retrieval")
class VLMRetrievalConfig(RetrieverConfig):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {
        "llm": ["llm"],
        "db": ["db"],
        "image_fetcher": ["image_fetcher"],
    }

    class VLMRetrievalParams(RetrieverConfig.RetrieverParams):
        multi_channel: Optional[bool] = DEFAULT_MULTI_CHANNEL
        top_k: Optional[int] = DEFAULT_RAG_TOP_K
        num_frames_per_chunk: Optional[int] = DEFAULT_NUM_FRAMES_PER_CHUNK

    params: VLMRetrievalParams


@register_function(config=VLMRetrievalConfig)
class VLMRetrievalFunc(Function):
    """VLM-only retrieval function that uses semantic similarity to retrieve documents,
    extracts images using the image fetcher tool, and then uses the VLM (LLM) to answer questions about the images."""

    config: dict
    output_parser = StrOutputParser()
    db: StorageTool
    vlm: Any  # This is actually the LLM tool used as VLM
    image_fetcher: ImageFetcher

    def setup(self):
        """Setup the VLM retrieval function."""
        # Get tools
        self.vlm = self.get_tool("vlm")  # VLM is an LLM tool supporting images.
        self.db = self.get_tool("db")
        self.image_fetcher = self.get_tool("image_fetcher")

        # Get parameters
        self.top_k = self.get_param("top_k", default=DEFAULT_RAG_TOP_K)
        self.num_frames_per_chunk = self.get_param(
            "num_frames_per_chunk", default=DEFAULT_NUM_FRAMES_PER_CHUNK
        )
        self.multi_channel = self.get_param("multi_channel", default=False)
        self.regex_object = compile(r"<(\d+[.]\d+)>")

        self.retriever = self.db.as_retriever(
            search_kwargs={
                "expr": "content_metadata['doc_type'] == 'caption'",
                "k": self.top_k,
            }
        )

        # Create the semantic similarity chain
        self.qa_partial = {
            "context": self.retriever | format_docs,
            "input": RunnablePassthrough(),
        } | PromptTemplate.from_template(BASIC_QA_PROMPT)

    def _get_retrieved_documents(self, question: str) -> List[Document]:
        """Retrieve documents from the database using semantic similarity."""
        try:
            docs = self.retriever.invoke(question)
            logger.debug(f"Retrieved {len(docs)} documents")
            logger.info(f"Docs: {docs}")
            return docs
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []

    def _extract_images_from_documents(
        self, documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract images from the retrieved documents using the image fetcher along with their metadata."""

        def extract_metadata(metadata: dict) -> dict:
            def ns_to_hms(ns):
                if ns is None:
                    return None
                seconds = int(ns) // 1_000_000_000
                hours = seconds // 3600
                minutes = (seconds % 3600) // 60
                secs = seconds % 60
                return f"{hours:02}:{minutes:02}:{secs:02}"

            start_time_ns = metadata.get("start_pts", None)
            end_time_ns = metadata.get("end_pts", None)
            start_time = ns_to_hms(start_time_ns)
            end_time = ns_to_hms(end_time_ns)
            start_ntp = metadata.get("start_ntp", None)
            end_ntp = metadata.get("end_ntp", None)
            asset_dir = metadata.get("asset_dir", None)
            return start_time, end_time, start_ntp, end_ntp, asset_dir

        images_with_metadata = []
        try:
            for doc in documents:
                # Extract asset directory from document metadata
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                # Test if the metadata has start_time and end_time or it's inside content_metadata
                if "content_metadata" in metadata:
                    metadata = metadata.get("content_metadata", {})

                start_time, end_time, start_ntp, end_ntp, asset_dir = extract_metadata(
                    metadata
                )

                if asset_dir:
                    # Get base64 encoded images from the asset directory
                    image_list_base64 = self.image_fetcher.get_image_base64(
                        asset_dir, self.num_frames_per_chunk
                    )
                    if image_list_base64:
                        # Create metadata dict for this set of images
                        chunk_metadata = {
                            "start_time": start_time,
                            "end_time": end_time,
                            "start_ntp": start_ntp,
                            "end_ntp": end_ntp,
                            "asset_dir": asset_dir,
                        }
                        images_with_metadata.append(
                            {"images": image_list_base64, "metadata": chunk_metadata}
                        )

                        logger.debug(
                            f"Extracted {len(image_list_base64)} images from asset_dir: {asset_dir} with metadata: {chunk_metadata}"
                        )
                    else:
                        logger.debug(f"No images found for asset_dir: {asset_dir}")
                else:
                    logger.debug("No asset_dir found in document metadata")
        except Exception as e:
            logger.error(f"Error extracting images from documents: {e}")

        return images_with_metadata

    async def _get_vlm_response(
        self, question: str, images_with_metadata: List[Dict[str, Any]]
    ) -> str:
        """Get response from VLM (LLM) about the images."""
        try:
            if not images_with_metadata:
                return "No images found to analyze."

            # Prepare image messages for the VLM with metadata
            image_message_list = []
            for img_data in images_with_metadata:
                imgs = img_data["images"]
                metadata = img_data["metadata"]

                # Create timestamp description
                timestamp_info = ""
                if (
                    metadata.get("start_time") is not None
                    and metadata.get("end_time") is not None
                ):
                    timestamp_info = f" (Timestamp: {metadata['start_time']} to {metadata['end_time']})"
                elif metadata.get("start_ntp") and metadata.get("end_ntp"):
                    timestamp_info = (
                        f" (Time: {metadata['start_ntp']} to {metadata['end_ntp']})"
                    )

                # Combine all metadata info
                metadata_text = f"{timestamp_info}".strip()

                # If we have metadata, add it as a text message before the image
                if metadata_text:
                    image_message_list.append(
                        {"type": "text", "text": f"Image metadata: {metadata_text}\n"}
                    )
                    logger.info(f"Image metadata: {metadata_text}")

                for img in imgs:
                    image_message_list.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img}",
                            },
                        }
                    )

            # Prepare the prompt for the VLM
            messages = [
                {
                    "role": "system",
                    "content": IMAGE_QA_PROMPT,
                },
                {
                    "role": "user",
                    "content": [
                        *image_message_list,
                        {"type": "text", "text": f"Question: {question}. Answer:\n"},
                    ],
                },
            ]

            # Get response from VLM
            response = await self.vlm.ainvoke(messages)
            return response.content

        except Exception as e:
            logger.error(f"Error getting VLM response: {e}")
            return f"Error analyzing images: {str(e)}"

    async def get_vlm_response(
        self,
        question: str,
        response_method: str | None = None,
        response_schema: dict | None = None,
        **kwargs,
    ):
        """Get VLM response for the given question and images."""
        if response_method is not None and response_method not in [
            "json_mode",
            "text",
            "function_calling",
        ]:
            raise ValueError(
                f"Invalid response_method: {response_method}, has to be one of json_mode, text, or function_calling"
            )

        if response_method is not None and response_method != "text":
            logger.debug(
                f"Changing response format to:{response_method}, schema:{response_schema}"
            )

            chain = self.qa_partial | self.vlm.with_structured_output(
                method=response_method, schema=response_schema
            )

        else:
            chain = (
                self.qa_partial
                | self.vlm
                | self.output_parser
                | RunnableLambda(lambda x: self.regex_object.sub(r"\g<1>", x))
            )

        result = await asyncio.get_running_loop().run_in_executor(
            None,
            lambda: chain.invoke(question),
        )
        return result

    async def acall(self, state: dict):
        """VLM-only retrieval function call.

        Args:
            state: State dictionary containing:
                - question: The question to answer
                - response_method: Optional response method (json_mode, text, function_calling)
                - response_schema: Optional response schema for structured output

        Returns:
            State dictionary with the response added
        """
        with Metrics(
            "VLMRetrieval/retrieval", "red", span_kind=Metrics.SPAN_KIND["AGENT"]
        ) as tm:
            try:
                logger.debug(
                    "Running VLM retrieval with question: %s", state["question"]
                )
                tm.input(state)

                # Step 1: Retrieve relevant documents using semantic similarity
                documents = self._get_retrieved_documents(state["question"])
                logger.debug(f"Retrieved {len(documents)} documents")

                # Step 2: Extract images from the retrieved documents
                images_with_metadata = self._extract_images_from_documents(documents)
                logger.debug(
                    f"Extracted {len(images_with_metadata)} images with metadata"
                )

                # Step 3: Get VLM response about the images
                if images_with_metadata:
                    response = await self._get_vlm_response(
                        state["question"], images_with_metadata
                    )
                else:
                    # If no images found, try to get a response based on the retrieved documents
                    response = await self.get_vlm_response(
                        state["question"],
                        state.get("response_method"),
                        state.get("response_schema"),
                    )

                state["response"] = response
                logger.debug("VLM retrieval response: %s", response)

            except Exception as e:
                logger.error(traceback.format_exc())
                logger.error("Error in VLMRetrievalFunc %s", str(e))
                state["error"] = str(e)
                tm.error(e)
            finally:
                tm.output({"state": state})

        return state

    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        """Process a document (not used in VLM-only retrieval)."""
        pass

    async def areset(self, state: dict):
        """Reset the VLM retrieval function."""
        await asyncio.sleep(0.001)
