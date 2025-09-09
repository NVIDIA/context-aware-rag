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

"""utils.py: File contains utility functions"""

import re


from vss_ctx_rag.utils.ctx_rag_logger import logger, Metrics
import asyncio
from langchain_text_splitters import RecursiveCharacterTextSplitter


def remove_think_tags(text_in):
    text_out = re.sub(r"<think>.*?</think>", "", text_in, flags=re.DOTALL)
    return text_out


def remove_lucene_chars(text: str) -> str:
    """
    Remove Lucene special characters from the given text.

    This function takes a string as input and removes any special characters
    that are used in Lucene query syntax. The characters removed are:
    +, -, &, |, !, (, ), {, }, [, ], ^, ", ~, *, ?, :, \\ and /.

    Args:
        text (str): The input string from which to remove Lucene special characters.

    Returns:
        str: The cleaned string with Lucene special characters replaced by spaces.
    """
    special_chars = [
        "+",
        "-",
        "&",
        "|",
        "!",
        "(",
        ")",
        "{",
        "}",
        "[",
        "]",
        "^",
        '"',
        "~",
        "*",
        "?",
        ":",
        "\\",
        "/",
    ]
    for char in special_chars:
        if char in text:
            text = text.replace(char, " ")
    return text.strip()


def add_timestamps_to_doc(doc: str, doc_meta: dict) -> str:
    """Add timestamps to document based on metadata"""
    if not doc_meta["is_last"] and "file" in doc_meta:
        if doc_meta["file"].startswith("rtsp://"):
            # if live stream summarization
            if "start_ntp" in doc_meta and "end_ntp" in doc_meta:
                doc = f"<{doc_meta['start_ntp']}> <{doc_meta['end_ntp']}> " + doc
            else:
                logger.info(
                    "start_ntp or end_ntp not found in doc_meta. "
                    "No timestamp will be added."
                )
        else:
            # if file summmarization
            if "start_pts" in doc_meta and "end_pts" in doc_meta:
                doc = (
                    f"<{doc_meta['start_pts'] / 1e9:.2f}> <{doc_meta['end_pts'] / 1e9:.2f}> "
                    + doc
                )
            else:
                logger.info(
                    "start_pts or end_pts not found in doc_meta. "
                    "No timestamp will be added."
                )
    return doc


async def call_token_safe(input_data, pipeline, retries_left):
    """
    Unified function to handle token limit errors for both text and batch processing

    TODO: Currently no attributes/APIs for checking the token limit.
    This function is a temporary solution to handle the token limit error.
    """
    try:
        return await pipeline.ainvoke(input_data)
    except Exception as e:
        if (
            "exceeds maximum input length" not in str(e).lower()
            and "maximum context length" not in str(e).lower()
            and "please reduce the length of the input messages" not in str(e).lower()
            and "please reduce the length of the messages" not in str(e).lower()
            and "max_tokens must be at least 1, got" not in str(e).lower()
        ):
            raise e

        logger.warning(f"Received token exceeds limit error from pipeline : {e}")

        if retries_left <= 0:
            logger.debug("Maximum recursion depth exceeded. Returning input as is.")
            return input_data

        if isinstance(input_data, str):
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=max(1, len(input_data) // 2),
                chunk_overlap=50,
                length_function=len,
                is_separator_regex=False,
            )

            chunks = text_splitter.split_text(input_data)
            first_half, second_half = chunks[0], chunks[1]

            logger.info(
                f"Text exceeds token length. Splitting into "
                f"two parts of lengths {len(first_half)} and {len(second_half)}."
            )

            tasks = [
                call_token_safe(first_half, pipeline, retries_left - 1),
                call_token_safe(second_half, pipeline, retries_left - 1),
            ]
            summaries = await asyncio.gather(*tasks)
            combined_summary = "\n".join(summaries)

            try:
                return await pipeline.ainvoke(combined_summary)
            except Exception:
                logger.debug(
                    "Error after combining summaries, returning combined summary."
                )
                return combined_summary
        elif isinstance(input_data, list):
            if len(input_data) == 1:
                with Metrics("OffBatSumm/BaseCase", "yellow"):
                    logger.debug("Base Case, batch size = 1")
                    text = input_data[0]
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=max(1, len(text) // 2),
                        chunk_overlap=50,
                        length_function=len,
                        is_separator_regex=False,
                    )

                    chunks = text_splitter.split_text(text)
                    first_half, second_half = chunks[0], chunks[1]

                    logger.debug(
                        f"Text exceeds token length. Splitting into "
                        f"two parts of lengths {len(first_half)} and {len(second_half)}."
                    )

                    tasks = [
                        call_token_safe([first_half], pipeline, retries_left - 1),
                        call_token_safe([second_half], pipeline, retries_left - 1),
                    ]
                    summaries = await asyncio.gather(*tasks)
                    combined_summary = "\n".join(summaries)

                    try:
                        aggregated = await pipeline.ainvoke([combined_summary])
                        return aggregated
                    except Exception:
                        logger.debug(
                            "Error after combining summaries, retrying with combined summary."
                        )
                        return await call_token_safe(
                            [combined_summary], pipeline, retries_left - 1
                        )
            else:
                midpoint = len(input_data) // 2
                first_batch = input_data[:midpoint]
                second_batch = input_data[midpoint:]

                logger.debug(
                    f"Batch size {len(input_data)} exceeds token length. "
                    f"Splitting into two batches of sizes {len(first_batch)} and {len(second_batch)}."
                )

                tasks = [
                    call_token_safe(first_batch, pipeline, retries_left - 1),
                    call_token_safe(second_batch, pipeline, retries_left - 1),
                ]
                results = await asyncio.gather(*tasks)

                combined_results = []
                for result in results:
                    if isinstance(result, list):
                        combined_results.extend(result)
                    else:
                        combined_results.append(result)

                try:
                    with Metrics("OffBatSumm/CombindAgg", "red"):
                        aggregated = await pipeline.ainvoke(combined_results)
                        return aggregated
                except Exception:
                    logger.debug(
                        "Error after combining batch summaries, retrying with combined summaries."
                    )
                    return await call_token_safe(
                        combined_results, pipeline, retries_left - 1
                    )
        else:
            raise ValueError(f"Unsupported input_data type: {type(input_data)}")


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def extract_external_rag_query(text: str) -> tuple[str, str]:
    """Extract external RAG query from text marked with <e> tags.
    Returns (text_without_tags, external_rag_query)"""
    import re

    pattern = r"<e>(.*?)<e>"
    # Remove the tagged content and get clean text
    clean_text = re.sub(pattern, "", text, flags=re.DOTALL).strip()
    # Extract the content between tags
    matches = re.findall(pattern, text, re.DOTALL)
    external_rag_query = matches[0] if matches else ""

    return clean_text, external_rag_query
