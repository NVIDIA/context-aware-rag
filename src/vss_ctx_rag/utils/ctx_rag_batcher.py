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

from typing import Optional
from collections import defaultdict
import threading
from vss_ctx_rag.utils.ctx_rag_logger import logger, TimeMeasure


class Batch:
    """
    Initializes a new instance of the Batch class.
    Parameters:
        batch_size (int): The maximum number of documents that can be stored in the batch.
    Returns:
        None
    """

    def __init__(self, batch_size):
        self._batch_size = batch_size
        self._batch_index = None
        self._batch = {}
        self._is_full = False
        self._is_last = False

    def add_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        """
        Adds a document to the batch.
        Args:
            doc (str): The content of the document.
            doc_i (int): The index of the document.
            doc_meta (Optional[dict], optional): Additional metadata for the document.
                                                Defaults to None.
        Raises:
            RuntimeError: If the document index already exists in the batch or if the batch is full.
        Returns:
            int: The index of the batch.
        """
        if self._batch_index is None:
            self._batch_index = doc_i // self._batch_size
        else:
            if doc_i // self._batch_size != self._batch_index:
                raise RuntimeError(
                    f"Document index {doc_i} is being added to incorrect batch: {self._batch_index}."
                )

        if doc_i in self._batch:
            raise RuntimeError(f"Duplicate doc_i: {doc_i}")

        if self.is_full():
            raise RuntimeError(f"Batch is already full: {doc_i} insertion failed.")

        self._batch[doc_i] = (doc, doc_i, doc_meta)

        if doc_meta and doc_meta.get("is_last", False):
            self._is_last = True

        if len(self._batch) == self._batch_size:
            self._is_full = True

        return self._batch_index

    def is_full(self):
        if self._is_last:
            # Check explicitly for last batch
            batch = self.as_list()
            return len(batch) == batch[-1][1] - batch[0][1] + 1
        else:
            return self._is_full

    def has(self, doc_i):
        return doc_i in self._batch

    def flush(self):
        """
        Flushes the batch and resets the state.
        This function clears the `self.batch` list and sets the `self._is_full` flag to `False`.
        Parameters:
            None
        Returns:
            None
        """
        if self._batch:
            self._batch = {}
            self._is_full = False

    def as_list(self, sort=True):
        if sort:
            # Will this remain sorted a dict is created?
            return [value for _, value in sorted(self._batch.items())]
        return list(self._batch.values())

    def __str__(self):
        return str(self.as_list(sort=False))


class Batcher:
    def __init__(self, batch_size):
        logger.info("Setting up Batcher with batch size %d", batch_size)
        self.batch_size = batch_size
        self.batches = defaultdict(lambda: Batch(self.batch_size))
        self._lock = threading.Lock()

    def add_doc(self, doc: str, doc_i: int, doc_meta: Optional[dict] = None):
        """
        Adds a document to the batch.
        Args:
            doc (str): The content of the document.
            doc_i (int): The index of the document.
            doc_meta (Optional[dict], optional): Additional metadata for the document.
                                                Defaults to None.
        Returns:
            Batch: The batch that the document was added to
        """
        with TimeMeasure("Add Doc", "green"):
            logger.info(f"adding {doc_i} to batch")
            with self._lock:
                batch = self.batches[doc_i // self.batch_size]
                batch.add_doc(doc, doc_i, doc_meta)
                return batch

    def get_batch(self, doc_i: Optional[int] = None, batch_i: Optional[int] = None):
        """
        Retrieves a batch of documents based on the provided document or batch index.
        Args:
            doc_i (Optional[int], optional): The index of the document. Defaults to None.
            batch_i (Optional[int], optional): The index of the batch. Defaults to None.
        Returns:
            Batch or None: The batch of documents if the document index is present in the batch,
                           otherwise None.
        """
        with self._lock:
            if batch_i is None:
                batch_i = doc_i // self.batch_size
            batch = self.batches[batch_i]
            if doc_i is not None:
                if batch.has(doc_i):
                    return batch
                return None
            else:
                return batch

    def get_all_batches(self):
        with self._lock:
            batches = []
            for i, batch in self.batches.items():
                batches.extend(batch.as_list())
            return batches

    def get_batch_index(self, doc_i):
        return doc_i // self.batch_size

    def __str__(self):
        with self._lock:
            print_val = ""
            for i, batch in self.batches.items():
                print_val += "batch " + str(i) + ": " + str(batch.as_list()) + "\n"
            return print_val

    def flush(self):
        with self._lock:
            for batch in self.batches.values():
                batch.flush()
