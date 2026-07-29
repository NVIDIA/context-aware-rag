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

import pytest

from vss_ctx_rag.utils.ctx_rag_batcher import Batch, Batcher


class TestBatch:
    """Test cases for Batch class."""

    def test_batch_initialization(self):
        """Test Batch initialization."""
        batch = Batch(batch_size=5)

        assert batch.get_batch_size() == 5
        assert batch.get_batch_index() is None  # Initially None
        assert not batch.is_full()

    def test_add_doc_basic(self):
        """Test adding documents to batch."""
        batch = Batch(batch_size=2)

        batch.add_doc("doc1", 0, {"meta": "data1"})
        batch.add_doc("doc2", 1, {"meta": "data2"})

        assert batch.is_full()
        docs = batch.as_list()
        assert len(docs) == 2
        # Fix: docs are tuples (doc, doc_i, doc_meta)
        assert docs[0][0] == "doc1"  # doc content
        assert docs[1][0] == "doc2"  # doc content

    def test_batch_has_doc(self):
        """Test checking if batch has specific document."""
        batch = Batch(batch_size=3)
        batch.add_doc("doc1", 5)

        assert batch.has(5)
        assert not batch.has(6)

    def test_batch_flush(self):
        """Test batch flush functionality."""
        batch = Batch(batch_size=2)
        batch.add_doc("doc1", 0)
        batch.add_doc("doc2", 1)

        # Get docs before flush
        docs_before = batch.as_list()
        assert len(docs_before) == 2

        # Flush and check
        batch.flush()
        docs_after = batch.as_list()
        assert len(docs_after) == 0
        assert not batch.is_full()

    def test_batch_as_list_sorted(self):
        """Test as_list with sorting."""
        batch = Batch(batch_size=3)
        batch.add_doc("doc1", 1)  # Start with doc_i=1 (batch index 0)
        batch.add_doc("doc2", 2)  # Same batch

        # Test sorted (default)
        sorted_docs = batch.as_list(sort=True)
        assert [doc[1] for doc in sorted_docs] == [1, 2]  # doc_i values

        # Test unsorted
        unsorted_docs = batch.as_list(sort=False)
        assert len(unsorted_docs) == 2

    def test_batch_str_representation(self):
        """Test string representation of batch."""
        batch = Batch(batch_size=2)
        batch.add_doc("doc1", 0, {"key": "value"})
        batch.add_doc("doc2", 1)

        str_repr = str(batch)
        assert str_repr is not None
        assert len(str_repr) > 0
        # Verify actual content is in string representation
        assert "doc1" in str_repr
        assert "doc2" in str_repr
        assert "0" in str_repr  # doc_i should be present
        assert "1" in str_repr
        assert "key" in str_repr  # metadata should be present
        assert "value" in str_repr

    def test_batch_is_full_with_last_batch(self):
        """Test is_full method with _is_last flag."""
        batch = Batch(batch_size=3)
        batch.add_doc(
            "doc1", 3, {"is_last": True}
        )  # Same batch (index 1), marked as last

        # Should check if all documents in range are present
        result = batch.is_full()
        assert result  # Last batch logic

    def test_batch_flush_when_empty(self):
        """Test flush on empty batch."""
        batch = Batch(batch_size=2)

        # Flush empty batch should work without error
        batch.flush()
        assert not batch.is_full()
        assert len(batch.as_list()) == 0

    def test_batch_as_list_unsorted(self):
        """Test as_list with sort=False."""
        batch = Batch(batch_size=3)
        batch.add_doc("doc1", 0)  # All same batch (index 0)
        batch.add_doc("doc2", 1)
        batch.add_doc("doc3", 2)

        # Test unsorted - should maintain insertion order in values
        unsorted = batch.as_list(sort=False)
        assert len(unsorted) == 3

        # Test sorted - should be ordered by doc_i
        sorted_docs = batch.as_list(sort=True)
        assert [doc[1] for doc in sorted_docs] == [0, 1, 2]  # doc_i values

    def test_batch_duplicate_doc_error(self):
        """Test error when adding duplicate document index."""
        batch = Batch(batch_size=3)
        batch.add_doc("doc1", 1)

        # Adding same doc_i should raise RuntimeError
        with pytest.raises(RuntimeError, match="Duplicate doc_i"):
            batch.add_doc("doc1_duplicate", 1)

    def test_batch_overfull_error(self):
        """Test error when adding to full batch."""
        from unittest.mock import patch

        batch = Batch(batch_size=2)
        batch.add_doc("doc1", 0)  # Batch index 0

        # Mock is_full() to return True to simulate a full batch
        # This allows us to test the overfull condition without filling the batch
        with patch.object(batch, "is_full", return_value=True):
            with pytest.raises(RuntimeError, match="already full"):
                batch.add_doc("doc2", 1)  # Try to add to same batch when it's "full"


class TestBatcher:
    """Test cases for Batcher class."""

    def test_batcher_initialization(self):
        """Test Batcher initialization."""
        manager = Batcher(batch_size=3)
        assert manager.get_batch_index(0) == 0

    def test_add_doc_to_batcher(self):
        """Test adding documents to batcher."""
        manager = Batcher(batch_size=2)

        manager.add_doc("doc1", 0, {"meta": "data1"})
        manager.add_doc("doc2", 1, {"meta": "data2"})
        manager.add_doc("doc3", 2, {"meta": "data3"})  # Should create new batch

        # Should have documents in batches
        all_batches = manager.get_all_batches()
        assert len(all_batches) == 3  # 3 documents total

    def test_get_batch_by_doc_index(self):
        """Test retrieving batch by document index."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0)
        manager.add_doc("doc2", 1)
        manager.add_doc("doc3", 2)

        # Get batch containing doc_i=1
        batch = manager.get_batch(doc_i=1)
        assert batch is not None
        assert batch.has(1)

    def test_get_batch_by_batch_index(self):
        """Test retrieving batch by batch index."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0)
        manager.add_doc("doc2", 1)
        manager.add_doc("doc3", 2)

        # Get second batch (batch_i=1)
        batch = manager.get_batch(batch_i=1)
        assert batch is not None
        assert batch.has(2)

    def test_get_all_full_batches(self):
        """Test getting only full batches."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0)
        manager.add_doc("doc2", 1)  # First batch full
        manager.add_doc("doc3", 2)  # Second batch partial

        full_batches = manager.get_all_full_batches()
        assert len(full_batches) == 1  # Only first batch is full

        all_batches = manager.get_all_batches()
        assert len(all_batches) == 3  # 3 documents total (not batch objects)

    def test_batcher_flush(self):
        """Test flushing all batches in batcher."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0)
        manager.add_doc("doc2", 1)

        # Flush all
        manager.flush()

        # After flush, batches should be empty
        all_batches = manager.get_all_batches()
        assert (
            len(all_batches) == 0
        )  # get_all_batches returns documents, not batch objects

    def test_batcher_str_representation(self):
        """Test string representation of batcher."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0, {"meta": "data1"})
        manager.add_doc("doc2", 1)
        manager.add_doc("doc3", 2)  # New batch

        str_repr = str(manager)
        assert str_repr is not None
        assert len(str_repr) > 0
        # Verify batch structure is shown
        assert "batch 0" in str_repr  # First batch index
        assert "batch 1" in str_repr  # Second batch index
        assert "doc1" in str_repr
        assert "doc2" in str_repr
        assert "doc3" in str_repr
        # Verify metadata is included
        assert "meta" in str_repr
        assert "data1" in str_repr

    def test_batcher_thread_safety(self):
        """Test batcher thread safety with concurrent access."""
        import threading
        import time

        manager = Batcher(batch_size=10)
        errors = []
        success_count = [0]  # Use list to allow modification in nested function

        def add_docs_worker(start, end):
            """Add documents in a range with some randomization."""
            try:
                for i in range(start, end):
                    manager.add_doc(
                        f"doc{i}", i, {"thread": threading.current_thread().name}
                    )
                    success_count[0] += 1
                    # Small random delay to increase chance of race conditions
                    time.sleep(0.0001)
            except Exception as e:
                errors.append((threading.current_thread().name, str(e)))

        # Create multiple threads adding documents concurrently
        threads = []
        num_threads = 5
        docs_per_thread = 20

        for i in range(num_threads):
            t = threading.Thread(
                target=add_docs_worker,
                args=(i * docs_per_thread, (i + 1) * docs_per_thread),
                name=f"Thread-{i}",
            )
            threads.append(t)
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Verify results
        assert len(errors) == 0, f"Thread errors occurred: {errors}"
        assert success_count[0] == num_threads * docs_per_thread

        # Verify all documents were added correctly
        all_docs = manager.get_all_batches()
        assert len(all_docs) == num_threads * docs_per_thread

        # Verify no duplicate doc_i values (would indicate race condition)
        doc_indices = [doc[1] for doc in all_docs]
        assert len(set(doc_indices)) == len(doc_indices), "Duplicate doc indices found"

        # Verify batches are properly formed
        full_batches = manager.get_all_full_batches()
        expected_full_batches = (num_threads * docs_per_thread) // 10
        assert len(full_batches) == expected_full_batches

    def test_batcher_get_batch_nonexistent_doc(self):
        """Test getting batch for non-existent document."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 0)

        # Try to get batch for doc that doesn't exist
        result = manager.get_batch(doc_i=5)
        assert result is None

    def test_batcher_get_batch_by_batch_index_only(self):
        """Test getting batch by batch index without doc_i."""
        manager = Batcher(batch_size=2)
        manager.add_doc("doc1", 4)  # Will be in batch index 2

        # Get batch by index only
        batch = manager.get_batch(batch_i=2)
        assert batch is not None
        assert batch.has(4)

    def test_batcher_multiple_batches(self):
        """Test batcher with multiple batches."""
        manager = Batcher(batch_size=2)

        # Add docs to different batches
        manager.add_doc("doc0", 0)  # Batch 0
        manager.add_doc("doc1", 1)  # Batch 0
        manager.add_doc("doc2", 2)  # Batch 1
        manager.add_doc("doc3", 3)  # Batch 1
        manager.add_doc("doc4", 4)  # Batch 2

        # Should have 5 total documents across batches
        all_docs = manager.get_all_batches()
        assert len(all_docs) == 5

        # Should have 2 full batches
        full_batches = manager.get_all_full_batches()
        assert len(full_batches) == 2

    def test_batch_index_calculation(self):
        """Test batch index calculation."""
        manager = Batcher(batch_size=3)

        assert manager.get_batch_index(0) == 0
        assert manager.get_batch_index(1) == 0
        assert manager.get_batch_index(2) == 0
        assert manager.get_batch_index(3) == 1
        assert manager.get_batch_index(6) == 2

    def test_batch_boundary_negative_index(self):
        """Test handling of negative document indices."""
        batch = Batch(batch_size=3)

        # Should work with negative indices (Python's // operator handles this)
        batch.add_doc("doc_neg", -1)
        assert batch.has(-1)
        assert batch.get_batch_index() == -1  # -1 // 3 = -1

    def test_batch_boundary_large_indices(self):
        """Test handling of very large document indices."""
        batch = Batch(batch_size=100)
        large_index = 999999

        batch.add_doc("doc_large", large_index)
        assert batch.has(large_index)
        assert batch.get_batch_index() == large_index // 100

    def test_batch_wrong_batch_assignment(self):
        """Test error when adding document to wrong batch."""
        batch = Batch(batch_size=3)

        # First doc establishes batch index
        batch.add_doc("doc0", 0)  # Batch index 0

        # Try to add doc from different batch
        with pytest.raises(RuntimeError, match="incorrect batch"):
            batch.add_doc("doc5", 5)  # Would be batch index 1

    def test_batcher_boundary_conditions(self):
        """Test batcher with various boundary conditions."""
        manager = Batcher(batch_size=2)

        # Test with zero index
        batch = manager.add_doc("doc0", 0)
        assert batch is not None

        # Test batch boundaries
        manager.add_doc("doc1", 1)  # Still batch 0
        manager.add_doc("doc2", 2)  # New batch 1

        # Verify batch assignments
        assert manager.get_batch(doc_i=0).has(0)
        assert manager.get_batch(doc_i=0).has(1)
        assert manager.get_batch(doc_i=2).has(2)
        assert not manager.get_batch(doc_i=0).has(2)

    def test_batch_metadata_handling(self):
        """Test comprehensive metadata handling in batches."""
        batch = Batch(batch_size=3)

        # Add docs with various metadata
        batch.add_doc("doc1", 0, {"type": "text", "size": 100})
        batch.add_doc("doc2", 1, {"type": "image", "size": 500, "format": "png"})
        batch.add_doc("doc3", 2)  # No metadata

        docs = batch.as_list()

        # Verify metadata is preserved
        assert docs[0][2] == {"type": "text", "size": 100}
        assert docs[1][2] == {"type": "image", "size": 500, "format": "png"}
        assert docs[2][2] is None

    def test_batch_is_last_metadata(self):
        """Test is_last metadata flag behavior."""
        batch = Batch(batch_size=5)

        # Add documents with last one marked as is_last
        batch.add_doc("doc1", 10)
        batch.add_doc("doc2", 11)
        batch.add_doc("doc3", 12, {"is_last": True})

        # Should be full since we have consecutive docs ending with is_last
        assert batch.is_full()

        # Test with gap
        batch2 = Batch(batch_size=5)
        batch2.add_doc("doc1", 20)
        batch2.add_doc("doc3", 22, {"is_last": True})  # Gap at index 21

        # Should not be full due to gap
        assert not batch2.is_full()

    def test_batcher_metadata_preservation(self):
        """Test that metadata is preserved through batcher operations."""
        manager = Batcher(batch_size=2)

        # Add documents with metadata
        meta1 = {"source": "file1.txt", "timestamp": 123456}
        meta2 = {"source": "file2.txt", "timestamp": 123457, "encoding": "utf-8"}

        manager.add_doc("content1", 0, meta1)
        manager.add_doc("content2", 1, meta2)
        manager.add_doc("content3", 2)  # No metadata

        # Get all batches and verify metadata
        all_docs = manager.get_all_batches()

        # Find docs by index and check metadata
        doc_map = {doc[1]: doc for doc in all_docs}

        assert doc_map[0][2] == meta1
        assert doc_map[1][2] == meta2
        assert doc_map[2][2] is None

    def test_batcher_out_of_order_additions(self):
        """Test adding documents out of order to batcher."""
        manager = Batcher(batch_size=3)

        # Add documents in non-sequential order
        manager.add_doc("doc5", 5)  # Batch 1
        manager.add_doc("doc0", 0)  # Batch 0
        manager.add_doc("doc8", 8)  # Batch 2
        manager.add_doc("doc2", 2)  # Batch 0
        manager.add_doc("doc3", 3)  # Batch 1
        manager.add_doc("doc1", 1)  # Batch 0

        # Verify all documents are in correct batches
        batch0 = manager.get_batch(batch_i=0)
        batch1 = manager.get_batch(batch_i=1)
        batch2 = manager.get_batch(batch_i=2)

        assert batch0.has(0) and batch0.has(1) and batch0.has(2)
        assert batch1.has(3) and batch1.has(5)
        assert batch2.has(8)

        # Verify batch 0 is full, others are not
        assert batch0.is_full()
        assert not batch1.is_full()  # Missing doc 4
        assert not batch2.is_full()  # Missing docs 6, 7

    def test_batch_with_gaps(self):
        """Test batch behavior with gaps in document indices."""
        batch = Batch(batch_size=4)

        # Add docs with gaps
        batch.add_doc("doc0", 0)
        batch.add_doc("doc2", 2)  # Gap at index 1
        batch.add_doc("doc3", 3)

        # Batch should not be full due to gap
        assert not batch.is_full()

        # Verify sorted order is maintained
        docs_sorted = batch.as_list(sort=True)
        assert [d[1] for d in docs_sorted] == [0, 2, 3]

    def test_complex_batch_scenario(self):
        """Test complex scenario with multiple batches, gaps, and metadata."""
        manager = Batcher(batch_size=2)

        # Create a complex scenario
        docs_to_add = [
            (0, "doc0", {"priority": "high"}),
            (1, "doc1", {"priority": "low"}),
            (3, "doc3", None),  # Gap at index 2
            (4, "doc4", {"priority": "medium"}),
            (7, "doc7", {"is_last": True}),  # Gaps at 5, 6
        ]

        for doc_i, content, meta in docs_to_add:
            manager.add_doc(content, doc_i, meta)

        # Check batch distribution
        batch0 = manager.get_batch(batch_i=0)  # Should have 0, 1
        batch1 = manager.get_batch(batch_i=1)  # Should have 3
        batch2 = manager.get_batch(batch_i=2)  # Should have 4
        batch3 = manager.get_batch(batch_i=3)  # Should have 7

        assert batch0.is_full()  # Has both 0 and 1
        assert not batch1.is_full()  # Missing index 2
        assert not batch2.is_full()  # Only has 4, missing 5
        assert batch3.is_full()  # Has 7 with is_last=True

        # Verify metadata preservation
        all_docs = manager.get_all_batches()
        doc_map = {doc[1]: doc for doc in all_docs}

        assert doc_map[0][2]["priority"] == "high"
        assert doc_map[7][2]["is_last"] is True

    def test_batch_as_list_order_consistency(self):
        """Test that as_list maintains consistent ordering."""
        batch = Batch(batch_size=5)

        # Add in random order
        batch.add_doc("doc3", 23)
        batch.add_doc("doc1", 21)
        batch.add_doc("doc4", 24)
        batch.add_doc("doc0", 20)
        batch.add_doc("doc2", 22)

        # Test sorted order multiple times
        for _ in range(3):
            sorted_docs = batch.as_list(sort=True)
            assert [d[1] for d in sorted_docs] == [20, 21, 22, 23, 24]

        # Test unsorted maintains insertion order (dict order in Python 3.7+)
        unsorted_docs = batch.as_list(sort=False)
        # Just verify we get all docs
        assert len(unsorted_docs) == 5
