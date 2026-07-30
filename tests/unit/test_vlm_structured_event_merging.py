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

"""Unit tests for Event class and event merging logic in VlmStructuredSummarization.

This module tests:
- Event.overlaps_with() method for time overlap and adjacency detection
- Event.merge_with() method for combining events
- VlmStructuredSummarization._merge_similar_events() for batch event merging
"""

import pytest

from vss_ctx_rag.functions.summarization.vlm_structured import (
    Event,
    VlmStructuredSummarization,
)


# =============================================================================
# Event Class Tests
# =============================================================================


class TestEventTimestampParsing:
    """Tests for video-relative timestamp conversion to seconds."""

    @pytest.mark.parametrize(
        "timestamp,expected_seconds",
        [
            ("00:00", 0.0),
            ("01:30", 90.0),
            ("65:00", 3900.0),
            ("00:01:30", 90.0),
            ("02:03:04", 7384.0),
            ("00:00:01.5", 1.5),
            # Elapsed time, not wall clock: hours are not capped at 24.
            ("24:00:00", 86400.0),
            ("99:59:59", 359999.0),
        ],
    )
    def test_converts_mm_ss_and_hh_mm_ss_to_seconds(self, timestamp, expected_seconds):
        event = Event(
            start_time=timestamp,
            end_time=expected_seconds + 1,
            type="test",
            description="Timestamp conversion",
        )

        assert event.start_time == expected_seconds

    @pytest.mark.parametrize(
        "timestamp",
        [
            "00:60",
            "01:60:00",
            "1:02",
            "1:02:03",
            "-01:30",
            "-01:00:00",
            "01:-30",
            "100:00:00",
        ],
    )
    def test_rejects_invalid_or_incomplete_video_timestamps(self, timestamp):
        with pytest.raises(ValueError, match="Cannot parse timestamp"):
            Event(
                start_time=timestamp,
                end_time=10,
                type="test",
                description="Invalid timestamp",
            )

    @pytest.mark.parametrize("timestamp", ["00:60", "01:60:00"])
    def test_out_of_range_components_report_the_range_violation(self, timestamp):
        """Out-of-range fields fail loudly instead of falling back to ISO parsing."""
        with pytest.raises(ValueError, match="must be < 60"):
            Event(
                start_time=timestamp,
                end_time=10,
                type="test",
                description="Out of range timestamp",
            )

    @pytest.mark.parametrize("timestamp", [None, [], {}])
    def test_rejects_non_numeric_non_string_timestamps(self, timestamp):
        with pytest.raises(TypeError, match="Expected numeric, MM:SS, HH:MM:SS"):
            Event(
                start_time=timestamp,
                end_time=10,
                type="test",
                description="Unsupported timestamp type",
            )

    def test_overlaps_when_mm_ss_ranges_overlap(self):
        """Overlapping MM:SS events of the same type should be detected."""
        event1 = Event(
            start_time="01:00",
            end_time="01:10",
            type="motion",
            description="Motion A",
        )
        event2 = Event(
            start_time="01:05",
            end_time="01:20",
            type="motion",
            description="Motion B",
        )

        # Overlap is 5 seconds (65-70), above default 0.1 threshold
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)
        assert event2.overlaps_with(event1, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_when_hh_mm_ss_ranges_overlap(self):
        """Overlapping HH:MM:SS events of the same type should be detected."""
        event1 = Event(
            start_time="00:01:00",
            end_time="00:01:10",
            type="fire",
            description="Fire A",
        )
        event2 = Event(
            start_time="00:01:05",
            end_time="00:01:20",
            type="fire",
            description="Fire B",
        )

        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_when_mm_ss_and_hh_mm_ss_are_mixed(self):
        """Mixed MM:SS and HH:MM:SS formats should compare after conversion."""
        event1 = Event(
            start_time="00:01:00",
            end_time="00:01:10",
            type="alert",
            description="Alert A",
        )
        event2 = Event(
            start_time="01:05",
            end_time="01:20",
            type="alert",
            description="Alert B",
        )

        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_adjacent_mm_ss_within_threshold(self):
        """Adjacent MM:SS events within the gap threshold should match."""
        event1 = Event(
            start_time="02:00",
            end_time="02:10",
            type="person",
            description="Person enters",
        )
        event2 = Event(
            start_time="02:12",
            end_time="02:20",
            type="person",
            description="Person exits",
        )

        # Gap is 2 seconds (132 - 130), threshold is 4
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_adjacent_hh_mm_ss_beyond_threshold(self):
        """Adjacent HH:MM:SS events beyond the gap threshold should not match."""
        event1 = Event(
            start_time="00:02:00",
            end_time="00:02:10",
            type="person",
            description="Person enters",
        )
        event2 = Event(
            start_time="00:02:20",
            end_time="00:02:30",
            type="person",
            description="Person exits",
        )

        # Gap is 10 seconds, threshold is 4
        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )

    def test_touching_video_timestamp_boundaries(self):
        """Events that touch exactly at MM:SS / HH:MM:SS boundaries are adjacent."""
        event1 = Event(
            start_time="00:00:50",
            end_time="00:01:00",
            type="traffic",
            description="Jam starts",
        )
        event2 = Event(
            start_time="01:00",
            end_time="01:10",
            type="traffic",
            description="Jam continues",
        )

        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_different_types_with_video_timestamps_do_not_overlap(self):
        """Different event types never overlap even with identical video times."""
        event1 = Event(
            start_time="01:00",
            end_time="01:10",
            type="fire",
            description="Fire",
        )
        event2 = Event(
            start_time="00:01:00",
            end_time="00:01:10",
            type="theft",
            description="Theft",
        )

        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )


class TestEventOverlapsWith:
    """Tests for Event.overlaps_with() method."""

    def test_overlaps_with_same_type_overlapping_events(self):
        """Test that overlapping events of the same type are detected as overlapping."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected in area A",
        )
        event2 = Event(
            start_time=5.0,
            end_time=15.0,
            type="fire",
            description="Fire spreading to area B",
        )

        # Overlap is 5 seconds (5.0 to 10.0), which is > 0.1 threshold
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)
        assert event2.overlaps_with(event1, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_with_different_types_no_overlap(self):
        """Test that events of different types never overlap (even with time overlap)."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected",
        )
        event2 = Event(
            start_time=5.0,
            end_time=15.0,
            type="theft",
            description="Theft incident",
        )

        # Same time range but different types - should NOT overlap
        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )
        assert not event2.overlaps_with(
            event1, overlap_threshold=0.1, adjacent_threshold=4
        )

    def test_overlaps_with_adjacent_events_within_threshold(self):
        """Test that adjacent events within threshold are detected as overlapping."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="accident",
            description="Initial collision",
        )
        event2 = Event(
            start_time=12.0,
            end_time=20.0,
            type="accident",
            description="Secondary impact",
        )

        # Gap is 2 seconds (12.0 - 10.0), threshold is 4 - should be adjacent
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)
        assert event2.overlaps_with(event1, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_with_adjacent_events_beyond_threshold(self):
        """Test that events beyond adjacency threshold are not overlapping."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected",
        )
        event2 = Event(
            start_time=20.0,
            end_time=30.0,
            type="fire",
            description="New fire incident",
        )

        # Gap is 10 seconds (20.0 - 10.0), threshold is 4 - should NOT be adjacent
        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )
        assert not event2.overlaps_with(
            event1, overlap_threshold=0.1, adjacent_threshold=4
        )

    def test_overlaps_with_touching_events(self):
        """Test that events touching exactly at boundaries are adjacent."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="traffic",
            description="Traffic jam starts",
        )
        event2 = Event(
            start_time=10.0,
            end_time=20.0,
            type="traffic",
            description="Traffic jam continues",
        )

        # Events touch at 10.0 - gap is 0, should be adjacent
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_with_small_overlap_below_threshold(self):
        """Test that very small overlaps below threshold are NOT detected as overlapping.

        When events have a tiny overlap that's below the overlap_threshold,
        they are not considered overlapping. Since they DO have some overlap,
        the adjacency check is not applied.
        """
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="motion",
            description="Motion detected",
        )
        event2 = Event(
            start_time=9.95,
            end_time=20.0,
            type="motion",
            description="Motion continues",
        )

        # Overlap is 0.05 seconds, below 0.1 threshold
        # Since there IS an overlap (even if small), adjacency check is not applied
        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )

    def test_overlaps_with_small_overlap_above_threshold(self):
        """Test that small overlaps meeting the threshold ARE detected."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="motion",
            description="Motion detected",
        )
        event2 = Event(
            start_time=9.85,
            end_time=20.0,
            type="motion",
            description="Motion continues",
        )

        # Overlap is 0.15 seconds, above 0.1 threshold - should be detected
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_with_contained_event(self):
        """Test that fully contained events are detected as overlapping."""
        event1 = Event(
            start_time=0.0,
            end_time=30.0,
            type="surveillance",
            description="Long surveillance window",
        )
        event2 = Event(
            start_time=10.0,
            end_time=20.0,
            type="surveillance",
            description="Activity within window",
        )

        # event2 is fully contained within event1
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=4)
        assert event2.overlaps_with(event1, overlap_threshold=0.1, adjacent_threshold=4)

    def test_overlaps_with_custom_thresholds(self):
        """Test overlap detection with custom threshold values."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="alert",
            description="Alert 1",
        )
        event2 = Event(
            start_time=15.0,
            end_time=25.0,
            type="alert",
            description="Alert 2",
        )

        # Gap is 5 seconds
        # With adjacent_threshold=4, should NOT overlap
        assert not event1.overlaps_with(
            event2, overlap_threshold=0.1, adjacent_threshold=4
        )

        # With adjacent_threshold=6, SHOULD overlap (5 <= 6)
        assert event1.overlaps_with(event2, overlap_threshold=0.1, adjacent_threshold=6)

    def test_overlaps_with_order_independence(self):
        """Test that overlap detection is order-independent.

        The same pair of events should produce the same overlap result
        regardless of which event is 'self' and which is 'other'.
        """
        # Touching events [10,20] and [0,10]
        event_a = Event(start_time=10.0, end_time=20.0, type="test", description="A")
        event_b = Event(start_time=0.0, end_time=10.0, type="test", description="B")

        # Both orderings should detect them as adjacent (touching)
        assert event_a.overlaps_with(
            event_b, overlap_threshold=0.1, adjacent_threshold=4
        )
        assert event_b.overlaps_with(
            event_a, overlap_threshold=0.1, adjacent_threshold=4
        )

        # Adjacent events with gap [0,10] and [12,20] (gap of 2)
        event_c = Event(start_time=0.0, end_time=10.0, type="test", description="C")
        event_d = Event(start_time=12.0, end_time=20.0, type="test", description="D")

        # Both orderings should detect them as adjacent
        assert event_c.overlaps_with(
            event_d, overlap_threshold=0.1, adjacent_threshold=4
        )
        assert event_d.overlaps_with(
            event_c, overlap_threshold=0.1, adjacent_threshold=4
        )

        # Non-adjacent events [0,10] and [20,30] (gap of 10)
        event_e = Event(start_time=0.0, end_time=10.0, type="test", description="E")
        event_f = Event(start_time=20.0, end_time=30.0, type="test", description="F")

        # Both orderings should NOT detect them as adjacent
        assert not event_e.overlaps_with(
            event_f, overlap_threshold=0.1, adjacent_threshold=4
        )
        assert not event_f.overlaps_with(
            event_e, overlap_threshold=0.1, adjacent_threshold=4
        )


class TestEventMergeWith:
    """Tests for Event.merge_with() method."""

    def test_merge_with_basic(self):
        """Test basic merging of two events."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire started",
        )
        event2 = Event(
            start_time=8.0,
            end_time=20.0,
            type="fire",
            description="Fire spreading",
        )

        merged = event1.merge_with(event2)

        assert merged.start_time == 0.0
        assert merged.end_time == 20.0
        assert merged.type == "fire"
        assert "Fire started" in merged.description
        assert "Fire spreading" in merged.description
        assert " | " in merged.description

    def test_merge_with_preserves_type(self):
        """Test that merge preserves the original event type."""
        event1 = Event(
            start_time=5.0,
            end_time=15.0,
            type="accident",
            description="Collision occurred",
        )
        event2 = Event(
            start_time=10.0,
            end_time=25.0,
            type="accident",  # Same type
            description="Emergency response",
        )

        merged = event1.merge_with(event2)
        assert merged.type == "accident"

    def test_merge_with_non_overlapping_times(self):
        """Test merging adjacent (non-overlapping) events."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="event",
            description="First event",
        )
        event2 = Event(
            start_time=15.0,
            end_time=25.0,
            type="event",
            description="Second event",
        )

        merged = event1.merge_with(event2)

        # Should span the entire range
        assert merged.start_time == 0.0
        assert merged.end_time == 25.0

    def test_merge_with_contained_event(self):
        """Test merging when one event is fully contained in another."""
        outer_event = Event(
            start_time=0.0,
            end_time=30.0,
            type="observation",
            description="Long observation",
        )
        inner_event = Event(
            start_time=10.0,
            end_time=20.0,
            type="observation",
            description="Specific observation",
        )

        merged = outer_event.merge_with(inner_event)

        # Should keep the outer boundaries
        assert merged.start_time == 0.0
        assert merged.end_time == 30.0
        assert "Long observation" in merged.description
        assert "Specific observation" in merged.description

    def test_merge_with_reversed_order(self):
        """Test that merge order doesn't affect time bounds (only description order)."""
        event1 = Event(
            start_time=0.0,
            end_time=10.0,
            type="test",
            description="First",
        )
        event2 = Event(
            start_time=5.0,
            end_time=20.0,
            type="test",
            description="Second",
        )

        merged_1_2 = event1.merge_with(event2)
        merged_2_1 = event2.merge_with(event1)

        # Time bounds should be the same
        assert merged_1_2.start_time == merged_2_1.start_time == 0.0
        assert merged_1_2.end_time == merged_2_1.end_time == 20.0

        # Description order may differ
        assert "First" in merged_1_2.description
        assert "Second" in merged_1_2.description


# =============================================================================
# VlmStructuredSummarization._merge_similar_events Tests
# =============================================================================


def create_vlm_structured_instance(
    time_overlap_threshold: float = 0.1,
    time_adjacent_threshold: float = 4.0,
    enable_llm_merging: bool = False,
) -> VlmStructuredSummarization:
    """Helper function to create a VlmStructuredSummarization instance for testing."""
    # Create minimal mock config
    mock_config = {
        "params": {
            "uuid": "test-uuid",
            "time_overlap_threshold": time_overlap_threshold,
            "time_adjacent_threshold": time_adjacent_threshold,
            "max_events_per_batch": 50,
            "enable_llm_merging": enable_llm_merging,
        },
        "tools": {},
    }

    # Create instance with mocked dependencies
    instance = VlmStructuredSummarization.__new__(VlmStructuredSummarization)
    instance.config = mock_config
    instance.time_overlap_threshold = time_overlap_threshold
    instance.time_adjacent_threshold = time_adjacent_threshold
    instance.enable_llm_merging = enable_llm_merging
    instance.accumulated_events = []

    # Mock the LLM description merge method to simulate LLM behavior
    # It combines descriptions with " + " to make it easy to verify in tests
    async def mock_merge_descriptions(event_type: str, descriptions: list) -> str:
        if len(descriptions) == 1:
            return descriptions[0]
        # When LLM merging is disabled, the method won't be called
        # This mock simulates what the LLM would return
        return " + ".join(descriptions)

    instance._merge_descriptions_with_llm = mock_merge_descriptions

    return instance


class TestMergeSimilarEvents:
    """Tests for VlmStructuredSummarization._merge_similar_events() method."""

    @pytest.mark.asyncio
    async def test_merges_events_after_converting_video_timestamps_to_seconds(self):
        instance = create_vlm_structured_instance(time_adjacent_threshold=4.0)
        events = [
            Event(
                start_time="00:00:50",
                end_time="00:01:00",
                type="motion",
                description="Motion starts",
            ),
            Event(
                start_time="01:03",
                end_time="01:10",
                type="motion",
                description="Motion continues",
            ),
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 1
        assert result[0].start_time == 50.0
        assert result[0].end_time == 70.0
        assert result[0].description == "Motion starts | Motion continues"

    @pytest.mark.asyncio
    async def test_merges_overlapping_hh_mm_ss_events(self):
        """Overlapping HH:MM:SS events of the same type should merge."""
        instance = create_vlm_structured_instance()
        events = [
            Event(
                start_time="00:01:00",
                end_time="00:01:10",
                type="fire",
                description="Fire starts",
            ),
            Event(
                start_time="00:01:05",
                end_time="00:01:20",
                type="fire",
                description="Fire spreads",
            ),
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 1
        assert result[0].start_time == 60.0
        assert result[0].end_time == 80.0
        assert "Fire starts" in result[0].description
        assert "Fire spreads" in result[0].description

    @pytest.mark.asyncio
    async def test_merges_overlapping_mm_ss_events(self):
        """Overlapping MM:SS events of the same type should merge."""
        instance = create_vlm_structured_instance()
        events = [
            Event(
                start_time="01:00",
                end_time="01:10",
                type="alert",
                description="Alert A",
            ),
            Event(
                start_time="01:05",
                end_time="01:20",
                type="alert",
                description="Alert B",
            ),
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 1
        assert result[0].start_time == 60.0
        assert result[0].end_time == 80.0

    @pytest.mark.asyncio
    async def test_does_not_merge_video_timestamps_beyond_adjacent_threshold(self):
        """Video-timestamp events beyond the adjacency gap should stay separate."""
        instance = create_vlm_structured_instance(time_adjacent_threshold=4.0)
        events = [
            Event(
                start_time="00:01:00",
                end_time="00:01:10",
                type="motion",
                description="Motion A",
            ),
            Event(
                start_time="01:20",
                end_time="01:30",
                type="motion",
                description="Motion B",
            ),
        ]

        result = await instance._merge_similar_events(events)

        # Gap is 10 seconds (80 - 70), threshold is 4
        assert len(result) == 2
        assert result[0].start_time == 60.0
        assert result[1].start_time == 80.0

    @pytest.mark.asyncio
    async def test_does_not_merge_different_types_with_video_timestamps(self):
        """Different types with overlapping video timestamps should not merge."""
        instance = create_vlm_structured_instance()
        events = [
            Event(
                start_time="01:00",
                end_time="01:10",
                type="fire",
                description="Fire",
            ),
            Event(
                start_time="00:01:05",
                end_time="00:01:15",
                type="theft",
                description="Theft",
            ),
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 2
        assert {e.type for e in result} == {"fire", "theft"}

    @pytest.mark.asyncio
    async def test_chain_merges_mixed_video_timestamp_formats(self):
        """A->B->C chain merge should work across MM:SS and HH:MM:SS formats."""
        instance = create_vlm_structured_instance()
        events = [
            Event(
                start_time="00:01:00",
                end_time="00:01:10",
                type="chain",
                description="A",
            ),
            Event(
                start_time="01:08",
                end_time="01:18",
                type="chain",
                description="B",
            ),
            Event(
                start_time="00:01:16",
                end_time="00:01:26",
                type="chain",
                description="C",
            ),
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 1
        assert result[0].start_time == 60.0
        assert result[0].end_time == 86.0
        assert "A" in result[0].description
        assert "B" in result[0].description
        assert "C" in result[0].description

    @pytest.mark.asyncio
    async def test_merge_similar_events_empty_list(self):
        """Test that empty event list returns empty list."""
        instance = create_vlm_structured_instance()
        result = await instance._merge_similar_events([])
        assert result == []

    @pytest.mark.asyncio
    async def test_merge_similar_events_single_event(self):
        """Test that single event is returned unchanged."""
        instance = create_vlm_structured_instance()
        event = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected",
        )

        result = await instance._merge_similar_events([event])

        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 10.0
        assert result[0].type == "fire"

    @pytest.mark.asyncio
    async def test_merge_similar_events_overlapping_same_type(self):
        """Test merging of overlapping events with same type."""
        instance = create_vlm_structured_instance()
        events = [
            Event(
                start_time=0.0, end_time=10.0, type="fire", description="Fire starts"
            ),
            Event(
                start_time=8.0, end_time=18.0, type="fire", description="Fire spreads"
            ),
            Event(
                start_time=16.0,
                end_time=25.0,
                type="fire",
                description="Fire continues",
            ),
        ]

        result = await instance._merge_similar_events(events)

        # All three events should merge into one
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 25.0
        assert result[0].type == "fire"
        assert "Fire starts" in result[0].description
        assert "Fire spreads" in result[0].description
        assert "Fire continues" in result[0].description

    @pytest.mark.asyncio
    async def test_merge_similar_events_different_types_no_merge(self):
        """Test that events of different types are not merged."""
        instance = create_vlm_structured_instance()
        events = [
            Event(start_time=0.0, end_time=10.0, type="fire", description="Fire"),
            Event(start_time=5.0, end_time=15.0, type="theft", description="Theft"),
            Event(
                start_time=10.0, end_time=20.0, type="accident", description="Accident"
            ),
        ]

        result = await instance._merge_similar_events(events)

        # No merging - all events should be separate
        assert len(result) == 3
        types = {e.type for e in result}
        assert types == {"fire", "theft", "accident"}

    @pytest.mark.asyncio
    async def test_merge_similar_events_adjacent_within_threshold(self):
        """Test merging of adjacent events within threshold."""
        instance = create_vlm_structured_instance(
            time_overlap_threshold=0.1,
            time_adjacent_threshold=5.0,  # 5 second threshold
        )
        events = [
            Event(start_time=0.0, end_time=10.0, type="motion", description="Motion 1"),
            Event(
                start_time=14.0, end_time=24.0, type="motion", description="Motion 2"
            ),
        ]

        result = await instance._merge_similar_events(events)

        # Gap is 4 seconds (14.0 - 10.0), threshold is 5, should merge
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 24.0

    @pytest.mark.asyncio
    async def test_merge_similar_events_beyond_threshold_no_merge(self):
        """Test that events beyond threshold are not merged."""
        instance = create_vlm_structured_instance(
            time_overlap_threshold=0.1,
            time_adjacent_threshold=4.0,
        )
        events = [
            Event(start_time=0.0, end_time=10.0, type="alert", description="Alert 1"),
            Event(start_time=20.0, end_time=30.0, type="alert", description="Alert 2"),
        ]

        result = await instance._merge_similar_events(events)

        # Gap is 10 seconds, threshold is 4, should NOT merge
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_merge_similar_events_chronological_order(self):
        """Test that merged events are returned in chronological order."""
        instance = create_vlm_structured_instance()
        events = [
            Event(start_time=20.0, end_time=30.0, type="event", description="Third"),
            Event(start_time=0.0, end_time=10.0, type="event", description="First"),
            Event(start_time=50.0, end_time=60.0, type="event", description="Fourth"),
            Event(start_time=10.0, end_time=20.0, type="other", description="Second"),
        ]

        result = await instance._merge_similar_events(events)

        # Results should be sorted by start_time
        for i in range(len(result) - 1):
            assert result[i].start_time <= result[i + 1].start_time

    @pytest.mark.asyncio
    async def test_merge_similar_events_complex_scenario(self):
        """Test complex scenario with multiple types and partial merges."""
        instance = create_vlm_structured_instance(
            time_overlap_threshold=0.1,
            time_adjacent_threshold=5.0,
        )
        events = [
            # Fire events - should merge (overlapping)
            Event(start_time=0.0, end_time=10.0, type="fire", description="Fire A"),
            Event(start_time=8.0, end_time=18.0, type="fire", description="Fire B"),
            # Theft events - separate (beyond threshold)
            Event(start_time=5.0, end_time=12.0, type="theft", description="Theft A"),
            Event(start_time=30.0, end_time=40.0, type="theft", description="Theft B"),
            # Single accident event
            Event(
                start_time=20.0, end_time=25.0, type="accident", description="Accident"
            ),
        ]

        result = await instance._merge_similar_events(events)

        # Expected: 1 merged fire, 2 separate thefts, 1 accident = 4 events
        assert len(result) == 4

        # Find the merged fire event
        fire_events = [e for e in result if e.type == "fire"]
        assert len(fire_events) == 1
        assert fire_events[0].start_time == 0.0
        assert fire_events[0].end_time == 18.0
        assert "Fire A" in fire_events[0].description
        assert "Fire B" in fire_events[0].description

        # Verify theft events are separate
        theft_events = [e for e in result if e.type == "theft"]
        assert len(theft_events) == 2

    @pytest.mark.asyncio
    async def test_merge_similar_events_chain_merge(self):
        """Test chain merging where A overlaps B, B overlaps C."""
        instance = create_vlm_structured_instance()
        events = [
            Event(start_time=0.0, end_time=10.0, type="chain", description="A"),
            Event(start_time=8.0, end_time=18.0, type="chain", description="B"),
            Event(start_time=16.0, end_time=26.0, type="chain", description="C"),
            Event(start_time=24.0, end_time=34.0, type="chain", description="D"),
        ]

        result = await instance._merge_similar_events(events)

        # All should chain merge into one
        assert len(result) == 1
        assert result[0].start_time == 0.0
        assert result[0].end_time == 34.0

    @pytest.mark.asyncio
    async def test_merge_similar_events_preserves_all_descriptions(self):
        """Test that all descriptions are preserved in merged events."""
        instance = create_vlm_structured_instance()
        descriptions = [
            "Person enters frame",
            "Person picks up object",
            "Person walks toward exit",
        ]
        events = [
            Event(
                start_time=i * 5, end_time=(i + 1) * 8, type="person", description=desc
            )
            for i, desc in enumerate(descriptions)
        ]

        result = await instance._merge_similar_events(events)

        assert len(result) == 1
        for desc in descriptions:
            assert desc in result[0].description

    @pytest.mark.asyncio
    async def test_merge_similar_events_uses_llm_for_descriptions(self):
        """Test that LLM is used to merge descriptions instead of simple concatenation."""
        instance = create_vlm_structured_instance(enable_llm_merging=True)

        # Track calls to the mock LLM merge function
        llm_calls = []
        original_merge_fn = instance._merge_descriptions_with_llm

        async def tracking_merge_fn(event_type: str, descriptions: list) -> str:
            llm_calls.append({"event_type": event_type, "descriptions": descriptions})
            return await original_merge_fn(event_type, descriptions)

        instance._merge_descriptions_with_llm = tracking_merge_fn

        events = [
            Event(
                start_time=0.0,
                end_time=10.0,
                type="fire",
                description="Fire detected in zone A",
            ),
            Event(
                start_time=8.0,
                end_time=18.0,
                type="fire",
                description="Fire spreading to zone B",
            ),
        ]

        result = await instance._merge_similar_events(events)

        # Verify LLM was called for description merging
        assert len(llm_calls) == 1
        assert llm_calls[0]["event_type"] == "fire"
        assert len(llm_calls[0]["descriptions"]) == 2
        assert "Fire detected in zone A" in llm_calls[0]["descriptions"]
        assert "Fire spreading to zone B" in llm_calls[0]["descriptions"]

        # Verify the result uses the LLM-merged description
        assert len(result) == 1
        assert result[0].type == "fire"
        # The mock uses " + " to join descriptions
        assert " + " in result[0].description

    @pytest.mark.asyncio
    async def test_merge_descriptions_with_llm_single_description(self):
        """Test that LLM merge returns single description unchanged."""
        instance = create_vlm_structured_instance()

        result = await instance._merge_descriptions_with_llm(
            "fire", ["Single fire event"]
        )

        assert result == "Single fire event"

    @pytest.mark.asyncio
    async def test_merge_descriptions_with_llm_multiple_descriptions(self):
        """Test that LLM merge combines multiple descriptions."""
        instance = create_vlm_structured_instance()

        descriptions = [
            "Fire started in kitchen",
            "Fire spread to living room",
            "Fire contained",
        ]
        result = await instance._merge_descriptions_with_llm("fire", descriptions)

        # The mock uses " + " to join descriptions
        assert "Fire started in kitchen" in result
        assert "Fire spread to living room" in result
        assert "Fire contained" in result


# =============================================================================
# Enable LLM Merging Flag Tests
# =============================================================================


class TestEnableLlmMerging:
    """Tests for the enable_llm_merging parameter functionality."""

    @pytest.mark.asyncio
    async def test_llm_merging_disabled_uses_simple_concatenation(self):
        """Test that when enable_llm_merging is False, simple concatenation is used."""
        instance = create_vlm_structured_instance(enable_llm_merging=False)

        # Track if LLM merge method was called
        llm_calls = []

        async def tracking_merge_fn(event_type: str, descriptions: list) -> str:
            llm_calls.append({"event_type": event_type, "descriptions": descriptions})
            return " + ".join(descriptions)

        instance._merge_descriptions_with_llm = tracking_merge_fn

        events = [
            Event(
                start_time=0.0,
                end_time=10.0,
                type="fire",
                description="Fire detected in zone A",
            ),
            Event(
                start_time=8.0,
                end_time=18.0,
                type="fire",
                description="Fire spreading to zone B",
            ),
        ]

        result = await instance._merge_similar_events(events)

        # LLM should NOT have been called since enable_llm_merging=False
        assert len(llm_calls) == 0

        # Result should use simple concatenation with " | "
        assert len(result) == 1
        assert " | " in result[0].description
        assert "Fire detected in zone A" in result[0].description
        assert "Fire spreading to zone B" in result[0].description

    @pytest.mark.asyncio
    async def test_llm_merging_enabled_calls_llm(self):
        """Test that when enable_llm_merging is True, LLM is used for merging."""
        instance = create_vlm_structured_instance(enable_llm_merging=True)

        # Track if LLM merge method was called
        llm_calls = []

        async def tracking_merge_fn(event_type: str, descriptions: list) -> str:
            llm_calls.append({"event_type": event_type, "descriptions": descriptions})
            return " + ".join(descriptions)

        instance._merge_descriptions_with_llm = tracking_merge_fn

        events = [
            Event(
                start_time=0.0,
                end_time=10.0,
                type="fire",
                description="Fire detected in zone A",
            ),
            Event(
                start_time=8.0,
                end_time=18.0,
                type="fire",
                description="Fire spreading to zone B",
            ),
        ]

        result = await instance._merge_similar_events(events)

        # LLM should have been called since enable_llm_merging=True
        assert len(llm_calls) == 1
        assert llm_calls[0]["event_type"] == "fire"

        # Result should use LLM-merged description (mock uses " + ")
        assert len(result) == 1
        assert " + " in result[0].description

    @pytest.mark.asyncio
    async def test_llm_merging_disabled_single_event_unchanged(self):
        """Test that single event is unchanged when LLM merging is disabled."""
        instance = create_vlm_structured_instance(enable_llm_merging=False)

        event = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected",
        )

        result = await instance._merge_similar_events([event])

        # Single event should be returned unchanged
        assert len(result) == 1
        assert result[0].description == "Fire detected"

    @pytest.mark.asyncio
    async def test_llm_merging_enabled_single_event_unchanged(self):
        """Test that single event is unchanged when LLM merging is enabled."""
        instance = create_vlm_structured_instance(enable_llm_merging=True)

        llm_calls = []

        async def tracking_merge_fn(event_type: str, descriptions: list) -> str:
            llm_calls.append({"event_type": event_type, "descriptions": descriptions})
            if len(descriptions) == 1:
                return descriptions[0]
            return " + ".join(descriptions)

        instance._merge_descriptions_with_llm = tracking_merge_fn

        event = Event(
            start_time=0.0,
            end_time=10.0,
            type="fire",
            description="Fire detected",
        )

        result = await instance._merge_similar_events([event])

        # Single event should be returned unchanged, no LLM call needed
        assert len(result) == 1
        assert result[0].description == "Fire detected"
        assert len(llm_calls) == 0  # No merge needed for single event

    @pytest.mark.asyncio
    async def test_llm_merging_disabled_multiple_types(self):
        """Test that multiple event types work correctly with LLM merging disabled."""
        instance = create_vlm_structured_instance(enable_llm_merging=False)

        events = [
            Event(start_time=0.0, end_time=10.0, type="fire", description="Fire A"),
            Event(start_time=8.0, end_time=18.0, type="fire", description="Fire B"),
            Event(start_time=5.0, end_time=15.0, type="theft", description="Theft A"),
            Event(start_time=12.0, end_time=22.0, type="theft", description="Theft B"),
        ]

        result = await instance._merge_similar_events(events)

        # Should have 2 merged events (one fire, one theft)
        assert len(result) == 2

        fire_events = [e for e in result if e.type == "fire"]
        theft_events = [e for e in result if e.type == "theft"]

        # Both should use simple concatenation
        assert len(fire_events) == 1
        assert " | " in fire_events[0].description
        assert "Fire A" in fire_events[0].description
        assert "Fire B" in fire_events[0].description

        assert len(theft_events) == 1
        assert " | " in theft_events[0].description
        assert "Theft A" in theft_events[0].description
        assert "Theft B" in theft_events[0].description

    @pytest.mark.asyncio
    async def test_llm_merging_default_is_disabled(self):
        """Test that LLM merging is disabled by default."""
        instance = create_vlm_structured_instance()  # No explicit enable_llm_merging

        # Default should be disabled
        assert instance.enable_llm_merging is False
