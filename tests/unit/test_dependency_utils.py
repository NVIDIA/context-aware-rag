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


from vss_ctx_rag.utils.dependency_utils import topological_sort


class TestDependencyUtils:
    """Test cases for dependency utility functions."""

    def test_topological_sort_simple_chain(self):
        """Test topological sort with simple dependency chain."""
        dependencies = {"A": ["B"], "B": ["C"], "C": []}

        result = topological_sort(dependencies)

        # C should come before B, B should come before A
        assert result.index("C") < result.index("B")
        assert result.index("B") < result.index("A")
        assert set(result) == {"A", "B", "C"}

    def test_topological_sort_no_dependencies(self):
        """Test topological sort with no dependencies."""
        dependencies = {"A": [], "B": [], "C": []}

        result = topological_sort(dependencies)

        # All items should be present, order doesn't matter for independent items
        assert set(result) == {"A", "B", "C"}
        assert len(result) == 3

    def test_topological_sort_multiple_dependencies(self):
        """Test topological sort with multiple dependencies."""
        dependencies = {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []}

        result = topological_sort(dependencies)

        # D should come first (no dependencies)
        # B and C should come before A
        assert result.index("D") < result.index("B")
        assert result.index("D") < result.index("C")
        assert result.index("B") < result.index("A")
        assert result.index("C") < result.index("A")
        assert set(result) == {"A", "B", "C", "D"}

    def test_topological_sort_empty_dict(self):
        """Test topological sort with empty dependencies."""
        dependencies = {}

        result = topological_sort(dependencies)

        assert result == []

    def test_topological_sort_single_item(self):
        """Test topological sort with single item."""
        dependencies = {"A": []}

        result = topological_sort(dependencies)

        assert result == ["A"]

    def test_topological_sort_complex_graph(self):
        """Test topological sort with complex dependency graph."""
        dependencies = {
            "frontend": ["backend", "database"],
            "backend": ["database", "auth"],
            "database": [],
            "auth": ["database"],
            "monitoring": ["backend"],
        }

        result = topological_sort(dependencies)

        # Database should be first (no dependencies)
        assert result[0] == "database"

        # Auth should come before backend
        assert result.index("auth") < result.index("backend")

        # Backend should come before frontend and monitoring
        assert result.index("backend") < result.index("frontend")
        assert result.index("backend") < result.index("monitoring")

        assert set(result) == {"frontend", "backend", "database", "auth", "monitoring"}

    def test_topological_sort_circular_dependency(self):
        """Test topological sort raises error for circular dependencies."""
        import pytest

        dependencies = {"A": ["B"], "B": ["C"], "C": ["A"]}

        with pytest.raises(ValueError) as excinfo:
            topological_sort(dependencies)
        assert "Circular dependency detected" in str(excinfo.value)
        assert (
            "A" in str(excinfo.value)
            or "B" in str(excinfo.value)
            or "C" in str(excinfo.value)
        )

    def test_topological_sort_self_dependency(self):
        """Test topological sort with self-dependency."""
        import pytest

        dependencies = {"A": ["A"]}

        with pytest.raises(ValueError) as excinfo:
            topological_sort(dependencies)
        assert "Circular dependency detected" in str(excinfo.value)
        assert "A" in str(excinfo.value)

    def test_topological_sort_implicit_dependencies(self):
        """Test topological sort with items appearing only in dependency lists."""
        dependencies = {"A": ["B", "C"]}  # B and C not defined as keys

        result = topological_sort(dependencies)

        # B and C should appear before A even though they're not keys
        assert "B" in result
        assert "C" in result
        assert result.index("B") < result.index("A")
        assert result.index("C") < result.index("A")
        assert len(result) == 3
