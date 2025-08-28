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

"""Dependency management utilities including topological sorting."""

from typing import Dict, List
from collections import defaultdict, deque


def topological_sort(dependencies: Dict[str, List[str]]) -> List[str]:
    """
    Perform topological sort using Kahn's algorithm.

    Args:
        dependencies: Dict mapping item names to lists of their dependencies

    Returns:
        List of item names in topological order (dependencies first)

    Raises:
        ValueError: If circular dependency is detected
    """
    # Calculate in-degree for each item
    in_degree = defaultdict(int)
    all_items = set(dependencies.keys())

    # Add all referenced items to the set
    for deps in dependencies.values():
        all_items.update(deps)

    # Initialize in-degree
    for item in all_items:
        in_degree[item] = 0

    # Calculate in-degrees
    for item, deps in dependencies.items():
        for dep in deps:
            in_degree[item] += 1

    # Start with items having no dependencies
    queue = deque([item for item in all_items if in_degree[item] == 0])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # Reduce in-degree for items that depend on current item
        for item, deps in dependencies.items():
            if current in deps:
                in_degree[item] -= 1
                if in_degree[item] == 0:
                    queue.append(item)

    # Check for circular dependencies
    if len(result) != len(all_items):
        remaining = [item for item in all_items if item not in result]
        raise ValueError(f"Circular dependency detected: {remaining}")

    return result
