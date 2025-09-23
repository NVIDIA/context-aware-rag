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

"""tool.py: File contains Tool class"""

from abc import ABC, abstractmethod
from typing import Optional, Dict

from vss_ctx_rag.utils.ctx_rag_logger import logger


class Tool(ABC):
    """Tool: This is a interface class that
    should be implemented to add a Tool that can be used in Functions
    """

    def __init__(self, name, config=None, tools=None) -> None:
        self.name = name
        self._tools = {}
        if tools:
            for tool_type, tool in tools.items():
                logger.info(f"Adding tool {tool_type} to {self.name}")
                self.add_tool(tool_type, tool)
        self.config = config

    def add_tool(self, tool_type: str, tool: "Tool") -> None:
        """Add a tool to this tool's collection of tools.

        Args:
            tool_type: The type of tool being added (e.g. 'embedding', 'reranker')
            tool: The tool instance to add
        """
        if tool_type in self._tools:
            logger.info(f"Updating tool {tool_type} in {self.name}")
        else:
            logger.info(f"Adding tool {tool_type} to {self.name}")
        self._tools[tool_type] = tool

    def get_tool(self, tool_type: str) -> "Tool":
        """Get a tool by its type.

        Args:
            tool_type: The type of tool to retrieve

        Returns:
            The tool instance if found, None otherwise
        """
        return self._tools.get(tool_type)

    @abstractmethod
    def update_tool(
        self, config, tools: Optional[Dict[str, Dict[str, "Tool"]]] = None
    ) -> "Tool":
        """Update a tool instance from a configuration.

        Args:
            config: The configuration for the tool
        """
        pass
