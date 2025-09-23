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

"""Tool factory for creating tools based on configuration."""

import importlib
from typing import Dict, List

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.context_manager_models import ContextManagerConfig
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.dependency_utils import topological_sort

from vss_ctx_rag.models.tool_models import _TOOL_IMPLEMENTATION_REGISTRY

ToolType = str
ToolName = str


class ToolFactory:
    """Factory class for creating tools based on configuration using the registry."""

    @staticmethod
    def _get_tool_dependencies(
        config: ContextManagerConfig,
    ) -> Dict[ToolType, List[ToolName]]:
        """
        Extract tool dependencies for topological sorting.

        Args:
            config: The configuration containing tool definitions

        Returns:
            Dict mapping tool names to lists of their dependencies (tool names they depend on)
        """
        dependencies = {}

        for tool_name, tool_config in config.tools.root.items():
            dependencies[tool_name] = []

            if hasattr(tool_config, "tools") and tool_config.tools:
                resolved_tools = tool_config.tools
                if hasattr(tool_config.tools, "root"):
                    resolved_tools = tool_config.tools.root

                for tool_keyword_name, referenced_tool_ref in resolved_tools.items():
                    if hasattr(referenced_tool_ref, "tool_name"):
                        referenced_tool_name = referenced_tool_ref.tool_name
                        if referenced_tool_name != tool_name:
                            dependencies[tool_name].append(referenced_tool_name)

        return dependencies

    @staticmethod
    def _topological_sort_tools(
        dependencies: Dict[ToolType, List[ToolName]],
    ) -> List[ToolName]:
        """
        Perform topological sort on tools using Kahn's algorithm.

        Args:
            dependencies: Dict mapping tool names to lists of their dependencies

        Returns:
            List of tool names in topological order (dependencies first)

        Raises:
            ValueError: If circular dependency is detected
        """
        return topological_sort(dependencies)

    @staticmethod
    def _get_tool_dependencies_for_creation(
        tool_name: ToolName,
        tool_config,
        all_tools: Dict[ToolType, Dict[ToolName, Tool]],
    ) -> Dict[ToolName, Tool]:
        """
        Get the dependencies that should be passed to a tool during creation.

        Args:
            tool_name: Name of the tool being created
            tool_config: Configuration for the tool
            all_tools: Dictionary of all available tools by category

        Returns:
            Dict mapping tool keyword names to tool instances for dependencies
        """
        creation_dependencies = {}

        if hasattr(tool_config, "tools") and tool_config.tools:
            resolved_tools = tool_config.tools
            if hasattr(tool_config.tools, "root"):
                resolved_tools = tool_config.tools.root

            for tool_keyword_name, referenced_tool_ref in resolved_tools.items():
                referenced_tool_name = None
                referenced_tool_type = None

                if hasattr(referenced_tool_ref, "tool_name") and hasattr(
                    referenced_tool_ref, "tool_type"
                ):
                    referenced_tool_name = referenced_tool_ref.tool_name
                    referenced_tool_type = referenced_tool_ref.tool_type
                else:
                    logger.warning(
                        f"Unknown tool reference type for {tool_keyword_name} in tool {tool_name}: {type(referenced_tool_ref)}"
                    )
                    continue

                if referenced_tool_name == tool_name:
                    logger.warning(
                        f"Tool '{tool_name}' cannot reference itself as '{tool_keyword_name}', skipping"
                    )
                    continue

                if (
                    referenced_tool_type in all_tools
                    and referenced_tool_name in all_tools[referenced_tool_type]
                ):
                    creation_dependencies[tool_keyword_name] = all_tools[
                        referenced_tool_type
                    ][referenced_tool_name]
                    logger.debug(
                        f"Added dependency '{referenced_tool_name}' as '{tool_keyword_name}' for tool '{tool_name}' creation"
                    )
                else:
                    logger.debug(
                        f"Dependency '{referenced_tool_name}' of type '{referenced_tool_type}' not yet available for tool '{tool_name}' creation"
                    )

        return creation_dependencies

    @staticmethod
    def create_all_tools(
        config: ContextManagerConfig,
        previous_tools: Dict[ToolType, Dict[ToolName, Tool]],
    ) -> Dict[ToolType, Dict[ToolName, Tool]]:
        """Create all tools based on the configuration in topological dependency order.

        Args:
            config: Configuration containing all tool configurations
            tools: Existing tools dictionary to update

        Returns:
            Dict[str, Dict[str, Tool]]: Dictionary mapping tool categories to
                                      dictionaries of tool instances by name

        Raises:
            ValueError: If tool type is not registered or tool creation fails
        """

        current_tools = previous_tools or {}

        for tool_category in _TOOL_IMPLEMENTATION_REGISTRY.keys():
            if tool_category not in current_tools:
                current_tools[tool_category] = {}

        dependencies = ToolFactory._get_tool_dependencies(config)
        sorted_tool_names = ToolFactory._topological_sort_tools(dependencies)

        logger.info(f"Creating tools in topological order: {sorted_tool_names}")
        for tool_name in sorted_tool_names:
            if tool_name not in config.tools.root:
                continue

            tool_config = config.tools.root[tool_name]
            tool_type = tool_config.tool_type

            logger.info(f"Processing tool: {tool_name} of type {tool_type}")

            tool_updated = False
            if tool_name in current_tools.get(tool_type, {}):
                logger.info(
                    f"Updating existing tool '{tool_name}' of type '{tool_type}'"
                )
                try:
                    creation_dependencies = (
                        ToolFactory._get_tool_dependencies_for_creation(
                            tool_name, tool_config, current_tools
                        )
                    )

                    current_tools[tool_type][tool_name].update_tool(
                        tool_config, creation_dependencies
                    )

                    tool_updated = True
                except Exception as e:
                    logger.error(
                        f"Failed to update existing tool '{tool_name}': {e}. Falling back to re-creation."
                    )

                    del current_tools[tool_type][tool_name]
                    tool_updated = False
            if tool_updated:
                continue
            if tool_type not in _TOOL_IMPLEMENTATION_REGISTRY:
                raise ValueError(
                    f"Tool type '{tool_type}' not found in implementation registry"
                )

            registry_info = _TOOL_IMPLEMENTATION_REGISTRY[tool_type]
            module_name = registry_info["module"]
            class_name = registry_info["class"]
            logger.debug(f"Module name: {module_name}")
            logger.debug(f"Class name: {class_name}")

            try:
                module = importlib.import_module(module_name)
                tool_class = getattr(module, class_name)
            except ImportError as e:
                raise ValueError(f"Failed to import module '{module_name}': {e}")
            except AttributeError as e:
                raise ValueError(
                    f"Class '{class_name}' not found in module '{module_name}': {e}"
                )

            creation_dependencies = ToolFactory._get_tool_dependencies_for_creation(
                tool_name, tool_config, current_tools
            )

            if creation_dependencies:
                logger.debug(
                    f"Creating tool '{tool_name}' with {len(creation_dependencies)} dependencies at creation time"
                )
                tool = tool_class(
                    name=tool_name, config=tool_config, tools=creation_dependencies
                )

                for tool_keyword_name, dependency_tool in creation_dependencies.items():
                    tool.add_tool(tool_keyword_name, dependency_tool)
                    logger.debug(
                        f"Added creation-time dependency '{tool_keyword_name}' to tool '{tool_name}'"
                    )
            else:
                logger.debug(
                    f"Creating tool '{tool_name}' without creation-time dependencies"
                )
                tool = tool_class(name=tool_name, config=tool_config)

            current_tools[tool_type][tool_name] = tool

        return current_tools

    @staticmethod
    def get_registered_tool_types() -> List[ToolType]:
        """Get all registered tool types from the registry.

        Returns:
            List[ToolType]: List of registered tool type names
        """
        return list(_TOOL_IMPLEMENTATION_REGISTRY.keys())
