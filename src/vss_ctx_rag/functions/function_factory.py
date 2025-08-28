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

"""Function factory for creating functions based on configuration."""

import importlib
from typing import Dict, List

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models.context_manager_models import (
    ContextManagerConfig,
)
from vss_ctx_rag.models.function_models import (
    _FUNCTION_IMPLEMENTATION_REGISTRY,
    FunctionConfig,
    FunctionRefsContainer,
    FunctionConfigsContainer,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger
from vss_ctx_rag.utils.dependency_utils import topological_sort

FunctionType = str
FunctionName = str


class FunctionFactory:
    """Factory class for creating functions based on configuration."""

    def __init__(
        self,
        config: ContextManagerConfig,
        tools: Dict[FunctionType, Dict[FunctionName, Tool]],
    ):
        """Initialize the function factory.

        Args:
            config: The complete context manager configuration
            tools: Dictionary mapping tool categories to dictionaries of tool instances
        """
        self.config = config
        self.tools = tools

    @staticmethod
    def _get_function_dependencies(
        config: ContextManagerConfig,
    ) -> Dict[FunctionType, List[FunctionName]]:
        """
        Extract function dependencies for topological sorting.

        Args:
            config: The configuration containing function definitions

        Returns:
            Dict mapping function names to lists of their dependencies (function names they depend on)
        """
        dependencies = {}

        for function_name, function_config in config.functions.root.items():
            dependencies[function_name] = []

            if hasattr(function_config, "functions") and function_config.functions:
                function_refs = function_config.functions
                function_refs_dict = function_refs
                if isinstance(function_refs, FunctionRefsContainer):
                    function_refs_dict = function_refs.root
                elif isinstance(function_refs, FunctionConfigsContainer):
                    function_refs_dict = function_refs.root

                for ref_function_name, ref_function_value in function_refs_dict.items():
                    if isinstance(ref_function_value, str):
                        # Avoid self-references
                        if ref_function_value != function_name:
                            dependencies[function_name].append(ref_function_value)
                    elif hasattr(ref_function_value, "function_name"):
                        ref_name = ref_function_value.function_name
                        # Avoid self-references
                        if ref_name != function_name:
                            dependencies[function_name].append(ref_name)

        return dependencies

    @staticmethod
    def _topological_sort_functions(
        dependencies: Dict[FunctionType, List[FunctionName]],
    ) -> List[FunctionName]:
        """
        Perform topological sort on functions using Kahn's algorithm.

        Args:
            dependencies: Dict mapping function names to lists of their dependencies

        Returns:
            List of function names in topological order (dependencies first)

        Raises:
            ValueError: If circular dependency is detected
        """
        return topological_sort(dependencies)

    def create_functions(self) -> Dict[FunctionName, Function]:
        """Create all functions based on the configuration in topological dependency order.

        Returns:
            Dict[FunctionName, Function]: Dictionary mapping function names to function instances,
            or a dictionary with "error" key if creation fails
        """
        try:
            functions = {}

            logger.info("Creating functions using FunctionFactory")

            # Get function dependencies and sort topologically
            dependencies = self._get_function_dependencies(self.config)
            sorted_function_names = self._topological_sort_functions(dependencies)

            logger.info(
                f"Creating functions in topological order: {sorted_function_names}"
            )

            for function_name in sorted_function_names:
                if function_name not in self.config.functions.root:
                    continue

                function_config = self.config.functions.root[function_name]

                function_instance = self.create_single_function(
                    function_name, function_config
                )
                functions[function_name] = function_instance

                self._add_sub_functions(function_instance, function_config, functions)

                function_instance.done()
                logger.debug(f"Completed setup for function: {function_name}")

            return functions
        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"Error creating functions: {e}")
            return {"error": str(e)}

    def create_single_function(
        self, function_name: FunctionName, function_config: FunctionConfig
    ) -> Function:
        """Create a single function based on configuration and add its tools.

        Args:
            function_name: Name of the function to create
            function_config: Configuration for the function

        Returns:
            Function: The created function instance with tools added

        Raises:
            ValueError: If function type not found or creation fails
        """
        logger.info(f"Creating single function: {function_name}")

        function_type = function_config.function_type
        logger.info(f"Adding function: {function_name} of type: {function_type}")
        if function_type not in _FUNCTION_IMPLEMENTATION_REGISTRY:
            raise ValueError(
                f"Function type '{function_type}' not found in implementation registry"
            )

        registry_info = _FUNCTION_IMPLEMENTATION_REGISTRY[function_type]
        module_name = registry_info["module"]
        class_name = registry_info["class"]

        try:
            module = importlib.import_module(module_name)
            function_class = getattr(module, class_name)
        except ImportError as e:
            raise ValueError(f"Failed to import module '{module_name}': {e}")
        except AttributeError as e:
            raise ValueError(
                f"Class '{class_name}' not found in module '{module_name}': {e}"
            )

        try:
            function_instance = function_class(function_name)

            if hasattr(function_config, "tools") and function_config.tools:
                logger.debug(f"Adding tools to function {function_name}")
                for (
                    tool_keyword_name,
                    tool_config,
                ) in function_config.tools.root.items():
                    tool_name = tool_config.tool_name
                    tool_type = tool_config.tool_type
                    logger.debug(
                        f"Looking for tool '{tool_name}' of type '{tool_type}'"
                    )

                    if (
                        tool_type not in self.tools
                        or tool_name not in self.tools[tool_type]
                    ):
                        raise ValueError(
                            f"Tool '{tool_name}' not found in available tools for function '{function_name}'"
                        )
                    tool_instance = self.tools[tool_type][tool_name]

                    if tool_instance is None:
                        raise ValueError(
                            f"Tool '{tool_name}' not found in available tools for function '{function_name}'"
                        )

                    function_instance.add_tool(tool_keyword_name, tool_instance)
                    logger.debug(
                        f"Added tool '{tool_name}' as '{tool_type}' to function '{function_name}'"
                    )

            function_instance.config(function_config)
            logger.info(f"Successfully created function: {function_name}")
            return function_instance

        except Exception as e:
            import traceback

            logger.error(traceback.format_exc())
            logger.error(f"Error creating function {function_name}: {e}")
            raise ValueError(f"Failed to create function '{function_name}': {e}")

    def _add_sub_functions(
        self,
        function_instance: Function,
        function_config: FunctionConfig,
        all_functions: Dict[FunctionName, Function],
    ) -> None:
        """Add sub-functions to a function instance.

        Args:
            function_instance: The function instance to add sub-functions to
            function_config: Configuration for the function
            all_functions: Dictionary of all created function instances
        """
        if hasattr(function_config, "functions") and function_config.functions:
            logger.debug(
                f"Adding sub-functions to function {function_instance.name}: {function_config.functions}"
            )
            for (
                function_keyword_name,
                sub_function_config,
            ) in function_config.functions.root.items():
                sub_function_name = sub_function_config.function_name
                logger.debug(f"Looking for sub-function '{sub_function_name}'")

                if sub_function_name not in all_functions:
                    raise ValueError(
                        f"Sub-function '{sub_function_name}' not found in available functions for function '{function_instance.name}'"
                    )

                sub_function_instance = all_functions[sub_function_name]

                if sub_function_instance is None:
                    raise ValueError(
                        f"Sub-function '{sub_function_name}' not found in available functions for function '{function_instance.name}'"
                    )

                function_instance.add_function(
                    function_keyword_name, sub_function_instance
                )
                logger.debug(
                    f"Added sub-function '{sub_function_name}' as '{function_keyword_name}' to function '{function_instance.name}'"
                )
