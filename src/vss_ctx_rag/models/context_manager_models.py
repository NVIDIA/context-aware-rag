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

from typing import List

from pydantic import BaseModel, model_validator
import copy
from .tool_models import (
    ToolsContainer,
    ToolRefsContainer,
    ToolConfigsContainer,
)
from .function_models import (
    FunctionsContainer,
    FunctionConfig,
    FunctionRefsContainer,
    FunctionConfigsContainer,
)
from vss_ctx_rag.utils.ctx_rag_logger import logger


class ContextManagerSettings(BaseModel):
    """Settings for context manager configuration."""

    functions: List[str] = []
    uuid: str = "default"
    """List of function names to be added to the context manager."""


class ContextManagerConfig(BaseModel):
    tools: ToolsContainer
    functions: FunctionsContainer
    context_manager: ContextManagerSettings

    def _detect_function_cycles(
        self, function_name: str, visited: set, path: list
    ) -> None:
        """
        Helper method to detect cycles in function dependencies using DFS.

        Args:
            function_name: Current function being checked
            visited: Set of all visited functions
            path: Current path being explored

        Raises:
            ValueError: If a cycle is detected
        """
        if function_name in path:
            cycle = path[path.index(function_name) :] + [function_name]
            raise ValueError(
                f"Circular dependency detected in functions: {' -> '.join(cycle)}"
            )

        if function_name in visited:
            return

        visited.add(function_name)
        path.append(function_name)

        function_config = self.functions.root.get(function_name)
        if function_config and function_config.functions:
            function_refs = function_config.functions
            function_refs_dict = function_refs
            if isinstance(function_refs, FunctionRefsContainer):
                function_refs_dict = function_refs.root
            elif isinstance(function_refs, FunctionConfigsContainer):
                function_refs_dict = function_refs.root

            for ref_function_name, ref_function_value in function_refs_dict.items():
                if isinstance(ref_function_value, str):
                    self._detect_function_cycles(ref_function_value, visited, path)
                elif hasattr(ref_function_value, "function_name"):
                    self._detect_function_cycles(
                        ref_function_value.function_name, visited, path
                    )

        path.pop()

    def _detect_tool_cycles(self, tool_name: str, visited: set, path: list) -> None:
        """
        Helper method to detect cycles in tool dependencies using DFS.

        Args:
            tool_name: Name of the current tool
            visited: Set of all visited tools
            path: Current path being explored

        Raises:
            ValueError: If a cycle is detected
        """
        if tool_name in path:
            cycle = path[path.index(tool_name) :] + [tool_name]
            raise ValueError(
                f"Circular dependency detected in tools: {' -> '.join(cycle)}"
            )

        if tool_name in visited:
            return

        visited.add(tool_name)
        path.append(tool_name)

        tool_config = self.tools.root.get(tool_name)
        if tool_config and tool_config.tools:
            internal_tools = tool_config.tools
            if isinstance(internal_tools, ToolRefsContainer):
                internal_tools = internal_tools.root
            elif isinstance(internal_tools, ToolConfigsContainer):
                internal_tools = internal_tools.root

            for _, internal_tool_ref in internal_tools.items():
                if isinstance(internal_tool_ref, str):
                    if internal_tool_ref in self.tools.root:
                        self._detect_tool_cycles(internal_tool_ref, visited, path)
                elif hasattr(internal_tool_ref, "tool_name"):
                    self._detect_tool_cycles(internal_tool_ref.tool_name, visited, path)

        path.pop()

    @model_validator(mode="before")
    @classmethod
    def transform_functions_and_tools(cls, values):
        """Transform the functions and tools from YAML format to their respective config formats"""

        if isinstance(values, dict) and "functions" in values:
            functions_data = values["functions"]
            if isinstance(functions_data, dict):
                transformed_functions = {}
                for function_instance_name, function_config in functions_data.items():
                    if isinstance(function_config, dict):
                        function_type = function_config.get("type")
                        if not function_type:
                            raise ValueError(
                                f"Function '{function_instance_name}' is missing required 'type' field"
                            )

                        if (
                            "tools" in function_config
                            and function_config["tools"] is not None
                        ):
                            if not isinstance(
                                function_config["tools"],
                                (ToolRefsContainer, ToolConfigsContainer),
                            ):
                                function_config["tools"] = ToolRefsContainer(
                                    root=function_config["tools"]
                                )

                        if (
                            "functions" in function_config
                            and function_config["functions"] is not None
                        ):
                            if not isinstance(
                                function_config["functions"],
                                (FunctionRefsContainer, FunctionConfigsContainer),
                            ):
                                function_config["functions"] = FunctionRefsContainer(
                                    root=function_config["functions"]
                                )

                        params_data = function_config.get("params", {})

                        context_uuid = None
                        if isinstance(values, dict) and "context_manager" in values:
                            context_manager_data = values["context_manager"]
                            if (
                                isinstance(context_manager_data, dict)
                                and "uuid" in context_manager_data
                            ):
                                context_uuid = context_manager_data["uuid"]

                        function_config["_uuid"] = context_uuid

                        from .function_models import _FUNCTION_CONFIG_REGISTRY

                        if function_type in _FUNCTION_CONFIG_REGISTRY:
                            mapping = _FUNCTION_CONFIG_REGISTRY[function_type]
                            try:
                                import importlib

                                module = importlib.import_module(mapping["module"])
                                config_class = getattr(module, mapping["class"])

                                validated_config = config_class(
                                    params=params_data,
                                    tools=function_config.get("tools"),
                                    functions=function_config.get("functions"),
                                )

                                if hasattr(validated_config, "params"):
                                    if hasattr(validated_config.params, "model_dump"):
                                        params_data = (
                                            validated_config.params.model_dump()
                                        )
                                    elif hasattr(validated_config.params, "dict"):
                                        params_data = validated_config.params.dict()
                                    else:
                                        params_data = function_config.get("params", {})

                            except Exception as e:
                                logger.warning(
                                    f"Failed to apply defaults for function {function_instance_name} of type {function_type}: {e}"
                                )
                                raise e
                        else:
                            raise ValueError(
                                f"Unknown function type: {function_type}. "
                                f"Available types: {list(_FUNCTION_CONFIG_REGISTRY.keys())}"
                            )

                        transformed_functions[function_instance_name] = FunctionConfig(
                            function_name=function_instance_name,
                            function_type=function_type,
                            params=params_data,
                            tools=function_config.get("tools"),
                            functions=function_config.get("functions"),
                            _uuid=context_uuid,
                        )
                    else:
                        transformed_functions[function_instance_name] = function_config

                values["functions"] = transformed_functions

        if isinstance(values, dict) and "tools" in values:
            tools_data = values["tools"]
            if isinstance(tools_data, dict):
                transformed_tools = {}

                for tool_name, tool_config in tools_data.items():
                    if isinstance(tool_config, dict):
                        tool_type = tool_config.get("type")
                        if not tool_type:
                            raise ValueError(
                                f"Tool '{tool_name}' is missing required 'type' field"
                            )

                        tools_field = tool_config.get("tools")
                        validated_tools = None
                        if tools_field is not None:
                            if not isinstance(
                                tools_field,
                                (ToolRefsContainer, ToolConfigsContainer),
                            ):
                                validated_tools = ToolRefsContainer(root=tools_field)
                            else:
                                validated_tools = tools_field

                        if "params" in tool_config:
                            params_data = tool_config["params"]
                        else:
                            params_data = {
                                k: v
                                for k, v in tool_config.items()
                                if k not in ["type", "tools"]
                            }

                        context_uuid = None
                        if isinstance(values, dict) and "context_manager" in values:
                            context_manager_data = values["context_manager"]
                            if (
                                isinstance(context_manager_data, dict)
                                and "uuid" in context_manager_data
                            ):
                                context_uuid = context_manager_data["uuid"]

                        if context_uuid and context_uuid != "default":
                            tool_config["_uuid"] = context_uuid

                        from .tool_models import _TOOL_CONFIG_REGISTRY

                        if tool_type in _TOOL_CONFIG_REGISTRY:
                            mapping = _TOOL_CONFIG_REGISTRY[tool_type]
                            try:
                                import importlib

                                module = importlib.import_module(mapping["module"])
                                config_class = getattr(module, mapping["class"])

                                validated_config = config_class(
                                    tools=validated_tools, **params_data
                                )

                                if hasattr(validated_config, "model_dump"):
                                    validated_dict = validated_config.model_dump()
                                    params_data = {
                                        k: v
                                        for k, v in validated_dict.items()
                                        if k != "tools"
                                    }
                                elif hasattr(validated_config, "dict"):
                                    validated_dict = validated_config.dict()
                                    params_data = {
                                        k: v
                                        for k, v in validated_dict.items()
                                        if k != "tools"
                                    }

                            except Exception as e:
                                logger.warning(
                                    f"Failed to apply defaults for tool {tool_name} of type {tool_type}: {e}"
                                )

                        specific_tool = {
                            "tool_type": tool_type,
                            "tool_name": tool_name,
                            "params": params_data,
                            "tools": validated_tools,
                        }

                        # Preserve the _uuid if it was injected
                        if "_uuid" in tool_config:
                            specific_tool["_uuid"] = tool_config["_uuid"]
                        transformed_tools[tool_name] = specific_tool

                values["tools"] = transformed_tools
        return values

    @model_validator(mode="after")
    def validate_function_and_tool_references(self):
        """
        Makes sure function and tool references are valid and have no circular dependencies.
        """
        tools = self.tools.root
        functions = self.functions.root

        visited_functions = set()
        for function_name in functions:
            if function_name not in visited_functions:
                self._detect_function_cycles(function_name, visited_functions, [])

        visited_tools = set()
        for tool_name in tools:
            if tool_name not in visited_tools:
                self._detect_tool_cycles(tool_name, visited_tools, [])

        for function_name, function_config in functions.items():
            function_tools = function_config.tools
            if isinstance(function_tools, ToolRefsContainer):
                function_tools = function_tools.root
            elif isinstance(function_tools, ToolConfigsContainer):
                function_tools = function_tools.root

            if function_tools:
                for tool_role, tool_name in function_tools.items():
                    if isinstance(tool_name, str) and tool_name not in tools:
                        raise ValueError(
                            f"Tool '{tool_name}' referenced in function '{function_name}' "
                            f"is not found in tools configuration. Available tools: {list(tools.keys())}"
                        )

            function_refs = function_config.functions

            if function_refs:
                function_refs_dict = function_refs
                if isinstance(function_refs, FunctionRefsContainer):
                    function_refs_dict = function_refs.root
                elif isinstance(function_refs, FunctionConfigsContainer):
                    function_refs_dict = function_refs.root

                for ref_function_name, ref_function_value in function_refs_dict.items():
                    if isinstance(ref_function_value, str):
                        if ref_function_value not in functions:
                            raise ValueError(
                                f"Function '{ref_function_value}' referenced in function '{function_name}' "
                                f"is not found in functions configuration. Available functions: {list(functions.keys())}"
                            )

        for tool_name, tool_config in tools.items():
            if tool_config.tools:
                internal_tools = tool_config.tools
                if isinstance(internal_tools, ToolRefsContainer):
                    internal_tools = internal_tools.root
                for internal_tool_role, internal_tool_ref in internal_tools.items():
                    if isinstance(internal_tool_ref, str):
                        if internal_tool_ref not in tools:
                            raise ValueError(
                                f"Tool '{internal_tool_ref}' referenced in "
                                f"{tool_name}.tools.{internal_tool_role} "
                                f"is not found in tools configuration. Available tools: {list(tools.keys())}"
                            )
                    elif hasattr(internal_tool_ref, "tool_name"):
                        ref_tool_name = internal_tool_ref.tool_name
                        if ref_tool_name not in tools:
                            raise ValueError(
                                f"Tool '{ref_tool_name}' referenced in "
                                f"{tool_name}.tools.{internal_tool_role} "
                                f"is not found in tools configuration. Available tools: {list(tools.keys())}"
                            )

        for function_name in self.context_manager.functions:
            if function_name not in functions:
                raise ValueError(
                    f"Function '{function_name}' specified in context_manager.functions "
                    f"is not found in functions configuration. Available functions: {list(functions.keys())}"
                )

        return self

    def _validate_tool_types(self) -> None:
        """
        Validates that tool configurations comply with their ALLOWED_TOOL_TYPES.
        If these mappings are empty, any tool type is allowed.
        """
        for tool_name, tool_config in self.tools.root.items():
            if tool_config.tools:
                allowed_types = getattr(tool_config, "ALLOWED_TOOL_TYPES", {})

                if hasattr(tool_config.tools, "root"):
                    resolved_tools = tool_config.tools.root
                else:
                    resolved_tools = tool_config.tools

                if resolved_tools:
                    present_tools = {}
                    for tool_keyword, resolved_tool in resolved_tools.items():
                        if hasattr(resolved_tool, "tool_type"):
                            tool_type = resolved_tool.tool_type
                            if tool_type not in present_tools:
                                present_tools[tool_type] = []
                            present_tools[tool_type].append(tool_keyword)

                    if allowed_types:
                        for tool_type, keywords in present_tools.items():
                            if tool_type not in allowed_types:
                                raise ValueError(
                                    f"Tool '{tool_name}' contains tools of invalid type: {tool_type}. "
                                    f"Allowed types: {list(allowed_types.keys())}"
                                )

                            allowed_keywords = allowed_types[tool_type]
                            if allowed_keywords:
                                invalid_keywords = set(keywords) - set(allowed_keywords)
                                if invalid_keywords:
                                    raise ValueError(
                                        f"Tool '{tool_name}' contains invalid keywords {invalid_keywords} for type {tool_type}. "
                                        f"Allowed keywords: {allowed_keywords}"
                                    )

        for function_name, function_config in self.functions.root.items():
            if function_config.tools:
                allowed_types = getattr(function_config, "ALLOWED_TOOL_TYPES", {})

                if hasattr(function_config.tools, "root"):
                    resolved_tools = function_config.tools.root
                else:
                    resolved_tools = function_config.tools

                if resolved_tools:
                    present_tools = {}
                    for tool_keyword, resolved_tool in resolved_tools.items():
                        if hasattr(resolved_tool, "tool_type"):
                            tool_type = resolved_tool.tool_type
                            if tool_type not in present_tools:
                                present_tools[tool_type] = []
                            present_tools[tool_type].append(tool_keyword)

                    if allowed_types:
                        for tool_type, keywords in present_tools.items():
                            if tool_type not in allowed_types:
                                raise ValueError(
                                    f"Function '{function_name}' contains tools of invalid type: {tool_type}. "
                                    f"Allowed types: {list(allowed_types.keys())}"
                                )

                            allowed_keywords = allowed_types[tool_type]
                            if allowed_keywords:
                                invalid_keywords = set(keywords) - set(allowed_keywords)
                                if invalid_keywords:
                                    raise ValueError(
                                        f"Function '{function_name}' contains invalid keywords {invalid_keywords} for type {tool_type}. "
                                        f"Allowed keywords: {allowed_keywords}"
                                    )

    def _validate_function_types(self) -> None:
        """
        Validates that function configurations comply with their ALLOWED_FUNCTION_TYPES.
        If these mappings are empty, any function type is allowed.
        """
        for function_name, function_config in self.functions.root.items():
            if function_config.functions:
                allowed_types = getattr(function_config, "ALLOWED_FUNCTION_TYPES", {})

                if hasattr(function_config.functions, "root"):
                    resolved_functions = function_config.functions.root
                else:
                    resolved_functions = function_config.functions

                if resolved_functions:
                    present_functions = {}
                    for (
                        function_keyword,
                        resolved_function,
                    ) in resolved_functions.items():
                        if hasattr(resolved_function, "function_type"):
                            function_type = resolved_function.function_type
                            if function_type not in present_functions:
                                present_functions[function_type] = []
                            present_functions[function_type].append(function_keyword)

                    if allowed_types:
                        for function_type, keywords in present_functions.items():
                            if function_type not in allowed_types:
                                raise ValueError(
                                    f"Function '{function_name}' contains functions of invalid type: {function_type}. "
                                    f"Allowed types: {list(allowed_types.keys())}"
                                )

                            allowed_keywords = allowed_types[function_type]
                            if allowed_keywords:
                                invalid_keywords = set(keywords) - set(allowed_keywords)
                                if invalid_keywords:
                                    raise ValueError(
                                        f"Function '{function_name}' contains invalid keywords {invalid_keywords} for type {function_type}. "
                                        f"Allowed keywords: {allowed_keywords}"
                                    )

    def _collect_used_tools(self, function_names: List[str]) -> set:
        """
        Collect all tools used by the given functions, including nested tools and tools from referenced functions.

        Args:
            function_names: List of function names to start collection from

        Returns:
            Set of tool names that are used
        """
        used_tools = set()

        def collect_tools_from_function(func_name: str, visited_funcs: set):
            """Recursively collect tools from a function and its referenced functions."""
            if func_name in visited_funcs or func_name not in self.functions.root:
                return
            visited_funcs.add(func_name)

            func_config = self.functions.root[func_name]

            if func_config.tools:
                if hasattr(func_config.tools, "root"):
                    tools_dict = func_config.tools.root
                else:
                    tools_dict = func_config.tools

                if tools_dict:
                    for tool_role, tool_ref in tools_dict.items():
                        if isinstance(tool_ref, str):
                            used_tools.add(tool_ref)
                            collect_tools_from_tool(tool_ref, set())
                        elif hasattr(tool_ref, "tool_name"):
                            used_tools.add(tool_ref.tool_name)
                            collect_tools_from_tool(tool_ref.tool_name, set())

            if func_config.functions:
                if hasattr(func_config.functions, "root"):
                    funcs_dict = func_config.functions.root
                else:
                    funcs_dict = func_config.functions

                if funcs_dict:
                    for ref_func_name, ref_func_value in funcs_dict.items():
                        if isinstance(ref_func_value, str):
                            collect_tools_from_function(ref_func_value, visited_funcs)
                        elif hasattr(ref_func_value, "function_name"):
                            collect_tools_from_function(
                                ref_func_value.function_name, visited_funcs
                            )

        def collect_tools_from_tool(tool_name: str, visited_tools: set):
            """Recursively collect nested tools from a tool configuration."""
            if tool_name in visited_tools or tool_name not in self.tools.root:
                return
            visited_tools.add(tool_name)

            tool_config = self.tools.root[tool_name]
            if tool_config.tools:
                if hasattr(tool_config.tools, "root"):
                    tools_dict = tool_config.tools.root
                else:
                    tools_dict = tool_config.tools

                if tools_dict:
                    for nested_tool_role, nested_tool_ref in tools_dict.items():
                        if isinstance(nested_tool_ref, str):
                            used_tools.add(nested_tool_ref)
                            collect_tools_from_tool(nested_tool_ref, visited_tools)
                        elif hasattr(nested_tool_ref, "tool_name"):
                            used_tools.add(nested_tool_ref.tool_name)
                            collect_tools_from_tool(
                                nested_tool_ref.tool_name, visited_tools
                            )

        visited_functions = set()
        for func_name in function_names:
            collect_tools_from_function(func_name, visited_functions)

        return used_tools

    def _collect_used_functions(self, function_names: List[str]) -> set:
        """
        Collect all functions used by the given functions, including transitively referenced functions.

        Args:
            function_names: List of function names to start collection from

        Returns:
            Set of function names that are used
        """
        used_functions = set()

        def collect_functions_recursively(func_name: str, visited_funcs: set):
            """Recursively collect functions referenced by a function."""
            if func_name in visited_funcs or func_name not in self.functions.root:
                return
            visited_funcs.add(func_name)
            used_functions.add(func_name)

            func_config = self.functions.root[func_name]

            if func_config.functions:
                if hasattr(func_config.functions, "root"):
                    funcs_dict = func_config.functions.root
                else:
                    funcs_dict = func_config.functions

                if funcs_dict:
                    for ref_func_name, ref_func_value in funcs_dict.items():
                        if isinstance(ref_func_value, str):
                            collect_functions_recursively(ref_func_value, visited_funcs)
                        elif hasattr(ref_func_value, "function_name"):
                            collect_functions_recursively(
                                ref_func_value.function_name, visited_funcs
                            )

        visited_functions = set()
        for func_name in function_names:
            collect_functions_recursively(func_name, visited_functions)

        return used_functions

    def _filter_unused_components(self) -> "ContextManagerConfig":
        """
        Filter out functions not in context manager and tools not used by any remaining functions.
        Only applies filtering if context_manager.functions is non-empty.

        Returns:
            ContextManagerConfig: A new config object with only used functions and tools
        """

        if not self.context_manager.functions:
            logger.info(
                f"No functions specified in context_manager (functions={self.context_manager.functions}), skipping filtering"
            )
            logger.info(
                f"Preserving {len(self.functions.root)} functions and {len(self.tools.root)} tools"
            )
            return copy.deepcopy(self)

        context_functions = set(self.context_manager.functions)
        used_functions = self._collect_used_functions(list(context_functions))

        used_tools = self._collect_used_tools(list(used_functions))

        filtered_functions = {
            name: config
            for name, config in self.functions.root.items()
            if name in used_functions
        }

        filtered_tools = {
            name: config
            for name, config in self.tools.root.items()
            if name in used_tools
        }

        filtered_config = copy.deepcopy(self)
        from .tool_models import ToolsContainer
        from .function_models import FunctionsContainer

        filtered_config.functions = FunctionsContainer(root=filtered_functions)
        filtered_config.tools = ToolsContainer(root=filtered_tools)

        logger.info(
            f"Filtered config: kept {len(filtered_functions)} functions out of {len(self.functions.root)}"
        )
        logger.info(
            f"Filtered config: kept {len(filtered_tools)} tools out of {len(self.tools.root)}"
        )

        return filtered_config

    @model_validator(mode="after")
    def resolve_references(
        self, keep_all_components: bool = False
    ) -> "ContextManagerConfig":
        """
        Resolves string references to tools in both functions and tools configurations,
        replacing them with actual tool configurations.

        Args:
            keep_all_components: If True, keeps all tools and functions regardless of
                context_manager.functions settings. If False (default), filters unused components.

        Returns:
            ContextManagerConfig: A new config object with resolved references
        """
        import copy

        resolved_config = copy.deepcopy(self)

        for tool_name, tool_config in resolved_config.tools.root.items():
            if tool_config.tools:
                resolved_tools = {}
                internal_tools = tool_config.tools
                if isinstance(internal_tools, ToolRefsContainer):
                    internal_tools = internal_tools.root
                for internal_tool_role, internal_tool_ref in internal_tools.items():
                    if (
                        isinstance(internal_tool_ref, str)
                        and internal_tool_ref in resolved_config.tools.root
                    ):
                        resolved_tool = resolved_config.tools.root[internal_tool_ref]
                        resolved_tools[internal_tool_role] = resolved_tool
                    elif hasattr(internal_tool_ref, "tool_type"):
                        resolved_tools[internal_tool_role] = internal_tool_ref

                tool_config.tools = ToolConfigsContainer(root=resolved_tools)

        for _, function_config in resolved_config.functions.root.items():
            if function_config.tools:
                function_tools = function_config.tools
                if isinstance(function_tools, ToolRefsContainer):
                    function_tools = function_tools.root
                resolved_tools = {}
                for tool_role, tool_name in function_tools.items():
                    if isinstance(tool_name, str):
                        if tool_name in resolved_config.tools.root:
                            resolved_tool = resolved_config.tools.root[tool_name]
                            resolved_tools[tool_role] = resolved_tool
                    elif hasattr(tool_name, "tool_type"):
                        resolved_tools[tool_role] = tool_name
                function_config.tools = ToolConfigsContainer(root=resolved_tools)

            if function_config.functions:
                function_refs = function_config.functions
                function_refs_dict = function_refs
                if isinstance(function_refs, FunctionRefsContainer):
                    function_refs_dict = function_refs.root
                elif isinstance(function_refs, FunctionConfigsContainer):
                    function_refs_dict = function_refs.root

                resolved_functions = {}
                for ref_function_name, ref_function_value in function_refs_dict.items():
                    if isinstance(ref_function_value, str):
                        if ref_function_value in resolved_config.functions.root:
                            resolved_functions[ref_function_name] = (
                                resolved_config.functions.root[ref_function_value]
                            )
                    elif hasattr(ref_function_value, "function_name"):
                        resolved_functions[ref_function_name] = ref_function_value
                function_config.functions = FunctionConfigsContainer(
                    root=resolved_functions
                )

        resolved_config._validate_tool_types()

        resolved_config._validate_function_types()

        if keep_all_components:
            return resolved_config
        else:
            filtered_config = resolved_config._filter_unused_components()
            return filtered_config
