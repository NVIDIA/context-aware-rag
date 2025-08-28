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

import importlib
from typing import List, Optional, Union, Any, Dict, Type, Mapping, ClassVar
import inspect

from pydantic import BaseModel, Field, model_validator

from .tool_models import (
    Container,
    ToolRefsContainer,
    ToolConfigsContainer,
)


_FUNCTION_CONFIG_REGISTRY: Dict[str, Dict[str, str]] = {}
_FUNCTION_IMPLEMENTATION_REGISTRY: Dict[str, Dict[str, str]] = {}


def register_function_config(function_type: str):
    """
    Decorator to register a config class for a specific function type.

    Args:
        function_type: The type identifier for the function (e.g., 'chat', 'ingestion', 'notification')

    Usage:
        @register_function_config("chat")
        class ChatConfig(BaseModel):
            ...
    """

    calling_frame = inspect.currentframe().f_back
    calling_module = calling_frame.f_globals["__name__"]

    def decorator(config_class: Type[BaseModel]):
        module_path = config_class.__module__
        class_name = config_class.__name__

        _FUNCTION_CONFIG_REGISTRY[function_type] = {
            "module": module_path,
            "class": class_name,
            "calling_module": calling_module,
        }

        return config_class

    return decorator


def register_function(config: Type[BaseModel]):
    """
    Decorator to register a function implementation class for a specific function type.

    Args:
        config: The configuration class for this function type

    Usage:
        @register_function(config=ChatConfig)
        class ChatFunction(Function):
            ...
    """

    calling_frame = inspect.currentframe().f_back
    calling_module = calling_frame.f_globals["__name__"]

    def decorator(function_class):
        module_path = function_class.__module__
        class_name = function_class.__name__

        function_type = None
        for registered_type, registered_info in _FUNCTION_CONFIG_REGISTRY.items():
            if (
                registered_info["class"] == config.__name__
                and registered_info["module"] == config.__module__
            ):
                function_type = registered_type
                break

        if not function_type:
            raise ValueError(
                f"Config class {config.__name__} not found in function config registry. "
                f"Make sure to register it with @register_function_config first."
            )

        _FUNCTION_IMPLEMENTATION_REGISTRY[function_type] = {
            "module": module_path,
            "class": class_name,
            "calling_module": calling_module,
            "config_class": config.__name__,
            "config_module": config.__module__,
        }

        return function_class

    return decorator


class FunctionsContainer(Container[Dict[str, "FunctionConfig"]]):
    """Container for function configurations."""

    pass


class FunctionRefsContainer(Container[Dict[str, str]]):
    """Container for function references in function configs."""

    pass


class FunctionConfigsContainer(Container[Dict[str, "FunctionConfig"]]):
    """Container for resolved function configurations."""

    pass


class FunctionConfig(BaseModel):
    """
    Dynamic function configuration that imports the appropriate pydantic model
    based on the function type using decorator-based registration.
    """

    function_name: str = Field(
        description="The name/key of the function instance (e.g., 'summarization', 'retriever')"
    )
    function_type: str = Field(
        description="The type of function implementation (e.g., 'batch_summarization', 'vector_retrieval')"
    )
    params: Dict[str, Any] = Field(
        default_factory=dict, description="Function-specific parameters"
    )
    tools: Optional[Union[ToolRefsContainer, ToolConfigsContainer]] = None
    functions: Optional[Union[FunctionRefsContainer, FunctionConfigsContainer]] = None
    _uuid: Optional[str] = None

    ALLOWED_FUNCTION_TYPES: ClassVar[Dict[str, List[str]]] = {}
    FUNCTION_TYPE_CONSTRAINTS: ClassVar[Dict[str, Any]] = {}

    @model_validator(mode="before")
    @classmethod
    def validate_and_import_config(cls, values):
        """Dynamically import and validate the function config based on type"""
        if isinstance(values, dict):
            function_type = values.get("function_type") or values.get("type")
            function_name = values.get("function_name") or values.get("name")
            config_data = values.get("config")
            context_uuid = values.get("_uuid")

            if not config_data and function_type and not values.get("type"):
                if context_uuid and context_uuid != "default" and "params" in values:
                    params_data = values.get("params", {})
                    if isinstance(params_data, dict):
                        params_data = params_data.copy()
                        params_data["uuid"] = context_uuid
                        values = values.copy()
                        values["params"] = params_data
                return values

            if function_type and config_data:
                mapping = _FUNCTION_CONFIG_REGISTRY.get(function_type)
                if not mapping:
                    raise ValueError(
                        f"Unknown function type: {function_type}. "
                        f"Available types: {list(_FUNCTION_CONFIG_REGISTRY.keys())}"
                    )

                try:
                    module = importlib.import_module(mapping["module"])
                    config_class = getattr(module, mapping["class"])

                    if (
                        isinstance(config_data, dict)
                        and "tools" in config_data
                        and config_data["tools"] is not None
                    ):
                        config_data["tools"] = (
                            config_data["tools"]
                            if isinstance(config_data["tools"], Mapping)
                            else {"root": config_data["tools"]}
                        )

                    if context_uuid and context_uuid != "default":
                        if "params" not in config_data:
                            config_data["params"] = {}
                        config_data["params"]["uuid"] = context_uuid

                    validated_config = config_class(**config_data)

                    return {
                        "function_name": function_name,
                        "function_type": function_type,
                        "params": getattr(validated_config, "params", {}),
                        "tools": getattr(validated_config, "tools", None),
                        "functions": getattr(validated_config, "functions", None),
                    }

                except ImportError as e:
                    raise ValueError(
                        f"Could not import module {mapping['module']}: {e}"
                    )
                except AttributeError as e:
                    raise ValueError(
                        f"Could not find class {mapping['class']} in module {mapping['module']}: {e}"
                    )
                except Exception as e:
                    raise ValueError(
                        f"Invalid configuration for function type {function_type}: {e}"
                    )

        return values

    @model_validator(mode="before")
    @classmethod
    def transform_tools(cls, values):
        """Transform tools dict to ToolRefsContainer"""
        if isinstance(values, dict):
            if "tools" in values and values["tools"] is not None:
                if not isinstance(
                    values["tools"], (ToolRefsContainer, ToolConfigsContainer)
                ):
                    values["tools"] = ToolRefsContainer(root=values["tools"])

            if "functions" in values and values["functions"] is not None:
                if not isinstance(
                    values["functions"],
                    (FunctionRefsContainer, FunctionConfigsContainer),
                ):
                    values["functions"] = FunctionRefsContainer(
                        root=values["functions"]
                    )
        return values

    @classmethod
    def add_function_type(cls, function_type: str, module_path: str, class_name: str):
        """
        Add a new function type mapping for dynamic import.

        Args:
            function_type: The type identifier for the function
            module_path: The module path where the config class is located
            class_name: The name of the config class
        """
        _FUNCTION_CONFIG_REGISTRY[function_type] = {
            "module": module_path,
            "class": class_name,
        }

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get all registered function types."""
        return list(_FUNCTION_CONFIG_REGISTRY.keys())


class FunctionModel(BaseModel):
    """
    Base model for function configurations that provides common fields
    used across different function types.
    """

    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {}
    ALLOWED_FUNCTION_TYPES: ClassVar[Dict[str, List[str]]] = {}
    FUNCTION_TYPE_CONSTRAINTS: ClassVar[Dict[str, Any]] = {}

    tools: Optional[Union[ToolRefsContainer, ToolConfigsContainer]] = None
    functions: Optional[Union[FunctionRefsContainer, FunctionConfigsContainer]] = None

    @model_validator(mode="after")
    def validate_function_constraints(self) -> "FunctionModel":
        """Validate function type constraints"""
        if not self.functions:
            return self

        function_data = None
        if isinstance(self.functions, FunctionRefsContainer):
            function_data = (
                self.functions.root if isinstance(self.functions.root, dict) else {}
            )
        else:
            function_data = (
                self.functions.root if isinstance(self.functions.root, dict) else {}
            )

        function_names = list(function_data.keys())
        function_types = list(function_data.values())

        max_count = self.FUNCTION_TYPE_CONSTRAINTS.get("max_count")
        if max_count is not None and len(function_names) > max_count:
            raise ValueError(
                f"Number of functions ({len(function_names)}) exceeds maximum allowed ({max_count})"
            )

        mutually_exclusive_groups = self.FUNCTION_TYPE_CONSTRAINTS.get(
            "mutually_exclusive_groups", []
        )
        for group in mutually_exclusive_groups:
            found_function_types = [f for f in group if f in function_types]
            if len(found_function_types) > 1:
                raise ValueError(
                    f"Function types {found_function_types} are mutually exclusive and cannot be used together"
                )

        return self

    @model_validator(mode="before")
    @classmethod
    def transform_tools(cls, values):
        """Transform tools dict to ToolRefsContainer and functions dict to FunctionRefsContainer"""
        if isinstance(values, dict):
            if "tools" in values and values["tools"] is not None:
                if not isinstance(
                    values["tools"], (ToolRefsContainer, ToolConfigsContainer)
                ):
                    values["tools"] = (
                        values["tools"]
                        if isinstance(values["tools"], Mapping)
                        else {"root": values["tools"]}
                    )

            if "functions" in values and values["functions"] is not None:
                if not isinstance(
                    values["functions"],
                    (FunctionRefsContainer, FunctionConfigsContainer),
                ):
                    values["functions"] = FunctionRefsContainer(
                        root=values["functions"]
                    )
        return values


class FunctionConfigContainer(BaseModel):
    """Container for function-specific configuration."""

    ALLOWED_FUNCTION_TYPES: ClassVar[Dict[str, List[str]]] = {}
    FUNCTION_TYPE_CONSTRAINTS: ClassVar[Dict[str, Any]] = {}

    params: Dict[str, Any]
    tools: Optional[Union[ToolRefsContainer, ToolConfigsContainer]] = None
    functions: Optional[Union[FunctionRefsContainer, FunctionConfigsContainer]] = None

    @model_validator(mode="before")
    @classmethod
    def transform_tools(cls, values):
        """Transform tools dict to ToolRefsContainer and functions dict to FunctionRefsContainer"""
        if isinstance(values, dict):
            if "tools" in values and values["tools"] is not None:
                if not isinstance(
                    values["tools"], (ToolRefsContainer, ToolConfigsContainer)
                ):
                    values["tools"] = ToolRefsContainer(root=values["tools"])

            if "functions" in values and values["functions"] is not None:
                if not isinstance(
                    values["functions"],
                    (FunctionRefsContainer, FunctionConfigsContainer),
                ):
                    values["functions"] = FunctionRefsContainer(
                        root=values["functions"]
                    )
        return values
