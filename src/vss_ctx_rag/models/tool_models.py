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
import inspect
from typing import (
    List,
    Optional,
    Union,
    Any,
    Dict,
    Type,
    TypeVar,
    Generic,
    ClassVar,
)

from pydantic import BaseModel, Field, model_validator, RootModel


_TOOL_CONFIG_REGISTRY: Dict[str, Dict[str, str]] = {}
_TOOL_IMPLEMENTATION_REGISTRY: Dict[str, Dict[str, str]] = {}

T = TypeVar("T")


def register_tool_config(tool_type: str):
    """
    Decorator to register a config class for a specific tool type.

    Args:
        tool_type: The type identifier for the tool (e.g., 'neo4j', 'milvus', 'llm', 'embedding', 'reranker')

    Usage:
        @register_tool_config("neo4j")
        class Neo4jDBConfig(BaseModel):
            ...
    """

    calling_frame = inspect.currentframe().f_back
    calling_module = calling_frame.f_globals["__name__"]

    def decorator(config_class: Type[BaseModel]):
        module_path = config_class.__module__
        class_name = config_class.__name__

        _TOOL_CONFIG_REGISTRY[tool_type] = {
            "module": module_path,
            "class": class_name,
            "calling_module": calling_module,
        }

        return config_class

    return decorator


def register_tool(config: Type[BaseModel]):
    """
    Decorator to register a tool implementation class for a specific tool type.

    Args:
        config: The configuration class for this tool type

    Usage:
        @register_tool(config=Neo4jDBConfig)
        class Neo4jGraphDB(StorageTool):
            ...
    """

    calling_frame = inspect.currentframe().f_back
    calling_module = calling_frame.f_globals["__name__"]

    def decorator(tool_class):
        module_path = tool_class.__module__
        class_name = tool_class.__name__

        tool_type = None
        for registered_type, registered_info in _TOOL_CONFIG_REGISTRY.items():
            if (
                registered_info["class"] == config.__name__
                and registered_info["module"] == config.__module__
            ):
                tool_type = registered_type
                break

        if not tool_type:
            raise ValueError(
                f"Config class {config.__name__} not found in tool config registry. "
                f"Make sure to register it with @register_tool_config first."
            )

        _TOOL_IMPLEMENTATION_REGISTRY[tool_type] = {
            "module": module_path,
            "class": class_name,
            "calling_module": calling_module,
            "config_class": config.__name__,
            "config_module": config.__module__,
        }

        return tool_class

    return decorator


class Container(RootModel[T], Generic[T]):
    """Generic container for attribute-style access to dictionary data."""

    def __getattr__(self, name: str) -> Any:
        """Enable attribute access."""
        try:
            return self.root[name]
        except KeyError:
            raise AttributeError(f"'{name}' not found")


class ToolsContainer(Container[Dict[str, "ToolsConfig"]]):
    """Container for tool configurations - flat structure where each tool has a type field."""

    pass


class ToolRefsContainer(Container[Dict[str, str]]):
    """Container for tool references in DBConfig and function configs."""

    pass


class ToolConfigsContainer(Container[Dict[str, "ToolsConfig"]]):
    """Container for resolved tool configurations."""

    pass


class ToolBaseModel(BaseModel):
    ALLOWED_TOOL_TYPES: ClassVar[Dict[str, List[str]]] = {}
    tools: Optional[Union[ToolRefsContainer, ToolConfigsContainer]] = None


class ToolsConfig(BaseModel):
    """Configuration for all tools including databases, LLMs, embeddings, and rerankers."""

    tool_type: str = Field(
        description="The type of tool (e.g., 'db', 'llm', 'embedding', 'reranker')"
    )
    tool_name: str = Field(description="The name/instance identifier of this tool")
    params: Any = Field(description="The tool-specific parameters")
    tools: Optional[Union[ToolRefsContainer, ToolConfigsContainer]] = None
    ALLOWED_TOOL_TYPES: Optional[Dict[str, List[str]]] = Field(
        default=None,
        description="Mapping of allowed tool types to their allowed keywords",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_and_import_config(cls, values):
        """Dynamically import and validate the tool config based on type"""
        if isinstance(values, dict):
            tool_type = values.get("tool_type")
            params_data = values.get("params")
            tools_data = values.get("tools")
            context_uuid = values.get("_uuid")

            if tool_type and params_data is not None:
                mapping = _TOOL_CONFIG_REGISTRY.get(tool_type)
                if not mapping:
                    raise ValueError(
                        f"Unknown tool type: {tool_type}. "
                        f"Available types: {list(_TOOL_CONFIG_REGISTRY.keys())}"
                    )

                try:
                    module = importlib.import_module(mapping["module"])
                    config_class = getattr(module, mapping["class"])

                    if context_uuid and context_uuid != "default":
                        params_data = params_data.copy()
                        has_collection_name = (
                            hasattr(config_class, "__fields__")
                            and "collection_name" in config_class.__fields__
                        ) or (
                            hasattr(config_class, "model_fields")
                            and "collection_name" in config_class.model_fields
                        )
                        if has_collection_name:
                            new_collection_name = f"default_{context_uuid}"
                            params_data["collection_name"] = new_collection_name

                    validated_params = config_class(**params_data)

                    allowed_types = getattr(config_class, "ALLOWED_TOOL_TYPES", {})

                    validated_tools = None
                    if tools_data is not None:
                        if not isinstance(
                            tools_data, (ToolRefsContainer, ToolConfigsContainer)
                        ):
                            validated_tools = ToolRefsContainer(root=tools_data)
                        else:
                            validated_tools = tools_data

                    return {
                        "tool_type": tool_type,
                        "tool_name": values.get("tool_name"),
                        "params": validated_params,
                        "tools": validated_tools,
                        "ALLOWED_TOOL_TYPES": allowed_types,
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
                        f"Invalid configuration for tool type {tool_type}: {e}"
                    )

        return values

    @classmethod
    def add_tool_type(cls, tool_type: str, module_path: str, class_name: str):
        """
        Add a new tool type mapping for dynamic import.

        Args:
            tool_type: The type identifier for the tool
            module_path: The module path where the config class is located
            class_name: The name of the config class
        """
        _TOOL_CONFIG_REGISTRY[tool_type] = {
            "module": module_path,
            "class": class_name,
        }

    @classmethod
    def get_registered_types(cls) -> List[str]:
        """Get all registered tool types."""
        return list(_TOOL_CONFIG_REGISTRY.keys())
