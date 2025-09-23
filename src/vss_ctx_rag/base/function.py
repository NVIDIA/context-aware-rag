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

"""function.py: File contains Function class"""

from abc import ABC, abstractmethod
from typing import Optional, Any

from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.models import FunctionConfig


class Function(ABC):
    """Base class for all functions in the RAG system.

    This class provides the core interface and functionality for all RAG operations.
    It handles tool management, parameter configuration, and function chaining.
    Each concrete function implementation should inherit from this class.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.is_setup: bool = False
        self._tools: dict[str, Tool] = {}
        self._functions: dict[str, Function] = {}
        self._config: FunctionConfig = None

    def add_tool(self, name: str, tool: Tool):
        """Adds a tool to the function

        Args:
            name (str): Tool name
            tool (Tool): tool object

        Raises:
            RuntimeError: Raises error if another tool
            with same name already present
        """
        # TODO(sl): Try Catch with custom exception
        if name in self._tools:
            raise RuntimeError(f"Tool {name} already added in {self.name} function")
        self._tools[name] = tool
        return self

    def remove_tool(self, name: str):
        """Removes a tool from the function"""
        if name in self._tools:
            del self._tools[name]
        else:
            raise RuntimeError(f"Tool {name} not found in {self.name} function")
        return self

    def add_function(self, name: str, function: "Function"):
        """Adds a function to the current function's sub-function container.

        Args:
            name (str): The name of the function to add.
            function (Function): The function object to be added.

        Raises:
            RuntimeError: If a function with the same name is already added.
        """
        if name in self._functions:
            raise RuntimeError(f"Function {name} already added in {self.name} function")
        self._functions[name] = function
        return self

    def get_tool(self, name):
        return self._tools[name] if name in self._tools else None

    def get_function(self, name: str) -> Optional["Function"]:
        """Retrieve the sub-function associated with the given name.

        Args:
            name (str): The name of the function to retrieve.

        Returns:
            Optional[Function]: The function object if it exists; otherwise, None.
        """
        return self._functions[name] if name in self._functions else None

    async def __call__(self, state: dict) -> dict:
        if not self.is_setup:
            raise RuntimeError("Function not setup. Call done()!")
        result = await self.acall(state)
        return result

    async def aprocess_doc_(
        self, doc: str, doc_i: int, doc_meta: Optional[dict] = None
    ):
        if not self.is_setup:
            raise RuntimeError("Function not setup. Call done()!")
        if doc_meta is None:
            doc_meta = {}
        result = await self.aprocess_doc(doc, doc_i, doc_meta)
        return result

    def config(self, config_obj: FunctionConfig = None, **kwargs):
        """Configure the function with a pydantic config object.

        Args:
            config_obj: Pydantic config object containing function configuration
            **kwargs: Legacy support for backward compatibility (will be ignored)
        """
        if config_obj:
            self._config = config_obj
        return self

    def get_param(self, key: str, default: Any = None):
        """Get a parameter from the config's params dictionary by traversing through keys.

        Args:
            key (str): The key to get the parameter for
            default (Any): Default value to return if parameter not found and not required

        Returns:
            Any: The parameter value if found, otherwise the default value

        Raises:
            AttributeError: If config is not properly initialized
        """
        if not self._config:
            return default
        return self._config.params.get(key, default)

    def update(self, config_obj: FunctionConfig):
        """Update the function configuration with a new config object.

        Args:
            config_obj: New pydantic config object to replace the current configuration
        """
        # Replace the config with the new one
        self._config = config_obj

        # Update all sub-functions with the same config
        for f in self._functions.values():
            f.update(config_obj)

        # Re-setup this function with the new config
        self.done()

        return self

    def done(self):
        self.setup()
        self.is_setup = True
        return self

    async def areset(self, state: dict):
        pass

    # TODO: change the function definition.
    # Pass **config, **tools, **functions instead of this
    # Or even better add _config, _tools and _functions in self and
    # expose a function like get_tool(), get_function(), get_config()
    @abstractmethod
    def setup(self) -> dict:
        """Abstract method that must be implemented by subclasses.
        This method is where the business logic of function
        should be implemented which can use tools.

        Returns:
            dict: The setup configuration.

        Raises:
            RuntimeError: If not implemented by subclass.
        """
        pass

    @abstractmethod
    async def acall(self, state: dict) -> dict:
        """This method is where the business logic of function
        should be implemented which can use tools. Each class
        extending Function class should implement this.

        Args:
            state (dict): This is the dict of the state
        """
        pass

    @abstractmethod
    async def aprocess_doc(self, doc: str, doc_i: int, doc_meta: dict):
        """This method is called every time a doc is added to
        the Context Manager. The function has the option to process the
        doc when the doc is added.

        Args:
            doc (str): document
            i (int): document index
            meta (dict): document metadata
        """
        pass
