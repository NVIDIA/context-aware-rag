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

from typing import Optional
from vss_ctx_rag.base import Tool
from vss_ctx_rag.utils.ctx_rag_logger import logger


class Function:
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
        self._params = {}

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

    # Update top_p, temperature, max_tokens for functions
    def update_llm(self, **params):
        """Updates LLM parameters (top_p, temperature, max_tokens) for the function.

        Args:
            **params: Dictionary containing LLM parameters to update
        """
        top_p = self.get_param("llm", "top_p")
        temperature = self.get_param("llm", "temperature")
        max_tokens = self.get_param("llm", "max_tokens")
        # If the params have changed at runtime, update the llm tool
        if params.get("llm", None) and top_p != params["llm"]["top_p"]:
            top_p = params["llm"]["top_p"]
        else:
            top_p = None
        if params.get("llm", None) and temperature != params["llm"]["temperature"]:
            temperature = params["llm"]["temperature"]
        else:
            temperature = None
        if params.get("llm", None) and max_tokens != params["llm"]["max_tokens"]:
            max_tokens = params["llm"]["max_tokens"]
        else:
            max_tokens = None
        if top_p is not None or temperature is not None or max_tokens is not None:
            if self.get_tool("llm") is not None:
                logger.debug(
                    f"Updating llm for function {self.name} with : top_p {top_p}, "
                    f"temperature {temperature}, max_tokens {max_tokens}"
                )
                self.get_tool("llm").update(
                    top_p=top_p, temperature=temperature, max_tokens=max_tokens
                )
            else:
                logger.debug(f"LLM tool not found for function{self.name}")
        else:
            logger.debug(f"LLM not updated for function{self.name}")

    def config(self, **params):
        self._params.update(params)
        return self

    # TODO: Add def update(self) this will be added later.
    # We have to implement stop() which will ensure that
    # updating the config values is threadsafe
    def update(self, **params):
        self.update_llm(**params)
        # Update all sub-functions
        for f in self._functions.values():
            f.update(**params)
        self.config(**params)
        self.done()

    # function finds the value of a param from a nested dictionary
    # param is provided in the form of keys to traverse the dictionary
    # eg : to Obtain the batch_size for summarization, func.get_param("params", "batch_size")
    def get_param(self, *keys, required: bool = True, params: dict = None):
        if len(keys) == 0 and params is None:  # if no key is provided
            logger.info(f"======= PARAMS {params}")
            raise ValueError("Empty param provided.")
        if params is None:  # Top level function call before recursion begins
            params = self._params  # save an object reference to the param store
        if isinstance(params, dict):
            if len(keys) == 0:
                raise ValueError("Required more param keyss.")
            if keys[0] not in params:
                if required:  # key not found but required
                    raise ValueError(f"Required param {keys[0]} not configured.")
                else:  # key not found
                    return None
            else:  # Call the same function for traversing the inner dictionary obtained by indexing
                return self.get_param(
                    *keys[1:], required=required, params=params[keys[0]]
                )
        else:  # Reached the value in the dictionary
            if len(keys) == 0:  # there are no more keys provided to traverse,
                return params
            if len(keys) > 0:  # there are more keys provided to traverse
                raise ValueError(f"Required param {keys} not configured.")

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
    def setup(self) -> dict:
        """Abstract method that must be implemented by subclasses.
        This method is where the business logic of function
        should be implemented which can use tools.

        Returns:
            dict: The setup configuration.

        Raises:
            RuntimeError: If not implemented by subclass.
        """
        raise RuntimeError("`setup` method not Implemented!")

    async def acall(self, state: dict) -> dict:
        """This method is where the business logic of function
        should be implemented which can use tools. Each class
        extending Function class should implement this.

        Args:
            state (dict): This is the dict of the state
        """
        raise RuntimeError("`call` method not Implemented!")

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
