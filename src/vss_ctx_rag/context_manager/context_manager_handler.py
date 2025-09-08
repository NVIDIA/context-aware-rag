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

"""Context manager handler implementation.

This module handles managing the input to LLM by calling the handlers of all
the tools it has access to.
"""

import asyncio
import copy
import json
import traceback
from typing import Dict, Optional, Union

from vss_ctx_rag.base.function import Function
from vss_ctx_rag.base.tool import Tool
from vss_ctx_rag.utils.globals import DEFAULT_CONCURRENT_DOC_PROCESSING_LIMIT
from vss_ctx_rag.utils.ctx_rag_logger import Metrics, logger
from vss_ctx_rag.tools.storage.storage_tool import StorageTool
from vss_ctx_rag.models.context_manager_models import (
    ContextManagerConfig,
)
from vss_ctx_rag.functions.function_factory import FunctionFactory
from vss_ctx_rag.tools.tool_factory import ToolFactory
from pydantic import ValidationError


class ContextManagerHandler:
    """Main controller for RAG system operations.

    This class orchestrates the flow of operations in the RAG system by:
    1. Managing function registration and execution
    2. Handling configuration updates
    3. Coordinating between different handlers (chat, summarization, etc.)
    4. Managing the lifecycle of LLM instances and database connections
    """

    # TODO: Is last separately for live stream case
    # TODO: How do we customize prompts from VIA-UI
    # TODO: Make the functions a list
    # TODO: Unit test for blocking function call when calling another function. Does add_doc block too?
    def __init__(
        self,
        config: Dict,
        process_index: int,
    ) -> None:
        """Initialize the context manager handler.

        Args:
            config: Configuration dictionary containing system settings
            process_index: Unique identifier for this handler instance
        """
        logger.info(f"Initializing Context Manager Handler no.: {process_index}")

        self._functions: dict[str, Function] = {}
        self.tools: dict[str, Dict[str, Tool]] = {}
        self.auto_indexing: Optional[bool] = None
        self.curr_doc_index: int = -1
        self.storage_tools: Dict[str, StorageTool] = {}
        self._process_index = process_index
        self.uuid = config.get("context_manager", {}).get("uuid", None)
        self._doc_processing_semaphore = asyncio.Semaphore(
            DEFAULT_CONCURRENT_DOC_PROCESSING_LIMIT
        )

        self.configure(config)

    def configure(
        self,
        config: Union[Dict, ContextManagerConfig],
    ):
        """Validate and store configuration, then invoke asynchronous setup."""

        asyncio.run(self.aconfigure(config))

    async def aconfigure(self, config: ContextManagerConfig):
        """Initialize system components based on configuration.

        Sets up function handlers based on the provided configuration using
        the FunctionFactory.

        Args:
            config: System configuration dictionary
        """
        try:
            for function_name in config["functions"]:
                await self.aremove_function(function_name)

            config = ContextManagerConfig(**config)

            self.config = config.resolve_references()

            logger.debug(
                f"Configuring init for {self._process_index} with config: {config}"
            )

            await self.aconfigure_tools(self.config)

            await self.aconfigure_functions(self.config)
        except ValidationError as e:
            logger.error(f"Error in configuring context manager handler: {e}")
            logger.error(f"Error in configuring context manager handler: {e.errors()}")
            raise

        except Exception as e:
            logger.error(f"Error in configuring context manager handler: {e}")
            logger.error(traceback.format_exc())

    async def aconfigure_tools(self, config: ContextManagerConfig):
        """Configure tools using the ToolFactory."""
        logger.info("Configuring tools using ToolFactory")
        self.tools.update(ToolFactory.create_all_tools(config, self.tools))
        logger.info(f"TOOLS: {self.tools}")

    async def aconfigure_functions(
        self,
        config: ContextManagerConfig,
    ):
        """
        Configure functions using the FunctionFactory.

        This method uses the FunctionFactory to create functions based on the
        configuration's function list, replacing the previous hardcoded logic.

        Args:
            config: Complete configuration provided by the caller.
        """
        logger.debug("Configuring functions using FunctionFactory")

        function_factory = FunctionFactory(config, self.tools)

        try:
            new_functions = function_factory.create_functions()

            functions_to_add = config.context_manager.functions

            if not functions_to_add:
                functions_to_add = list(new_functions.keys())
                logger.info(
                    "No functions specified in context_manager.functions, adding all functions to context manager"
                )

            if len(functions_to_add) != len(new_functions):
                logger.warning(
                    f"Number of functions to add ({len(functions_to_add)}) does not match number of functions created ({len(new_functions)})"
                )
                raise ValueError(
                    f"Number of functions to add ({len(functions_to_add)}) does not match number of functions created ({len(new_functions)})"
                )

            missing_functions = set(functions_to_add) - set(new_functions.keys())
            if missing_functions:
                logger.warning(
                    f"Functions specified in context_manager.functions not found in created functions: {missing_functions}"
                )
                raise ValueError(
                    f"Functions specified in context_manager.functions not found in created functions: {missing_functions}"
                )

            for function_name, function_instance in new_functions.items():
                if function_name in functions_to_add:
                    self.add_function(function_instance)
                    logger.info(
                        f"Successfully configured and added function to context manager: {function_name}"
                    )
                else:
                    logger.info(
                        f"Function created but not added to context manager: {function_name}"
                    )

        except Exception as e:
            logger.error(f"Error creating functions with FunctionFactory: {e}")
            logger.error(traceback.format_exc())
            raise

    def add_function(self, f: Function):
        assert f.name not in self._functions, str(self._functions)
        logger.debug(f"Adding function: {f.name}")
        self._functions[f.name] = f
        return self

    def get_function(self, fname):
        return self._functions[fname] if fname in self._functions else None

    async def aremove_function(self, fname: str):
        if fname in self._functions:
            logger.debug(
                f"Removing function {fname} from Context Manager index: {self._process_index} with UUID: {self.uuid}"
            )
            await self._functions[fname].areset({"expr": "pk > 0", "uuid": self.uuid})
            self._functions[fname]._functions.clear()
            del self._functions[fname]

    def update(self, config):
        config_to_print = copy.deepcopy(config)
        if config_to_print.get("api_key"):
            del config_to_print["api_key"]
        logger.info(
            f"Updating context manager with config:\n{json.dumps(config_to_print, indent=2)}"
        )
        try:
            for fn, fn_config in config.items():
                if fn in self._functions:
                    self._functions[fn].update(**fn_config)
        except Exception as e:
            logger.error(
                "Overriding failed for config %s with error: %s", config_to_print, e
            )
            logger.error(traceback.format_exc())
        del config_to_print

    async def aprocess_doc(
        self, doc: str, doc_i: Optional[int] = None, doc_meta: Optional[dict] = None
    ):
        """Process a document.

        Args:
            doc (str): The document content.
            doc_i (Optional[int]): Document index. Required if auto_indexing is False.
            doc_meta (Optional[dict]): Document metadata.

        Returns:
            List of results from all functions processing the document.
        """
        # Handle document indexing
        if self.curr_doc_index < 0:
            if doc_i is None:
                self.auto_indexing = True
                self.curr_doc_index = 0
            else:
                self.auto_indexing = False
                self.curr_doc_index = doc_i

        if self.auto_indexing:
            doc_i = self.curr_doc_index
            self.curr_doc_index += 1
        elif doc_i is None:
            logger.error("Param doc_i missing. Auto-indexing is disabled.")

        # Process document through all functions with semaphore control
        async with self._doc_processing_semaphore:
            tasks = []

            async def timed_function_call(func, doc, doc_i, doc_meta):
                with Metrics(f"context_manager/aprocess_doc/{func.name}", "yellow"):
                    return await func.aprocess_doc_(doc, doc_i, doc_meta)

            with Metrics(
                "context_manager/aprocess_doc/total",
                "green",
                span_kind=Metrics.SPAN_KIND["CHAIN"],
            ) as tm:
                tm.input({"doc": doc, "doc_i": doc_i, "doc_meta": doc_meta})

                for _, f in self._functions.items():
                    tasks.append(
                        asyncio.create_task(
                            timed_function_call(f, doc, doc_i, doc_meta), name=f.name
                        )
                    )
                output = await asyncio.gather(*tasks)
                return output

    async def call(self, state):
        """Execute registered functions with the given state.

        Args:
            state: Dictionary containing function names and their parameters
        Returns:
            Dictionary containing results from all function executions
        """
        results = {}
        with Metrics(
            "context_manager/call", "green", span_kind=Metrics.SPAN_KIND["CHAIN"]
        ) as tm:
            tm.input(state)

            tasks = []
            task_results = []
            for func, call_params in state.items():
                tasks.append(
                    asyncio.create_task(self._functions[func](call_params), name=func)
                )
            task_results = await asyncio.gather(*tasks)
            for index, func in enumerate(state):
                results[func] = task_results[index]

            tm.output(results)
        return results

    async def areset(self, state):
        """Reset the context manager and all registered functions.

        Args:
            state: Reset parameters for each function
        Returns:
            Results from resetting all functions
        """
        self.curr_doc_index = -1
        self.auto_indexing = False
        tasks = []

        for func, reset_params in state.items():
            if func in self._functions:
                tasks.append(
                    asyncio.create_task(
                        self._functions[func].areset(reset_params), name=func
                    )
                )
            else:
                logger.debug("Function %s not found. Not resetting.", func)
        return await asyncio.gather(*tasks)
