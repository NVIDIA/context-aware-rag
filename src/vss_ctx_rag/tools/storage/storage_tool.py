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

"""storage_handler.py:"""

from vss_ctx_rag.base import Tool


class StorageTool(Tool):
    def __init__(self, name="storage_tool") -> None:
        super().__init__(name)

    def add_summary(self, summary, metadata):
        pass

    async def aadd_summary(self, summary, metadata):
        pass

    def add_summaries(self, batch_summary, batch_metadata):
        pass

    async def aadd_summaries(self, batch_summary, batch_metadata):
        pass

    def get_text_data(self, fields, filter):
        pass

    async def aget_text_data(self, fields, filter):
        pass

    def search(self, search_query):
        pass
