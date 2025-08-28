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

"""Import all foundation RAG function modules to trigger registration decorators."""

from . import foundation_retrieval_func
from . import foundation_ingestion_func
import os

milvus_db_host = os.getenv("MILVUS_DB_HOST", "localhost")
milvus_db_port = os.getenv("MILVUS_DB_PORT", "19530")
app_vectorstore_url = f"http://{milvus_db_host}:{milvus_db_port}"
os.environ["APP_VECTORSTORE_URL"] = app_vectorstore_url
__all__ = ["foundation_retrieval_func", "foundation_ingestion_func"]
