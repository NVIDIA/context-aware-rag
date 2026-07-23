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

import json
from pathlib import Path


class GraphMetrics:
    def __init__(self):
        self.graph_create_tokens = 0
        self.graph_create_requests = 0
        self.graph_create_latency = 0
        self.graph_post_process_latency = 0

    def dump_json(self, file_name: str):
        """
        Dumps the object's attributes to a JSON file.

        Args:
            file_name (str, optional): The file name to write to.
        """
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "graph_create_tokens": self.graph_create_tokens,
            "graph_create_requests": self.graph_create_requests,
            "graph_create_latency": self.graph_create_latency,
            "graph_post_process_latency": self.graph_post_process_latency,
        }
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def reset(self):
        self.graph_create_tokens = 0
        self.graph_create_requests = 0
        self.graph_create_latency = 0
        self.graph_post_process_latency = 0


class SummaryMetrics:
    def __init__(self):
        self.summary_tokens = 0
        self.aggregation_tokens = 0
        self.summary_requests = 0
        self.summary_latency = 0
        self.aggregation_latency = 0
        # Dense-caption ES retrieval latency — populated ONLY when the DB-fetch
        # path actually runs (online / file+LVS_CAPTION_SOURCE=db). Left None so a
        # request that never hits the DB reports N/A downstream (via-engine keys its
        # metric off the presence of dense_captions_retrieval_latency), not a
        # misleading 0.
        self.dense_captions_retrieval_latency = None
        # LLM event-description merge — populated ONLY when enable_llm_merging=True.
        # Left None so a request with merging disabled reports N/A downstream
        # (via-engine keys its metric off the presence of llm_merge_latency), rather
        # than a misleading 0. llm_merge_calls follows the same None sentinel so it
        # can be interpreted independently of the latency field.
        self.llm_merge_latency = None
        self.llm_merge_tokens = None
        self.llm_merge_calls = None
        # LLM event-type inference
        self.event_type_infer_latency = 0
        self.event_type_infer_tokens = 0
        self.event_type_infer_calls = 0

    def dump_json(self, file_name: str):
        """
        Dumps the object's attributes to a JSON file.

        Args:
            file_name (str, optional): The file name to write to.
        """
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        with open(file_name, "w") as f:
            json.dump(self.dump_dict(), f, indent=4)

    def dump_dict(self):
        return {
            "summary_tokens": self.summary_tokens,
            "aggregation_tokens": self.aggregation_tokens,
            "summary_requests": self.summary_requests,
            "summary_latency": self.summary_latency,
            "aggregation_latency": self.aggregation_latency,
            "dense_captions_retrieval_latency": self.dense_captions_retrieval_latency,
            "llm_merge_latency": self.llm_merge_latency,
            "llm_merge_tokens": self.llm_merge_tokens,
            "llm_merge_calls": self.llm_merge_calls,
            "event_type_infer_latency": self.event_type_infer_latency,
            "event_type_infer_tokens": self.event_type_infer_tokens,
            "event_type_infer_calls": self.event_type_infer_calls,
        }

    def reset(self):
        self.summary_tokens = 0
        self.aggregation_tokens = 0
        self.summary_requests = 0
        self.summary_latency = 0
        self.aggregation_latency = 0
        self.dense_captions_retrieval_latency = None
        self.llm_merge_latency = None
        self.llm_merge_tokens = None
        self.llm_merge_calls = None
        self.event_type_infer_latency = 0
        self.event_type_infer_tokens = 0
        self.event_type_infer_calls = 0
