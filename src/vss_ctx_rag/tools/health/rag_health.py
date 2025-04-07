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

    def dump_json(self, file_name: str):
        """
        Dumps the object's attributes to a JSON file.

        Args:
            file_name (str, optional): The file name to write to.
        """
        data = {
            "summary_tokens": self.summary_tokens,
            "aggregation_tokens": self.aggregation_tokens,
            "summary_requests": self.summary_requests,
            "summary_latency": self.summary_latency,
            "aggregation_latency": self.aggregation_latency,
        }
        with open(file_name, "w") as f:
            json.dump(data, f, indent=4)

    def reset(self):
        self.summary_tokens = 0
        self.aggregation_tokens = 0
        self.summary_requests = 0
        self.summary_latency = 0
        self.aggregation_latency = 0
