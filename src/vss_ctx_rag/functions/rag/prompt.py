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

IMAGE_QA_PROMPT = """
You are an AI assistant that answers questions about images.
Analyze the provided images carefully and answer the question based on what you can see.
Pay attention to the timestamp metadata provided for each image as it may be relevant to the question.
If you cannot determine the answer from the images, say so clearly.
Be specific and descriptive in your answers.
"""
