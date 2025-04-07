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
FROM python:3.12-bookworm

RUN python3 -m pip install --user pipx \
    && python3 -m pipx ensurepath

ENV PATH=/root/.local/bin:$PATH

RUN pipx install uv

WORKDIR /app

COPY pyproject.toml README.md /app/

COPY src/ /app/src/
COPY service/ /app/service/
COPY tests/ /app/tests/
COPY config/ /app/config/

RUN mkdir /tmp/via-logs/

RUN uv build \
    && uv pip install dist/*.whl --system

CMD ["uvicorn", "service.service:app", "--host", "0.0.0.0", "--port", "8000"]
