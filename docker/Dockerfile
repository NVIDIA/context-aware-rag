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

RUN groupadd -g 1000 appuser && \
    useradd -r -u 1000 -g appuser -m -d /home/appuser appuser

RUN python3 -m pip install pipx && python3 -m pipx install uv
ENV PATH=/root/.local/bin:$PATH

USER 1000
ENV HOME=/home/appuser
RUN python3 -m pip install --user pipx && python3 -m pipx ensurepath && pipx install uv
ENV PATH=/home/appuser/.local/bin:$PATH

USER root

WORKDIR /app

COPY pyproject.toml README.md LICENSE LICENSE.3rdparty /app/

COPY src/ /app/src/
COPY service/ /app/service/
COPY test[s]/ /app/tests/
COPY config/ /app/config/
COPY packages/ /app/packages/

RUN mkdir -p /app/logs/
RUN touch /app/logs/vss_ctx_rag.log

RUN uv build --all-packages \
    && uv pip install dist/*.whl --system

RUN ln -sf /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12 /usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/libcudart.so

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/

RUN chown -R 1000:1000 /app /home/appuser

ENV VSS_LOG_FILE=/app/logs/vss_ctx_rag.log
ENV HOME=/home/appuser

USER 1000
ENV PATH=/home/appuser/.local/bin:$PATH

CMD ["uvicorn", "service.service:app", "--host", "0.0.0.0", "--port", "8000"]
