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
ifneq (,$(wildcard ../.env))
  include ../.env
  export
endif

IMAGE_NAME=vss_ctx_rag-$(USER)


build:
	docker build -t $(IMAGE_NAME) -f Dockerfile ..

start_ret:
	@if [ -f ../.env ]; then \
		docker run --env-file ../.env -it -p $(VSS_CTX_PORT_RET):8000 --env VIA_CTX_RAG_ENABLE_RET=true  $(IMAGE_NAME) $(if $(INTERACTIVE),/bin/bash); \
	else \
		docker run -it -p $(VSS_CTX_PORT_RET):8000 --env VIA_CTX_RAG_ENABLE_RET=true $(IMAGE_NAME) $(if $(INTERACTIVE),/bin/bash); \
	fi

start_in:
	@if [ -f ../.env ]; then \
		docker run --env-file ../.env -it -p $(VSS_CTX_PORT_IN):8000 --env VIA_CTX_RAG_ENABLE_RET=false  $(IMAGE_NAME) $(if $(INTERACTIVE),/bin/bash); \
	else \
		docker run -it -p $(VSS_CTX_PORT_IN):8000 --env VIA_CTX_RAG_ENABLE_RET=false $(IMAGE_NAME) $(if $(INTERACTIVE),/bin/bash); \
	fi

start_compose:
	docker compose -f deploy/compose.yaml up -d

stop_compose:
	docker compose -f deploy/compose.yaml down
