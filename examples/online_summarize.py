# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Summarize raw events already stored in the database for one or more UUIDs
using vlm_structured_online_summarization (no document ingestion needed).

This assumes raw_events documents were previously persisted to the
configured database backend (e.g. via elasticpull, the VLM pipeline,
or any other ingest path).

Required environment variables
──────────────────────────────
  # vss-ctx-rag config (resolved via config/config.yaml)
  LVS_LLM_MODEL_NAME, LVS_LLM_BASE_URL, NVIDIA_API_KEY, ...
  LVS_DATABASE_BACKEND   (default: elasticsearch_db)

  # Database connectivity (depends on backend)
  ES_HOST, ES_PORT       (for elasticsearch)
  MILVUS_DB_HOST, MILVUS_DB_GRPC_PORT  (for milvus)

Required arguments
──────────────────
  --uuid                 One or more UUIDs of the streams/documents whose
                         raw events should be fetched and summarized

Optional
────────
  --start-time           Only include events whose end_time >= this value (seconds)
  --end-time             Only include events whose start_time <= this value (seconds)
  CONFIG_PATH            Path to vss-ctx-rag config YAML
                         (default config/config.yaml)
"""

import argparse
import json
import os
import sys
from copy import deepcopy

from pyaml_env import parse_config

from vss_ctx_rag.context_manager.context_manager import ContextManager
from vss_ctx_rag.utils.ctx_rag_logger import logger


SUMM_FN = "vlm_structured_summarization_online"


def run(args):
    # ── 1. Load config ──────────────────────────────────────────────────
    config = deepcopy(parse_config(args.config_path))

    # Wire up the online summarization function with the target UUID(s)
    params = {
        "uuids": args.uuids,
        "time_overlap_threshold": 0.1,
        "time_adjacent_threshold": 5,
        "max_events_per_batch": 50,
        "enable_llm_merging": True,
    }
    if args.start_time is not None:
        params["start_time"] = args.start_time
    if args.end_time is not None:
        params["end_time"] = args.end_time

    config["functions"][SUMM_FN] = {
        "type": SUMM_FN,
        "params": params,
        "tools": config["functions"]["summarization"]["tools"],
    }

    config["context_manager"]["functions"] = [SUMM_FN]

    # ── 2. Create context manager ────────────────────────────────────────
    ctx_mgr = ContextManager(config=deepcopy(config))
    logger.info(f"Context manager initialised - will summarize uuids {args.uuids}")

    # ── 3. Run summarization (fetches raw_events from DB by UUID(s)) ─────
    logger.info("Running online summarization …")
    result = ctx_mgr.call({SUMM_FN: {"uuids": args.uuids}})

    if "error" in result.get(SUMM_FN, {}):
        logger.error(f"Summarization error: {result[SUMM_FN]}")
        sys.exit(1)

    summary_raw = result[SUMM_FN].get("result", "")
    logger.info(f"Summarization complete. Raw length: {len(summary_raw)} chars")

    try:
        summary_data = json.loads(summary_raw)
        print(json.dumps(summary_data, indent=2))
    except json.JSONDecodeError:
        print(summary_raw)

    # ── Cleanup ──────────────────────────────────────────────────────────
    ctx_mgr.process.stop()
    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize raw events in the database for one or more UUIDs"
    )
    parser.add_argument(
        "--uuid",
        nargs="+",
        dest="uuids",
        default=None,
        help="One or more UUIDs whose raw events should be summarized "
        "(default: $ES_UUID)",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Only include events whose end_time >= this value (seconds)",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Only include events whose start_time <= this value (seconds)",
    )
    parser.add_argument(
        "--config-path",
        default=os.getenv("CONFIG_PATH", "config/config.yaml"),
        help="Path to vss-ctx-rag config YAML (default: config/config.yaml)",
    )

    args = parser.parse_args()
    if not args.uuids:
        default_uuid = os.getenv("ES_UUID", None)
        if default_uuid:
            args.uuids = [default_uuid]
        else:
            parser.error("--uuid is required (or set ES_UUID env var)")
    run(args)


if __name__ == "__main__":
    main()
