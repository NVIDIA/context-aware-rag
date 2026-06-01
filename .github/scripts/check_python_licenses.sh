#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Mirrors CI Job 9 (license-check-python). Scans only runtime deps that
# actually ship — dev-only tools (yamllint is GPL, others vary) are
# excluded via --no-default-groups. We use `uv pip install` (not
# `uv run pip install`) and `uv run --no-sync` so uv does not auto-resync
# the dev group back into the venv.

set -euo pipefail

repo_root=$(git rev-parse --show-toplevel 2>/dev/null || printf '%s\n' "${GITHUB_WORKSPACE:-$PWD}")
cd "$repo_root"

uv sync --frozen --no-default-groups --quiet
uv pip install --quiet pip-licenses

# docutils is multi-licensed (BSD/PSF/Public Domain/GPL); we use it under BSD/PSF.
# We do not import docutils directly in our code, we get it from these open source projects:
# - https://github.com/executablebooks/MyST-Parser/blob/master/LICENSE (MIT)
# - https://github.com/sphinx-doc/sphinx/blob/master/LICENSE.rst (BSD-2-Clause)
# - https://github.com/spatialaudio/nbsphinx/blob/master/LICENSE (MIT)
# - https://github.com/pydata/pydata-sphinx-theme/blob/main/LICENSE (BSD-3-Clause)
# - https://github.com/wasi-master/rich-rst/blob/main/LICENSE (MIT)
DISALLOWED=$(uv run --no-sync --quiet pip-licenses --format=csv \
  | grep -iE 'GPL|AGPL|SSPL|BUSL' \
  | grep -v 'LGPL' \
  | grep -v '^"docutils"' \
  || true)
if [ -n "$DISALLOWED" ]; then
  echo "ERROR: Found packages with disallowed licenses:"
  echo "$DISALLOWED"
  exit 1
fi
echo "OK: No disallowed licenses found."
