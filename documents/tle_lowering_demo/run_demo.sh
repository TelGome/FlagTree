#!/usr/bin/env bash
# Copyright 2026-     FlagOS Contributors
# SPDX-License-Identifier: MIT
#
# Run the TLE distributed lowering demo.
#
#   bash run_demo.sh            # use the python on PATH
#   PYTHON=/opt/venv/bin/python bash run_demo.sh
#
# Requires: Ascend backend build with TRITON_BUILD_TD=1, CANN env sourced,
# and at least one visible NPU (compile only, no multi-rank launch).

set -euo pipefail

cd "$(dirname "$0")"
PYTHON=${PYTHON:-python}

export TRITON_ALWAYS_COMPILE=1
export TRITON_DEBUG=1

echo "### step 1: IR-level lowering (no NPU execution)"
"$PYTHON" ir_demo.py || echo "IR demo reported un-converted ops (expected before the core_type fix)"

echo
echo "### step 2: end-to-end compile (succeeds once the core_type fix is in the frontend)"
"$PYTHON" compile_demo.py || echo "compile failed: missing distributed.core_type (pre-fix build)"

echo
echo "### step 3: end-to-end compile with --fix-core-type (no-op; kept for CLI compatibility)"
"$PYTHON" compile_demo.py --fix-core-type
