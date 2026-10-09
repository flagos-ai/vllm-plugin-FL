#!/bin/bash
# Copyright 2026 FlagOS Contributors
set -euo pipefail
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${THEAD_CI_WORK_DIR:?Run the THead setup script first}"
: "${THEAD_CI_PYTHON:?Run the THead setup script first}"
test "${VLLM_PLUGINS:-}" = fl
test "${GEMS_VENDOR:-}" = thead
"$THEAD_CI_PYTHON" -I -B "$HERE/stack.py" check --compare "$THEAD_CI_WORK_DIR/vendor-before.json"
