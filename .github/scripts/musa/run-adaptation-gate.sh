#!/usr/bin/env bash
# Run in the already isolated four-device CI container; install nothing here.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
: "${MUSA_CI_PYTHON:=${PYTHON:-python}}"
command -v timeout >/dev/null
command -v "$MUSA_CI_PYTHON" >/dev/null
# Four bounded routes plus four preflight/cleanup costs and one observer summary.
# TERM lets the Python finally blocks clean only their owned process groups.
exec timeout --signal=TERM --kill-after=65s 29400s \
  "$MUSA_CI_PYTHON" -I -B "$script_dir/gate_entry.py" "$@"
