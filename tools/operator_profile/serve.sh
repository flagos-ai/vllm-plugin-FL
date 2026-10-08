#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "usage: $0 <model-path> <served-model-name> <run-dir> [vllm args...]" >&2
  exit 2
fi

MODEL_PATH=$1
SERVED_MODEL_NAME=$2
RUN_DIR=$3
shift 3

mkdir -p "$RUN_DIR"
RUN_DIR=$(cd "$RUN_DIR" && pwd)
PROFILE_DIR="$RUN_DIR/profile"
FLAGGEMS_OPLIST=${FLAGGEMS_ENABLE_OPLIST_PATH:-$RUN_DIR/flaggems_enable_oplist.txt}
mkdir -p "$PROFILE_DIR"
mkdir -p "$(dirname "$FLAGGEMS_OPLIST")"
if find "$PROFILE_DIR" -type f -name '*.pt.trace.json*' -print -quit |
  grep -q .; then
  echo "profile directory already contains a trace: $PROFILE_DIR" >&2
  exit 1
fi

# The plugin creates this file during startup. Removing an earlier run's file
# makes copied evidence unambiguously belong to this server and keeps a native
# baseline from inheriting plugin evidence.
rm -f "$FLAGGEMS_OPLIST"
export FLAGGEMS_ENABLE_OPLIST_PATH="$FLAGGEMS_OPLIST"

printf -v PROFILER_CONFIG \
  '{"profiler":"torch","torch_profiler_dir":"%s","torch_profiler_record_shapes":true,"torch_profiler_with_stack":false,"torch_profiler_dump_cuda_time_total":false,"torch_profiler_with_memory":false,"ignore_frontend":true}' \
  "$PROFILE_DIR"

printf 'VLLM_PLUGINS=%q\n' "${VLLM_PLUGINS-}"
printf 'run_dir=%q\n' "$RUN_DIR"

exec vllm serve "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --profiler-config "$PROFILER_CONFIG" \
  "$@"
