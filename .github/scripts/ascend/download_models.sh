#!/bin/bash
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Provision models on the host-mounted /data volume. Images must not contain
# model weights; every platform test refers to the same host path convention.
set -euo pipefail

export FL_MODEL_BASE_PATH="${FL_MODEL_BASE_PATH:-/data/models}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

QWEN_ROOT="${FL_MODEL_BASE_PATH}/Qwen"
MODEL_IDS=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3.6-27B"
    "Qwen/Qwen3.6-35B-A3B"
)

mkdir -p "${QWEN_ROOT}"

for MODEL_ID in "${MODEL_IDS[@]}"; do
    MODEL_DIR="${QWEN_ROOT}/${MODEL_ID#Qwen/}"
    if [[ -f "${MODEL_DIR}/config.json" ]]; then
        echo "Model already available: ${MODEL_DIR}"
        continue
    fi

    export MODEL_ID MODEL_DIR
    echo "Downloading ${MODEL_ID} from ${HF_ENDPOINT} to ${MODEL_DIR}"
    python - <<'PY'
import os

from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["MODEL_ID"],
    local_dir=os.environ["MODEL_DIR"],
    endpoint=os.environ["HF_ENDPOINT"],
)
PY

    test -f "${MODEL_DIR}/config.json"
    echo "Model ready: ${MODEL_DIR}"
done
