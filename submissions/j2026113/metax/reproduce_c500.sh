#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 5 && $# != 6 ]]; then
    printf '%s\n' 'Usage: bash reproduce_c500.sh DOCKER_SOCKET PINNED_IMAGE MODEL_DIR NEW_NVME_STUDY GPU_INDEX [performance|quality]' >&2
    exit 2
fi
root=$(realpath "$(dirname "$0")")
socket=$(realpath -e "$1")
image=$2
weights=$(realpath -e "$3")
study=$(realpath -m "$4")
gpu=$5
mode=${6:-performance}
[[ "$mode" == performance || "$mode" == quality ]]
[[ "$gpu" =~ ^[0-9]+$ ]]
[[ "$image" == sha256:86c334f18ab914fa3de5e0ddbf313311d5bfe4b52b4fc6011c91476987988171 ]]
test -S "$socket"
test -f "$weights/config.json"
test -f "$root/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py"
test -f "$root/FlagGems/flag_gems/__init__.py"
test -d "$(dirname "$study")"
test ! -e "$study"
(cd "$root" && sha256sum --quiet -c SOURCE_SHA256SUMS)
python3 - "$weights" "$root/evidence/model_file_hashes.json" <<'PY'
from hashlib import sha256
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
for name, expected in json.loads(Path(sys.argv[2]).read_bytes()).items():
    path = root / name
    digest = sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f'Model file hash mismatch: {path}')
PY
docker=(docker -H "unix://$socket")
test -z "$("${docker[@]}" ps -q)"
mx-smi -i "$gpu" --show-process | grep -q 'no process found'
mkdir "$study"
mkdir "$study"/{tmp,triton,inductor,cache,config,nvcc,mxlog,metax,flaggems}
args=(run -d --pull never --read-only --network none --ipc host --cpus 8 --memory 128g --device /dev/dri --device /dev/mxcd)
args+=(-v "$root/vllm-plugin-FL:/workspace/vllm-plugin-FL:ro" -v "$root/FlagGems:/gems:ro")
args+=(-v "$root/runtime/overlay:/screen:ro" -v "$root/runtime/arch:/arch:ro" -v "$root/runtime/kernels:/kernels:ro")
args+=(-v "$weights:/workspace/MiniCPM5-2B:ro" -v "$study:/study" -v "$study/tmp:/tmp")
for pair in cache:.cache config:.config nvcc:_nvcc mxlog:mxlog metax:.metax flaggems:.flaggems; do
    args+=(-v "$study/${pair%%:*}:/root/${pair#*:}")
done
args+=(-e CUDA_VISIBLE_DEVICES="$gpu" -e MACA_VISIBLE_DEVICES="$gpu" -e GEMS_VENDOR=metax -e VLLM_PLUGINS=fl)
args+=(-e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1)
args+=(-e PYTHONPATH=/workspace/vllm-plugin-FL:/gems:/screen:/kernels)
args+=(-e TRITON_CACHE_DIR=/study/triton -e TORCHINDUCTOR_CACHE_DIR=/study/inductor -e TMPDIR=/tmp)
args+=(-e XDG_CACHE_HOME=/root/.cache -e C500_MIXED_EVIDENCE=/study -e C500_FORMAL_MODE=candidate -e C500_MIXED_BACKEND=1)
args+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2)
id=$("${docker[@]}" "${args[@]}" "$image" /opt/conda/bin/vllm serve /workspace/MiniCPM5-2B \
    --port 9031 --served-model-name minicpm --gpu-memory-utilization 0.85 --max-model-len 131072)
printf '%s\n' "$id" > "$study/container.id"
finish() {
    "${docker[@]}" stop -t 25 "$id" >/dev/null 2>&1 || true
    "${docker[@]}" logs "$id" > "$study/server.log" 2>&1
    "${docker[@]}" inspect "$id" > "$study/container_state.json"
}
trap finish EXIT
for ((n=0; n<300; n++)); do
    [[ "$("${docker[@]}" inspect -f '{{.State.Running}}' "$id")" == true ]]
    if "${docker[@]}" exec "$id" /opt/conda/bin/python -c \
        "from urllib.request import urlopen; assert urlopen('http://127.0.0.1:9031/health', timeout=3).status==200" >/dev/null 2>&1; then
        break
    fi
    sleep 4
done
(( n < 300 ))
envpath=$("${docker[@]}" inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$id" | sed -n 's/^PATH=//p')
if [[ "$mode" == quality ]]; then
    "${docker[@]}" exec -w /study -e "PATH=/opt/conda/bin:$envpath" "$id" /opt/conda/bin/evalscope eval \
        --model minicpm --api-url http://127.0.0.1:9031/v1/chat/completions \
        --api-key EMPTY --eval-type openai_api --datasets math_500 \
        --dataset-args '{"math_500":{"dataset_id":"/workspace/evalscope-datasets/math_500","subset_list":["Level 3"]}}' \
        --eval-batch-size 8 --timeout 3600 \
        --generation-config '{"temperature":1.0,"top_p":0.95,"max_tokens":32768}' \
        --work-dir /study/eval --ignore-errors 2>&1 | tee "$study/evalscope.log"
    exit
fi
"${docker[@]}" exec -w /study -e "PATH=/opt/conda/bin:$envpath" "$id" /opt/conda/bin/python \
    /workspace/vllm-plugin-FL/benchmarks/benchmark_throughput_serve.py \
    --model /workspace/MiniCPM5-2B --served-model-name minicpm --port 9031 \
    --test-cases '[[4096,1024,64,256],[16384,1024,64,128]]' 2>&1 | tee "$study/benchmark.log"
