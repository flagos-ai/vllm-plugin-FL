#!/bin/bash
set -euo pipefail
if [[ $# != 3 ]]; then
    echo "Usage: bash run_smoke.sh DOCKER_SOCKET MODEL_DIRECTORY NEW_NVME_STUDY" >&2
    exit 2
fi
pkg=$(realpath "$(dirname "$0")")
socket=$(realpath -e "$1")
weights=$(realpath -e "$2")
study=$(realpath -m "$3")
test -S "$socket"
test -f "$weights/config.json"
test "$(findmnt -n -o TARGET -T "$(dirname "$study")")" = /nvme2
d=(docker --host "unix://$socket")
test -z "$("${d[@]}" ps --format '{{.Names}}')"
mx-smi -i 1 --show-process | grep -q 'no process found'
(cd "$pkg" && sha256sum --quiet -c SHA256SUMS)
mkdir "$study"
mkdir "$study"/{tmp,triton,inductor,cache,config,nvcc,mxlog,metax,flaggems}
args=(run -d --pull never --read-only --network none --ipc host --cpus 8 --device /dev/dri --device /dev/mxcd)
for name in plugin overlay arch diag kernels gems; do args+=(-v "$pkg/$name:/$name:ro"); done
args+=(-v "$study:/study" -v "$weights:/workspace/MiniCPM5-2B:ro")
args+=(-v "$study/tmp:/tmp" -v "$study/cache:/root/.cache" -v "$study/config:/root/.config")
args+=(-v "$study/nvcc:/root/_nvcc" -v "$study/mxlog:/root/mxlog" -v "$study/metax:/root/.metax" -v "$study/flaggems:/root/.flaggems")
args+=(-e CUDA_VISIBLE_DEVICES=1 -e MACA_VISIBLE_DEVICES=1 -e GEMS_VENDOR=metax -e VLLM_PLUGINS=fl)
args+=(-e PYTHONHASHSEED=0 -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONPATH=/plugin:/gems:/overlay:/kernels:/diag)
args+=(-e TRITON_CACHE_DIR=/study/triton -e TORCHINDUCTOR_CACHE_DIR=/study/inductor -e TMPDIR=/tmp)
args+=(-e C500_MIXED_EVIDENCE=/study -e C500_W8A8_ROUTE_EVIDENCE=/study/w8a8_dispatch.jsonl)
args+=(-e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e OMP_NUM_THREADS=2)
image=sha256:86c334f18ab914fa3de5e0ddbf313311d5bfe4b52b4fc6011c91476987988171
id=$("${d[@]}" "${args[@]}" "$image" timeout -k 10 420 /opt/conda/bin/python -Bu /diag/probe_vendor_model_r1119.py)
printf '%s\n' "$id" | tee "$study/container.id"
