#!/bin/bash
set -euo pipefail
root=/nvme2/minimax/experiments/flagos_s2_recovery_20261113
stage="$root/w8a8-submission-stage-r1239"
src="$root/vendor-mixed-input-r1118"
study="$root/w8a8-submission-smoke-r1240"
kernels="$root/w8a8-submission-input-r1239"
arch=/nvme2/minimax/experiments/flagos_s2_cache_arch_gate_20261026
weights=/nvme2/minimax/experiments/flagos_s2_minicpm5_w8a8_20261019/models/MiniCPM5-2B-w8a8-r1
d=(docker --host "unix://$root/docker.sock")
test "$(findmnt -n -o TARGET /nvme2)" = /nvme2
test -z "$("${d[@]}" ps --format '{{.Names}}')"
mx-smi -i 1 --show-process | grep -q 'no process found'
mkdir "$study"
mkdir "$study"/{tmp,triton,inductor,cache,config,nvcc,mxlog,metax,flaggems}
args=(run -d --pull never --read-only --network none --name flagos_w8a8_submission_smoke_r1240 --ipc host --cpus 8)
args+=(--device /dev/dri --device /dev/mxcd)
args+=(-v "$study:/study" -v "$src:/diag:ro" -v "$arch:/arch:ro" -v "$kernels:/kernels:ro")
args+=(-v "$stage/plugin:/plugin:ro" -v "$stage/overlay:/overlay:ro")
args+=(-v "$root/gems_route_stage_r862/gems:/gems:ro" -v "$weights:/workspace/MiniCPM5-2B:ro")
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
