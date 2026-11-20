#!/bin/bash
set -euo pipefail

root=/data/pr726-e8-crosshost/native-vendorpg-sdk-20261120
state=/data/pr726-e8-crosshost
image=harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-kunlunxin-pr726-runtime-7c7329e@sha256:93caedbfc36d665c2e7d9f70a38bbb2501a860c9389be660888b0b6d66bc4402
route=$1
mode=$2
role=$3
address=$4
[[ "$route" == direct || "$route" == native ]]
[[ "$mode" == original || "$mode" == capture ]]
[[ "$role" == 0 || "$role" == 1 ]]
name="codex-flagcx729-vendorpg-sdk-20261120-$route-$mode-$role"
label="vendorpg-sdk-20261120-$route-$mode-$role"
output="$root/$route-$mode-role-$role"
port=33565
[[ "$mode" != capture ]] || port=33523
cd "$root"
sha256sum -c probe-inputs.sha256

# Hold only our selected bank and recheck immediately before starting Docker.
exec 9>/tmp/codex-p800-4-6.lock
flock -n 9 || { echo 'Selected bank is cooperatively reserved'; exit 12; }
xpu-smi > "$output.pre-smi"
python3 - "$output.pre-smi" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
lines = text.splitlines()
seen = {}
for index, line in enumerate(lines):
    match = re.match(r'\|\s+([46])\s+P800\b', line)
    if match:
        memory = re.search(r'(\d+)MiB\s*/', lines[index + 1])
        assert memory, line
        seen[int(match[1])] = int(memory.group(1))
assert seen == {4: 0, 6: 0}, seen
processes = text.split('| Processes:', 1)[1]
assert not re.search(r'^\|\s+[46]\s+\d+', processes, re.M), processes
PY
if docker inspect "$name" >/dev/null 2>&1; then
    echo 'Owned name already exists; inspect it before retrying' >&2
    exit 13
fi

finish() {
    if [[ "$(docker inspect --format '{{index .Config.Labels "codex.flagcx729"}}' "$name" 2>/dev/null || true)" == "$label" ]]; then
        docker stop -t 10 "$name" >/dev/null 2>&1 || true
    fi
}
trap finish EXIT
set +e
timeout --signal=TERM --kill-after=15s 355s docker run --pull=never --rm --name "$name" \
    --label "codex.flagcx729=$label" --network host --ipc host \
    --device /dev/xpuctrl --device /dev/xpu4 --device /dev/xpu5 \
    --device /dev/xdrdrv --device /dev/infiniband --ulimit memlock=-1 \
    -v /etc/libibverbs.d:/etc/libibverbs.d:ro \
    -v /usr/lib/x86_64-linux-gnu/libibverbs:/usr/lib/x86_64-linux-gnu/libibverbs:ro \
    -v "$root/libbkcl-vendor.so:/usr/local/xccl_Linux_x86_64/so/libbkcl.so:ro" -v "$state/rdma-lib:/rdma-lib:ro" -v "$root:/probe:ro" \
    -e CUDA_VISIBLE_DEVICES=0,1 -e "NODE_RANK=$role" -e "VLLM_HOST_IP=$address" \
    -e "PROBE_ROUTE=$route" -e "REPLAY_STREAM=$mode" -e "MASTER_PORT=$port" \
    --entrypoint /bin/bash "$image" -ec '
        source /etc/bash_env.sh
        unset BASH_ENV FLAGCX_PATH VLLM_USE_V2_MODEL_RUNNER VLLM_FL_FLAGOS_WHITELIST
        unset VLLM_FL_FLAGOS_BLACKLIST VLLM_FL_FLAGOS_BLACKLIST_APPEND
        export PATH=/flagos/bin:$PATH
        export TORCH_DEVICE_BACKEND_AUTOLOAD=0
        export PYTHONDONTWRITEBYTECODE=1 PYTEST_ADDOPTS="-p no:cacheprovider"
        export LD_LIBRARY_PATH=/rdma-lib:${LD_LIBRARY_PATH:-}
        export VLLM_PLUGINS=fl GEMS_VENDOR=kunlunxin USE_FLAGGEMS=1
        export GLOO_SOCKET_IFNAME=eth0 FLAGCX_SOCKET_IFNAME=eth0 BKCL_SOCKET_IFNAME=eth0
        export FLAGCX_DEBUG=INFO BKCL_RDMA_NICS=bond0,bond1,bond2,bond3,bond4,bond5,bond6,bond7
        export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200 VLLM_COMPUTE_NANS_IN_LOGITS=1
        if [[ "$PROBE_ROUTE" == direct ]]; then
            mkdir -p /tmp/plugin
            tar -xzf /probe/plugin-23b9007f.tar.gz -C /tmp/plugin
            SETUPTOOLS_SCM_PRETEND_VERSION=0.5.0.dev0 python -m pip install --no-index --no-deps --no-build-isolation /tmp/plugin
        fi
        python -c "import torch; assert torch.cuda.device_count() == 2; assert [torch.cuda.get_device_properties(i).pci_bus_id for i in range(2)] == [132, 182]"
        python -m torch.distributed.run --nnodes=2 --nproc-per-node=2 \
            --node-rank "$NODE_RANK" --master-addr <MASTER_ADDR> --master-port "$MASTER_PORT" \
            "/probe/probe-vendor-pg-graph.py" --replay-stream "$REPLAY_STREAM"
    ' > "$output.log" 2>&1
result=$?
set -e
printf '%s\n' "$result" > "$output.exit"
finish
trap - EXIT
xpu-smi > "$output.post-smi"
exit "$result"
