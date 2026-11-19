#!/usr/bin/env bash
set -Eeuo pipefail
export PYTHONUNBUFFERED=1
unset VLLM_USE_V2_MODEL_RUNNER VLLM_FL_FLAGOS_WHITELIST VLLM_FL_USE_FLAGCX FLAGCX_LOG_LEVEL
cd /qualification/src
git config --global --add safe.directory /qualification/src
test "$(git rev-parse HEAD^{tree})" = 2c375e273dd48b7752bfe1faabf16473a5ec41c3
python -m pip install --no-index --no-build-isolation --no-deps .
python - <<'PY'
import hashlib
import importlib.metadata as md
import json
import os
import pathlib
import torch
import triton
import vllm
import vllm_fl

provider = md.distribution('flaggems-vllm')
provenance = json.loads(provider.read_text('direct_url.json'))
assert provenance['url'].rstrip('/').endswith('e00557103e284db5b57168386db60faa051989ef'), provenance
assert '2e19546b78a838c188b2e4e4f12b1a40b395a13fc75f24fbabb7229e63461656' in json.dumps(provenance), provenance
assert md.version('vllm') == '0.32.0+empty', md.version('vllm')
assert str(pathlib.Path(triton.__file__).resolve()).startswith('/opt/flagtree/'), triton.__file__
assert torch.cuda.device_count() == 4, torch.cuda.device_count()
assert 'VLLM_USE_V2_MODEL_RUNNER' not in os.environ
assert os.environ['FLAGCX_PATH'] == '/opt/FlagCX'
core = pathlib.Path('/opt/FlagCX/build/lib/libflagcx.so')
wrapper = pathlib.Path('/opt/FlagCX/plugin/interservice/flagcx_wrapper.py')
assert hashlib.sha256(core.read_bytes()).hexdigest() == '33b3c77248366928f89b6b75ad6b5eb579126512d39381ebd59a4c245896f0ad'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest() == 'a791806a0ab36da872cd200b5466c36001ef798c8fb444fc62b1d1f6805ca5cf'
assert 'VLLM_FL_FLAGOS_WHITELIST' not in os.environ
devices = [torch.cuda.get_device_name(i) for i in range(4)]
print('RUNTIME', json.dumps({'provider': provenance, 'vllm': vllm.__version__, 'vllm_dist': md.version('vllm'), 'triton': triton.__file__, 'plugin': vllm_fl.__file__, 'devices': devices}))
PY
set +e
torchrun --nnodes=1 --master-addr=127.0.0.1 --master-port=29580 --nproc-per-node=4 --module pytest --noconftest \
    tests/functional_tests/distributed/test_metax_flagcx_collectives.py \
    -q -s -p no:cacheprovider 2>&1 | tee /qualification/collectives.log
collective_exit="${PIPESTATUS[0]}"
set -e
printf '%s\n' "$collective_exit" > /qualification/collective-exit
test "$collective_exit" = 0
mkdir -p /qualification/server-logs
(
    while true; do
        for log in test-results/logs/*.log; do
            if [ -f "$log" ]; then cp "$log" /qualification/server-logs/; fi
        done
        sleep 15
    done
) &
observer=$!
trap 'kill "$observer" 2>/dev/null || true; wait "$observer" 2>/dev/null || true' EXIT
overall=0
for case in "$@"; do
    mkdir -p "/qualification/reports/$case"
    set +e
    python tests/run.py --platform metax --device c550 --scope e2e --task adaptation_gate \
        --cases "[{\"model\":\"qwen3_6\",\"case\":\"$case\"}]" \
        --output-dir "/qualification/reports/$case" 2>&1 | tee "/qualification/reports/$case/run.log"
    status="${PIPESTATUS[0]}"
    set -e
    printf '%s\n' "$status" > "/qualification/reports/$case/exit"
    if (( status != 0 )); then
        overall=1
        continue
    fi
    if ! python - "$case" <<'PY' | tee "/qualification/reports/$case/contract.json"
import json
import sys
from pathlib import Path
from tests.e2e_tests.adaptation_gate.gate_contract import validate_gate_documents

root = Path('test-results/gate/metax/c550/qwen3_6') / sys.argv[1]
matches = sorted(path for path in root.glob("*/port-*") if path.is_dir())
assert len(matches) == 1, matches
print(json.dumps(validate_gate_documents(matches[0]), sort_keys=True))
PY
    then overall=1; fi
done
exit "$overall"
