#!/bin/bash
set -euo pipefail
for profile in target ci; do
  export PYTHONPATH=/candidate/src:/candidate
  if [[ "$profile" == target ]]; then
    python=/usr/bin/python3
    export PYTHONPATH=/compiler:$PYTHONPATH:/deps
  else
    python=/work/source/.venv/bin/python
  fi
  "$python" - <<'PY'
import json,torch,triton,vllm,importlib
from flaggems_vllm.utils.triton_version_utils import HAS_TLE
m=importlib.import_module('flaggems_vllm.ops.mhc.mhc_pre')
reader=m.mhc_pre_packed_reader_kernel
assert len(reader.fn.configs)==15
assert all(c.num_stages==1 for c in reader.fn.configs if c.kwargs['BLOCK_M']>=64)
print('RUNTIME',json.dumps({'torch':torch.__version__,'triton':triton.__version__,'vllm':vllm.__version__,'tle':HAS_TLE,'libentry':type(reader).__name__,'source':m.__file__}),flush=True)
PY
  export TRITON_CACHE_DIR="/probe/hopper15-$profile-triton"
  export FLAGGEMS_CACHE_DIR="/probe/hopper15-$profile-gems"
  for temperature in cold warm; do
    mkdir -p "/probe/reports/hopper15-$profile-$temperature"
    cd "/probe/reports/hopper15-$profile-$temperature"
    status=0
    "$python" -m pytest -q -c /candidate/pytest.ini -p no:cacheprovider --tb=short /candidate/tests/test_mhc_with_norm.py > "/probe/hopper15-$profile-$temperature.log" 2>&1 || status=$?
    printf '%s\n' "$status" > "/probe/hopper15-$profile-$temperature.status"
    tail -6 "/probe/hopper15-$profile-$temperature.log"
    if [[ "$status" != 0 ]]; then exit "$status"; fi
  done
  if [[ "$profile" == target ]]; then
    export TRITON_CACHE_DIR=/cache/target-v032-flagtree-triton
    export FLAGGEMS_CACHE_DIR=/cache/target-v032-flagtree-gems
  else
    export TRITON_CACHE_DIR=/cache/fixed-ci-triton
    export FLAGGEMS_CACHE_DIR=/cache/fixed-ci-gems
  fi
  mkdir -p "/probe/reports/hopper15-$profile-nearby"
  cd "/probe/reports/hopper15-$profile-nearby"
  status=0
  "$python" -m pytest -q -c /candidate/pytest.ini -p no:cacheprovider --tb=short /candidate/tests/test_mhc_ops.py /candidate/tests/test_mhc_post.py /candidate/tests/test_mhc_pre_broadcast.py /candidate/tests/test_mhc_shifted_post_pre.py > "/probe/hopper15-$profile-nearby.log" 2>&1 || status=$?
  printf '%s\n' "$status" > "/probe/hopper15-$profile-nearby.status"
  tail -6 "/probe/hopper15-$profile-nearby.log"
  if [[ "$status" != 0 ]]; then exit "$status"; fi
done
