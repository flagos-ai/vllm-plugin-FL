# Real worker/runner lifecycle regression (vLLM 0.32.0)

`runner_lifecycle.py` loads a real model on the selected devices. It checks the
actual worker runner class and configuration, completed greedy generation,
`update_config` propagation, and prefix-cache reset with stable output tokens.
It accepts the genuine `0.32.0` and empty-build `0.32.0+empty` package metadata;
other versions fail. No mocked worker or generated model substitutes are used.

Example from the repository root (replace paths with real local assets):

```bash
CUDA_VISIBLE_DEVICES=8,9 PYTHONPATH=$PWD \
VLLM_PLUGINS=fl VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
FLAGCX_PATH=/path/to/FlagCX \
python tests/e2e_tests/inference/runner_lifecycle.py \
  --runner v1 --tp 2 --model /path/to/Qwen3-4B \
  --output /path/to/results/v1-tp2.json
```

The serialization opt-in is needed for this trusted local probe's callable
worker-inspection RPC. Do not enable it on an untrusted service or change
production defaults. The interpreter must have the target vendor runtime and
this worktree must be selected through `PYTHONPATH`.

Use `--runner default`, `v1`, and `v2` as separate runs. `default` explicitly
unsets `VLLM_USE_V2_MODEL_RUNNER`; explicit modes set it before importing vLLM.
Use `--graph` to exercise compilation/capture rather than eager mode, and
`--tp`/`--pp` for supported parallel layouts. PP greater than one disables async
scheduling because the retained vLLM 0.32 V1 PP path requires it.

`--sleep` additionally requests sleep/wake and generation afterwards. Only run
this positive case on a platform advertising sleep support. Iluvatar currently
reports sleep unavailable; its ordinary lifecycle results do not prove wake-up.
Unsupported V2 selections on other vendors must be evaluated separately: a
constructor rejection is not a successful generation case.

The probe writes JSON only after its assertions pass and prints
`R6_RUNTIME_PASS`. Inspect the full process log and exit status as well: a
successful generation does not establish clean worker shutdown. Preserve each
run under a distinct output path and record the plugin revision and environment.

Iluvatar PP currently uses a vendor-instance-local synchronous FlagCX payload
fallback, with Gloo metadata/CPU transport. This validates correctness only;
it makes no overlap or throughput claim. Same-device token stability is a
functional regression check, not cross-vendor precision alignment. Hardware
backends not executed independently remain unverified.

## Automatic fallback with real external-launcher PP

The target vLLM 0.32 V2 runner does not implement PP output broadcast for the
external launcher. With runner selection unset, this case must actually choose
V1 and generate successfully on both ranks:

```bash
CUDA_VISIBLE_DEVICES=8,9 PYTHONPATH=$PWD VLLM_PLUGINS=fl \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
FLAGCX_PATH=/path/to/FlagCX \
python -m torch.distributed.run --standalone --nproc-per-node=2 \
  tests/e2e_tests/inference/runner_lifecycle.py \
  --runner default --external-launcher --pp 2 \
  --model /path/to/Qwen3-4B --output /path/to/results/fallback
```

Each rank writes `<output>.rank<RANK>.json`, preventing concurrent writes to the
same file. Require both rank results. The probe also explicitly invokes the
0.32 engine-core client's shutdown with a 30-second timeout; JSON generation
alone does not imply that subsequent teardown passed.
