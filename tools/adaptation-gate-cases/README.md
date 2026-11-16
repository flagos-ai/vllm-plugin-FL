# Adaptation gate: release/0.2, DeepSeek V4 MetaX

This branch uses vLLM 0.20.2. Its DS gate covers the supported text-only model.
It is adapted from the text tasks and quality checks of the
[main-branch gate at d6d8380](https://github.com/flagos-ai/vllm-plugin-FL/tree/d6d83803980fb78d84dcc1b3bc5b915b1fddb237/tools/adaptation-gate-cases).
It does not claim to run that gate's Qwen model or image matrix.

| Scenario | Requests per mode | Check |
|---|---:|---|
| Detailed introduction | 1 | Required concepts, at least 256 characters |
| Concurrent short text | 8 | Independent factual/arithmetic answers |
| Long text | 1 | Approximately 16K input; exact verification code |
| Mixed prompt lengths | 8 | Independent codes across short and long prefills |
| Decode batch coverage | 64 | Distinct codes; inspect server capture logs for buckets |
| Image / mixed text-image | Not applicable | DS checkpoint is text-only |

Run all five scenarios in **both** eager and the deployed FULL_DECODE_ONLY mode.
Prefill CUDA graphs are intentionally unsupported; labeling this mode as full
prefill graph coverage is incorrect. The endpoint does not attest graph mode:
retain the actual startup configuration and graph capture log with the results.
HTTP 200 alone does not pass: answers must meet semantic and corruption checks,
and truncated responses fail. Each scenario saves its exact inputs and API outputs.
Exceptions fail the gate without retries.

Start the example service with a fresh cache and unused port. For the eager
run append --enforce-eager to the same launch command. Stop only your own test
service before switching modes; never stop another workload to free resources.

```sh
python -I examples/deepseek_v4_metax/serve.py \
  --model /models/DeepSeek-V4-Flash-W8A8-INT8 \
  --cache-dir /work/cache-graph --port 18141 --execute
# Run from another terminal:
BASE_URL=http://127.0.0.1:18141/v1 \
SERVED_MODEL_NAME=deepseek-v4-flash-w8a8-int8 \
EXECUTION_MODE=decode_graph RESULTS_DIR=/work/gate-results \
  pytest tools/adaptation-gate-cases/test_deepseek_v4.py -v
```

Repeat with --enforce-eager, another cache/port and EXECUTION_MODE=eager.
The client requires requests and pytest; the
[platform usage guide](../../docs/usage/deepseek-v4-metax.md) describes the
fixed runtime and checkpoint. Save current source/diff identity, dependency
SHAs, image digest, effective config, server log and per-case JSON as CI or PR
evidence. Separate passed, failed, not-run and inapplicable items.
This is functional adaptation validation, not a quality dataset or throughput benchmark.
