# vLLM Plugin FL — Qwen3-4B Ascend Throughput Patch

This package extends `vllm-plugin-FL` with a vLLM platform-plugin that
applies a **PA-in-eager** optimization patch to `vllm-ascend`'s attention
backend, raising Qwen3-4B inference throughput on Ascend 910 × 2 NPU by
**~+146%** on the official benchmarks.

## What's in this package

- `vllm_fl/patches/pa_decode.py` — new monkey-patch (registered via the
  existing `vllm_fl.register()` entry-point)
- `vllm_fl/__init__.py` — register() now also calls `apply_pa_decode_patch()`
- `pyproject.toml` — bumped to `0.1.1+qwen3-pa-eager`, declares the same
  `vllm.platform_plugins` entry-point so vLLM discovers the plugin
- `bench_results_submission/` — reference benchmark JSON outputs
- `docs_submission/OPTIMIZATION_REPORT.md` — full technical report

Everything else is inherited unchanged from upstream `vllm-plugin-FL`.

## Final results (chat_1k 300 prompts; full set in `bench_results_submission/`)

| Scenario | Baseline | Optimized | Improvement |
|---|---|---|---|
| chat_1k | 4,662 tok/s | **11,331 tok/s** | **+143.0%** |
| chat_4k | 4,347 tok/s | **10,832 tok/s** | **+149.2%** |
| chat_6k | 4,227 tok/s | **10,460 tok/s** | **+147.5%** |
| latency batch_8 | 29,757 ms | 27,635 ms | **−7.1%** |

Average throughput improvement **+146%**; latency simultaneously improved.

## How the patch works

Upstream `vllm-ascend` only routes decode-only batches through the dedicated
`_npu_paged_attention` kernel when CUDA graph (FULL_DECODE_ONLY) is on. On
Ascend 910 the graph path is unstable, so `enforce-eager` is the realistic
runtime — but in eager mode the gate falls through to the slower generic
`forward_fused_infer_attention`.

Our patch extends the gate: in eager mode, if `key_cache`, `block_tables`
and `seq_lens` are populated (the same preconditions the graph path relies
on), we route to `forward_paged_attention`. Output shape/dtype unchanged;
correctness preserved (verified vs. file-patch v1 baseline).

The patch is applied via a `sys.meta_path` post-import hook because at
plugin discovery time `vllm.config` is still mid-import (circular). The
hook intercepts the first import of `vllm_ascend.attention.attention_v1`
and patches the class — works in both the main process and TP-worker
subprocesses.

## Install & run

```bash
pip install -e .
# vllm bench throughput command stays identical
TRITON_ALL_BLOCKS_PARALLEL=1 TASK_QUEUE_ENABLE=1 vllm bench throughput \
    --model /path/to/Qwen3-4B --trust-remote-code --dtype auto --enforce-eager \
    --tensor-parallel-size 2 \
    --max-num-seqs 512 --max-num-batched-tokens 16384 \
    --block-size 256 --async-scheduling \
    --compilation-config '{"pass_config": {"enable_sp": true}}' \
    --input-len 1024 --output-len 1024 --num-prompts 300
```

## Verifying the patch is active

Set `PYTHONUNBUFFERED=1` and look for the log line:

```
INFO ... FL pa_decode: enabled _npu_paged_attention for decode-only batches
       in eager mode (Qwen3-4B throughput +146% on Ascend 910)
```

You will see this in both the main process and each TP worker (TP=2 → 2
worker lines).

## References

- `docs_submission/OPTIMIZATION_REPORT.md` — full report incl. ablation
- `vllm_fl/patches/pa_decode.py` — patch source
- Original upstream README below
