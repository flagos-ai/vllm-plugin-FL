# 64K single-request GDN backend experiment

These scripts compare native FlashInfer and Triton/FLA GDN prefill dispatch
on Qwen3.6-35B-A3B, using vLLM 0.20.2 and four H100 80GB GPUs. They are an
explicit experiment, never loaded by plugin registration. The backend change
is **not bitwise equivalent**. Matching selected outputs does not establish
broad quality equivalence.

The campaign uses 65,536 input tokens and one request per invocation. Timing
produces one real output token; correctness also uses 128-token free generation
and six known-answer retrieval prompts. It does not test multi-request serving.

## Local reproduction

Stage these scripts into a new directory directly beneath
`/vllm-workspace/validation`. Existing result directories are rejected.
The scripts use the local model `/data/model`, the prompt file
`/vllm-workspace/validation/qwen36_64k_20260928/prompt_ids.json`, the shared lock
`/vllm-workspace/validation/qwen36_fused_layout_64k_20260929/.gpu.lock`, and the
worktree `/vllm-workspace/vllm-plugin-FL-pcp-gdn-backend`.

From the fresh validation directory, run correctness:

```bash
PYTHONPATH="$PWD:/vllm-workspace/vllm-plugin-FL-pcp-gdn-backend" \
VLLM_PLUGINS=fl_qwen_cp VLLM_FL_QWEN_CP=1 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_WORKER_MULTIPROC_METHOD=spawn \
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONUNBUFFERED=1 \
python3 gpu_guard.py correctness.log python3 correctness.py
```

Inspect `correctness.json` before timing. Nonfinite values or modified input
state abort the check; numerical differences are recorded without arbitrary
tolerance acceptance. Same-history logits are compared against both baseline
repeat noise and candidate output. A separate free generation avoids treating
teacher-forced token equality as evidence of natural output equality.

```bash
python3 run_timing.py
python3 report.py
```

`run_timing.py` holds the shared lock and checks GPU process, memory and
utilization state before each process. If occupied, it refuses the launch.
It executes TP → PCP → PCP → TP, reversing variants in round two. Each
condition has three warmups and ten measured calls per round, or 20 formal
samples total. Diagnostics are removed before warmup/timing. TP also receives
the same Norm and GDN choices so generic improvements are separately visible.

`benchmark_fused_norm.py` is byte-identical to the previously validated
Norm-ablation helper. `benchmark_norm_ablation.py` only changes its imports
to use that local helper and the current PCP package. The two Triton kernels
are identical to the current production fused-Norm kernels; the helper adds
an explicit native fallback for experiments installing cached module dispatch.
GDN gated Norm is not altered by this Norm selector.

Raw per-round JSON, source snapshots, launch settings and GPU checks stay in
the validation directory. `report.py` verifies them and writes `REPORT.md`,
`summary.json` and `samples.csv`. The exact measured source includes uncommitted
`gdn_backend_probe.py` saved in each source snapshot, based on `d5798be`.

This run changes no production default. vLLM also exposes the selected native
backend at startup through `additional_config={"gdn_prefill_backend":"triton"}`;
the probe exists to switch the same loaded model for controlled comparisons.
