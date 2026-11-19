# GDN 64K single-request trace attribution

Measured with Nsight Systems 2025.3.2 on the same four H100 GPUs and the
`c4a888c` PCP backend experiment. No production kernel or dispatch default
changes are included. Each backend is warmed up five times, then three
single-request 65,536-token prefills are captured, producing one token each.

The raw reports are retained in:
`/vllm-workspace/validation/qwen36_gdn_trace_64k_20261013`.
`flashinfer/prefill.nsys-rep` and `triton/prefill.nsys-rep` can be opened in
Nsight Systems. Both also have SQLite exports, launch manifests, input shapes,
source snapshots, per-layer/per-kernel CSVs and complete analysis JSON.

For reproduction, stage the five Python files in this directory into a new
direct child directory of `/vllm-workspace/validation`. Also stage these files
from the parent benchmark directory: `gpu_guard.py`, `timing_worker.py`,
`benchmark_norm_ablation.py`, `benchmark_fused_norm.py`.

Run from the fresh directory:

```bash
python3 capture.py
python3 analyze.py
python3 report.py
```

The capture script uses the existing workspace model, prompt file, shared
GPU lock, experiment worktree, and Nsight installation. It refuses to launch
if any GPU is occupied. Capture directories must not exist. The report also
reads the prior matched unprofiled benchmark summary for context, retaining
that benchmark's original path rather than treating trace timings as a new
unprofiled performance claim.

Attribution joins CUDA activities to host launch APIs by global process ID
and correlation ID, then identifies enclosing request/core/stage NVTX ranges.
It preserves rank and sample instead of adding GPU times across devices.
Nsight's global process field contains a VM ID; only the process portion is
used to match worker rank. Analysis requires all 360 GDN ranges per backend
to have GPU activity and all 30 layers per request/rank to be represented.

Measured launch grids establish the actual decomposition. Nsight Systems
does not supply the achieved-occupancy/SM-active/bandwidth hardware counters
needed to assign every millisecond to one microarchitectural cause. Numerical
differences between the backends remain documented by the earlier correctness
experiment; these traces do not establish broader quality equivalence.
