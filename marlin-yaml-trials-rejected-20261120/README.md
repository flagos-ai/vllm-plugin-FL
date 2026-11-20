# Standard-YAML Marlin trials rejected

All three approved diagnostic candidates passed their original numeric checks and all14 generated-input fingerprints, then failed the unchanged performance Gate. No candidate was pushed to production; no full14 candidate acceptance is claimed.

| Variant and target | Baseline speedup | Candidate speedup | Result |
|---|---:|---:|---|
| INT8 T1 large, BN8/K128/U2/W8/S1 | 0.810568 | 0.700339 | Reject |
| INT8 T1 large, BN4/K128/U2/W4/S1 | 0.810568 | 0.590049 | Reject |
| FP8 T4 large, Down N64/K128/W8/S1 | 0.819179 | 0.648340 | Reject |

The canonical baseline ran original unfiltered14 twice: accuracy14/14, then formal numeric14/14 and performance9/14. Its five deficits are INT8 large T1=0.810568, T4=0.329829, T16=0.742637; FP8 large T1=0.568160, T4=0.819179. All raw three paired rounds and all14 cases remain in results/baseline-formal-qualification.json. Tiny INT8 T1 is a separate unified-path control, not the route-GEMV target.

Git e5c1a998ff233a0e5ab09f861fcb150ce4c11f7a/tree8c0f7997e9221a6f8972c2fb019786d4edf06b75, Torch2.13.0+cu130, vLLM0.32.0, FlagTree0.7.0/Triton3.6, original core385c1089. The new archive contains canonical Git LF operator77739b56; historical Windows CRLF operator8332eb9e is byte-different. LF normalization and Python AST match for operator/test/both benchmarks. The new canonical baseline establishes the boundary rather than assuming the old byte hash. Linux CPU preflight verified1491 original Git blobs including3 real symlinks; no source modification except the candidate's ordinary Hopper YAML is allowed. Frozen candidate-manifest.json predates execution, so its no_gpu_or_remote_execution_yet flag describes package preparation only.

Captured YAML, probe and compiler-IR byte streams retain their original line endings and whitespace. The three original probes are preserved byte-for-byte. Baseline ran probe.py with no filter. Candidates used the validated replay helper that calls all14 generators before selection, retaining the FP8 cached-bank and RNG sequence; all14 tensor fingerprints exactly match the original full-input reference in all8 phases. Selected cases were two INT8 T1 controls or one FP8 T4 diagnostic. Correctness preceded timing. BF16, maxdiff<.04, 10ms warmup/50ms repetition and three alternating native/gems pairs are unchanged; the Gate uses each case's median-of-three speedups>=.9. No profiler ran during formal. Registers, spills, launch metadata and TTGIR/PTX were collected after timing; see each resources.json and IR file.

Lower registers did not improve these targets: INT8 Gate/Down registers64/39 baseline,40/32 BN8W8,48/32 BN4W4; all spills0. The FP8 Down W8 candidate had62 registers,0spills,9216B shared. These are observations, not proof of an occupancy or bandwidth root cause. Existing source attaches pipeline_hoist_copy=True to this FP8 stage1 config; this trial did not introduce it.

Each variant used its own new task cache, accuracy then formal sharing only that variant's ordinary cache. The target was physical GPU7 exposed as logical0, task-exclusive. Other GPUs had external activity; this does not claim whole-host idleness. hardware/ preserves target clocks/power/temperature and host load immediately before/after formal; UUIDs are redacted. Every owned container, lock and tmux session was released; targetGPU7 and observedGPU3 were0MiB/0% at08:18:12Z. No external process was stopped. No additional GPU profiling is part of this bundle.

raw-archive-sha256.json seals all8 original archive hashes; originals remain local and unchanged. original-sanitized-manifest.json maps every included byte stream to its original and sanitized hashes. External process details and private GPU UUIDs are excluded.
