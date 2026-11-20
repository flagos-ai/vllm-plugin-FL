# PR675 follow-up: execution consistency and logprobs cost

Source under test: `e8a8215dddb9cc1e850c7953de0582864fcda66a`.
This folder records observations, including failed checks; it is not a claim
that R1–R8 or the platform CI have all passed.

## RMSNorm: a failing strict consistency diagnostic

`rms-input.json` contains actual BF16 tensors captured from
`model.layers.6.post_attention_layernorm`, not generated model inputs.
Its source SHA256 identifies the original capture. The exact BF16 values are
stored as JSON numbers so no private checkpoint path is needed.

On S60, from the repository root, run:

```bash
python docs/validation/pr675-e8a8215-followup/rms_execution_matrix.py
```

The script uses device 5 and requires the same GCU software environment as the
previous gate evidence. It intentionally exits nonzero when bitwise equality
fails. With shape `[12, 1024]`, no-residual compile/replay matched GCU eager
exactly. With residual, the normalized output differed at **1/12288** elements,
maximum absolute difference **0.0009765625**; the residual sum matched exactly.
Three graph replays retained that discrepancy. The recorded assertion failure
is not a passed precision gate and is not evidence of a NVIDIA-provider error.
Independent NVIDIA execution remains outstanding; the authorized endpoint was
not accepting SSH connections during this follow-up.

## Public top-k probe

`probe_public_topk.py` exercises the installed public FlagGems API and checks
both returned values and indices against CPU top-k. Random operator inputs
are only an operator microbenchmark, not model correctness evidence.
Batch 1 × vocabulary 151936 completed exactly after **590.652 seconds** of
cold compilation, with mean warmed call time **2.456 ms** (ten calls).
The long active LLVM compilation must not be reported as a proven deadlock.
Other shapes require their own completed observations before replacing the
production fallback. Cold cost and warmed end-to-end cost are distinct.

## Service measurements

The service comparison warms **both** logprobs modes at **each** concurrency
using the same 32-token generation length before timing ABBA trials. Initial
batch-1-only warmup measurements were discarded from this comparison because
larger batches had cold execution costs. Streaming `token_ids` establish
one-token arrivals; text chunks are not treated as tokens. CPU load means
user+system CPU seconds over wall time for the server process tree, not host
utilization percent. Two trials per mode are descriptive, not confidence
intervals or proof of causality. Different models/TP sizes do not isolate
vocabulary size. Other device compilation can contend for host CPU.

### Observed service results

Each row averages two timed trials per mode. ITL is milliseconds; CPU is
average server-tree occupied cores. Raw samples remain in each model folder.

| Model (vocab, TP) | Concurrency | ITL off → on | ITL change | CPU off → on | Throughput off → on (tokens/s) |
|---|---:|---:|---:|---:|---:|
| Qwen3-0.6B (151936, 1) | 1 | 214.60 → 215.53 | +0.44% | 1.019 → 1.021 | 4.628 → 4.606 |
| Qwen3-0.6B (151936, 1) | 4 | 218.43 → 221.25 | +1.29% | 1.021 → 1.025 | 17.336 → 17.091 |
| Qwen3-0.6B (151936, 1) | 16 | 254.54 → 262.81 | +3.25% | 1.022 → 1.031 | 56.665 → 54.927 |
| Qwen3.6-27B (248320, 2) | 1 | 500.95 → 505.37 | +0.88% | 3.042 → 3.047 | 1.977 → 1.961 |
| Qwen3.6-27B (248320, 2) | 4 | 528.58 → 531.15 | +0.49% | 2.802 → 3.046 | 5.228 → 6.858 |
| Qwen3.6-27B (248320, 2) | 16 | 570.00 → 591.58 | +3.79% | 2.940 → 2.733 | 22.144 → 16.900 |

The 27B throughput means are strongly affected by wall-time outliers: one
logprobs-off concurrency-4 trial and one logprobs-on concurrency-16 trial.
Do not interpret +31.18% / −23.68% throughput changes as isolated top-k effects.
This is a cost characterization, not a provider migration or performance
acceptance claim. A controlled CPU-versus-public-device-provider service
comparison is still needed before changing dispatch priority.

`run_r8_matrix.py` reproduces the eager service setup using `/models` weights,
GCU devices 2/3 and port 18581; set `OUTPUT_DIR` to choose its output parent.
It writes raw measurements; the summary here retains the completed original
run, not predicted output from a future run.

## Unit-test import isolation

The remote S60 unit job `136146408198` reported 4 failures in the Kunlunxin
VL compatibility mocks (2565 passed, 287 skipped), not four device numerical
failures. The fake module replaced `sys.modules` without replacing a cached
parent-module attribute. The test fixture now keeps both consistent and
restores previous bindings. A regression explicitly supplies a stale parent
binding. Targeted validation: **7 passed**. Remote CI must rerun before this
can be called a platform-CI pass.

Public top-k batch 4 × vocabulary 151936 also completed exactly: cold
compilation **607.901 seconds**, warmed mean **6.813 ms**. The 248320-vocabulary
cases were still running when this evidence snapshot was prepared; absent
rows are pending, not passes.

### Local unit-suite scope

The unfiltered `python -m pytest -q tests/unit_tests` attempt stopped during
collection because the GCU environment lacks `torch_npu`, required by
`dispatch/test_ascend_vocab_parallel_embedding.py`. This is not a pass.
The explicit platform-scoped follow-up command was:

```bash
python -m pytest -q tests/unit_tests \
  --ignore=tests/unit_tests/dispatch/test_ascend_vocab_parallel_embedding.py
```

Observed result: **2570 passed, 293 skipped, 5 warnings in 100.85 seconds**.
The complete output is `unit-followup-gcu.log`. No failing test was filtered
from this completed run; the one ignored file requires a different backend.
