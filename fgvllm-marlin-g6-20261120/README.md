# Marlin native G6 investigation, 2026-11-20

Source: FlagGems-vllm PR #1347 (`fdb5e39`); production kernel SHA256
`beab14356ede698e7087131e20a7efa7affd70b5bd3fa4d5e8c2bd98adcbdb7c`.
Runtime: NVIDIA H20-3e, vLLM 0.32.0, Torch 2.13.0+cu130,
FlagTree 0.7 / Triton 3.6, BF16. Runtime packages were isolated from the CI
vLLM 0.20.2 / Torch 2.11 profile. This is single-GPU operator validation,
not collective ABI qualification.

The frozen active set has seven INT8 and seven FP8 cases, identical to
`probe.py`. Native vLLM Marlin uses the same codes/scales in its own repacked
layout. Input generation and repacking are outside timing. Every candidate
passes the unchanged max-diff <0.04 numerical check before timing. Three
rounds alternate reference/candidate using the repository's median
`triton.testing.do_bench`, 10ms warmup and 50ms repetitions. Seeds are fixed
per shape. No numerical tolerance, case, dtype or 0.9 speedup gate is relaxed.

The source baseline passes all 14 numerical cases but only 8/14 performance
cases. See `baseline-receipt.json` for every ratio. CPU/CUDA profiler data is
diagnostic only; the baseline timings are from an unprofiled process.
Kernel summary durations are microseconds summed across three calls.

Trials are separate full source copies. They test four-token route GEMV,
scalar instead of packed dequant, grouped routing, separate gate/up kernels,
route-adapter libentry, and FP8 single-token GEMV. Valid trial timings and
numerical records are retained. The FP8 GEMV prototype had setup/name and
masked-load typing errors corrected before its valid run
(`trial-fp8-gemv-mask.log`); those setup errors are not operator Gate failures.
None of these trials closes full G6, and none was pushed into production.
Only the benchmark API compatibility was committed in `fdb5e39`.

The automatic PR run 44651168345 passed while selecting zero operator tests;
it does not establish full acceptance. The prior full integrated functional
run 44570711676 at c40baf66 passed 11024 tests, 2379 skips, zero failures.
A new exact integrated run is required after current repairs. FlagTree #1691
tracks the illegal-instruction candidate controls. Compiler #1540 remains
under investigation and is not claimed as a fix.
