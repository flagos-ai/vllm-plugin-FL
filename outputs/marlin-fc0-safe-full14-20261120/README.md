# Fresh complete FlagTree main: unchanged Marlin safe-table baseline

The complete fc0ec83b compiler was built from unmodified source, including final
PR1540 and the earlier cp.async fixes. Full source/wheel/core identities are in
new-g1-public.json. This is a separate frozen G1, not a relabeling of old core385
or the earlier6ed3 build. Source e84e05be is the ordinary E5 plus main50d merge;
its operator, all23 Marlin tuning entries, tests, generators and probes are unchanged.

The original full14 order, all154 tensor fingerprints, BF16 maxdiff<0.04,
10ms warmup/50ms repetition and three alternating native/Gems paired rounds were
retained. Every case requires median paired speedup>=0.9. All14 numerical checks
passed. Performance is **9/14: FAIL**, not merge-ready. The five failures remain
INT8 large T1/T4/T16 and FP8 large T1/T4. Every raw latency and paired ratio is
retained in g6-result.json; root independently recomputed them.

Actual slow-five compiled winners were recorded only by an extra natural call
after each case's original timers completed. No hook, candidate restriction,
forced winner or profiler was installed in those timers. Their constexprs,
stages, warps, registers, spills and PTX instruction counts are retained.

The task started11:50:05Z and completed11:52:58Z within its original25min ceiling.
Source-after verified all1520 Git blobs and3 symlinks. Probe/source checks exited0;
performance aggregation/task exited2. Cleanup exited0 and independently verified
the exact owned container absent, both reserved GPUs idle/no target PID, and all
four cooperative locks released. Machine identifiers and unrelated workloads are
omitted here; original resource receipts and IR remain in the raw archive.

The successful build and numerical results do not establish performance acceptance
or imply that the old compiler can safely enable multistage candidates. No source,
production dependency or CI toolchain has been promoted by this experiment.
