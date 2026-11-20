# Unpromoted FP8 unified relaxed-ordering trial: full Gate still fails

Baseline e84e05be and local candidate7845e327 differ only in FP8 unified OUT
atomic ordering. INT8 retains acquire/release. This generic FP8 unified change
is not restricted to Hopper; the evidence here covers H20 only. The trial is
not pushed into the production PR and must not be described as completing G6.

G5 passed both target(vLLM0.32/Torch2.13) and CI(vLLM0.20.2/Torch2.11): each
candidate full492 file is316pass/176originalskip, plus37packing and60stream,
graph and alias checks. Both target baseline/candidate14 numerical cases and
154input fingerprints match. The stronger native-provider/dispatcher guard
belongs to G6; it was not retrospectively claimed for G5.

G6 kept original full14 order/154inputs/BF16 maxdiff<0.04/10ms warmup/50ms
repetition/three alternating native-Gems pairs and per-case median ratio>=0.9.
Fixed complete baseline then complete candidate, independent fresh caches,
original core385, actual native binaries/dispatcher and compiler tools equal.
All28numerical checks pass. Both variants remain **9/14 performance: FAIL**.
All original cases and paired rounds, including every failure, are retained.

Large FP8T1 median Gems latency268.864us ->241.376us (27.488us/10.22% lower).
Paired speedup0.566912 ->0.637135 still misses0.9. INT8T1/T4/T16 and FP8T4
also remain below0.9. Natural I32/N128/KH128/W4/S1 winner,80regs/0spill is
unchanged. Real SASS replaces the ordering/ATOMG sequence with REDG; static
instruction evidence does not prove residual atomic contention is dominant.

Both complete timers finish before CPU SASS disassembly. Winner observation
is a separate natural replay after each original timer; no forced winner,
subset generation or profiler timing replaces acceptance. The1500s total
budget was not reset:12:30:54Z ->12:35:42Z,287.438s, workexit2/cleanup0.
Source-after verified both1520blobs/3symlinks; owned container absent, both
reserved GPUs idle/no target PID and four locks released. Machine identifiers
and unrelated workload details remain in the local raw receipt only.
