# Matched-input Marlin five-path diagnostic

This is Nsight Systems diagnosis, not formal G6. Production remains PR1347
head e5c1a998ff233a0e5ab09f861fcb150ce4c11f7a, canonical operator SHA
77739b56a594b408732bed9605bd802a48dd49c7957b4510290c3ddd7486231e.
The original full14 formal gate remains 14/14 numerical, 9/14 performance;
this profile does not replace or reinterpret those results.

The complete original14 input bank was generated in the original order before
selection. All14 identities matched the frozen formal inputs and all14 numerical
checks passed, maximum difference0.03223933 against the original0.04 threshold.
Five retained slow cases were traced for3 alternating native/Gems calls each,
30 NVTX ranges. Qualification and JIT/warmup completed before capture. Retaining
all14 until qualification changes the diagnostic allocator/cache footprint;
these are one-call profiling samples, not the formal timer.

| Case | Median native kernel sum, us | Median Gems kernel sum, us |
| --- | ---: | ---: |
| int8-T1-E8-H4096-I14336-K2 | 156.193 | 199.169 |
| int8-T4-E8-H4096-I14336-K2 | 278.465 | 859.843 |
| int8-T16-E8-H4096-I14336-K2 | 525.377 | 726.594 |
| fp8-T1-E8-H4096-I14336-K2 | 143.297 | 258.752 |
| fp8-T4-E8-H4096-I14336-K2 | 365.697 | 440.641 |

Kernel work is higher on all5. INT8 T4's gate/up is636.418us, about74% of Gems
kernel time, followed by220.417us down. Its8 singleton routes target4 distinct
experts, with counts[2,0,2,1,3,0,0,0]. INT8 T16 launches32 M blocks, but only8
are effective and24 return early; do not treat all32 as arithmetic work.
FP8 T1 is the unified split-I kernel,256.224us, grid(2,448), I32/N128/KH128,
80 registers and0 spill. Its BM2 routing preparation is not its CUDA CTA grid.
Neither Nsight Systems nor the resource count proves an atomic/DRAM bottleneck;
targeted counters are still needed before choosing a repair.

CPU analysis joins CUDA correlation IDs to NVTX calling threads and rejects
ambiguous/missing associations. Kernel union/span, empty GPU intervals and API
intervals are separate: synchronize overlaps GPU execution and must not be added
to kernel duration. Empty GPU intervals alone are not proven launch overhead.

The raw local archive is37927a24c16c85e33ba4ed7c494d7c6e1ad0a40941a219d22ff7a3a0042dfbdd.
The real native Nsight report is244150bytes, SHA
1b3bd90eae3a2575c55178726f6651e3a8ded876ebe0784860b0b95ec4a78548.
Raw report/hardware inventories remain local; this public subset contains exact
text/JSON receipts and the CPU analyzer, with no machine UUIDs or unrelated
process inventories. No source, YAML, frozen probes or numerical threshold was
modified, no performance winner was promoted, and the owned GPU/container/locks
were released before CPU export/analysis.

Frozen original profile package SHA66510cc18c36e1963354e97e81d4422ebdee91c7f3d2fe5bffad176e82e830e3;
explicit postcapture add-on v2 SHA30acc713512f6e9e1de0ecea7f681a85731a0cf2752ae60e572f19876c8be710.
The add-on runs the original helper unchanged, then transfers only the retained
small routing IDs/weights after capture/qualification, without new operator or
RNG calls. FP8T1 actual tile/grid above comes from matched trace/resources, not
the metadata-only adapter statistics.
