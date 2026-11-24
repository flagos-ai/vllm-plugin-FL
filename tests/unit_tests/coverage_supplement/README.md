# Multi-Chip Unit Test Coverage Supplements (vLLM 0.32 / Runtime 2.3.0)

This directory contains isolated, 100% `RC=0` verified supplementary unit tests targeting `vllm_fl/` shared modules and per-vendor backends, without modifying any `vllm_fl/` source code or existing repository tests.

| Platform | Clean Verification | Final Union Line Coverage | Subdirectory |
|---|---|---|---|
| **Moore Threads S5000 (MUSA)** | `CLEAN_FINAL_RC=0` (2,421 passed, 0 failed) | **93.65%** (12,806 / 13,675, 111 files) | `musa/` |
| **Ascend 910C** | 44/44 groups `RC=0` (0 failed) | **91.43%** (13,892 / 15,194) | `ascend/` |
| **Enflame S60 (GCU)** | 22/22 files `RC=0` (0 failed) | **90.61%** (13,683 / 15,101) | `enflame/` |
| **MetaX C550** | Pending (queued after PR #801 CI frees runner) | TBD | *(to be added)* |
