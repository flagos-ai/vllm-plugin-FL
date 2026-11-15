# Ascend 910C triangular solve in Qwen3.6-35B-A3B GDN

Qwen3.6-35B-A3B TP2 uses 16 value heads per worker. A 65-token GDN prefill crosses the 64-token boundary, so each worker computes 32 unit-lower FP32 systems. The triangular input is shaped [1,16,2,64,64] and the right-hand side is [1,16,2,64,128]. The [standalone reproducer](triangular-repro.py) uses seeded tensors on a single 910C and enables only the FlagGems solve for the comparison.

| Batch shape | Pinned FlagGems 6ed2f39 | Ascend operator file from upstream 0a81e44 |
| --- | --- | --- |
| 16 heads, one chunk | 1.34042e-7 | 1.34042e-7 |
| 16 heads, two chunks | 0.071648 to 0.077728 | 1.33834e-7 |
| 24 heads, two chunks | 1.34321e-7 | 1.34321e-7 |

Each value is relative L2 error against the same torch-npu solve. There were three trials per shape in each disposable container. Replacing just the installed Ascend implementation file with the one from merged [FlagGems #6596](https://github.com/flagos-ai/FlagGems/pull/6596) restored the numerical behavior without changing dependencies. That upstream change bounds the grid and iterates over remaining work per core. The image-pinned [6ed2f39 operator](https://github.com/flagos-ai/FlagGems/blob/6ed2f39071ef85db26b7e64f0652a162f434959c/src/flag_gems/runtime/backend/_ascend/ops/linalg_solve_triangular.py) still uses the old launch.

On the 65-token synthetic GDN reference, the original plugin dispatch policy produced relative L2 distance 0.124063 on the output and 0.063355 on the final state. With only FlagGems linalg_solve_triangular excluded, both became bitwise equal to native ATen. The [receipts](receipt.json) record the full standalone comparison and the GDN trace excerpt. The [Ascend plugin change](https://github.com/flagos-ai/vllm-plugin-FL/commit/3b887716f6005718876e5794e444652a445f4bf6) uses torch-npu for the affected normal solve overload while the CI image remains pinned.

Environment used the [Ascend CI image](https://github.com/flagos-ai/vllm-plugin-FL/blob/5257d614511f2ba7752032564cbe556ffb15f0d5/docker/ascend/README.md) with CANN 9.0.0, torch-npu 2.10.0, and FlagTree 0.7.0+ascend3.5 providing Triton 3.5.1.
