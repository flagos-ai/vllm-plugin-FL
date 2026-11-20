# GLA static-padding CPU compile comparison

This is a CPU-only controlled compiler diagnostic, not a normal GPU Gate, numerical validation, benchmark acceptance or merge-ready claim. The production change remains uncommitted/unpushed pending normal-GPU G5/G6 and review.

| Same fixed specialization | Original | Candidate |
|---|---:|---:|
| Compile wall seconds |262.311241|4.525772|
| PTX bytes |2,621,908|707,430|
| Declared PTX virtual registers |56,752|17,084|
| Cubin bytes |6,678,200|1,064,376|

The offline compile ratio is57.9594x. Virtual-register declarations are not physical GPU register usage. Neither .n_regs/.run nor driver handles were queried. Historical normal-GPU artifacts had different size/virtual-register totals; this does not claim exact reproduction of their wall time. Raw LLIR/cubin hashes are retained in each compile receipt; binaries and full original logs remain in the raw archive (SHA bd73e1aeeb9ee450b1e746bfd66175ee263af80e359294a22135c4a5462812b7).

Base main14a3602284b86337ffcf9fa7d025f9ff89a411bb has byte-identical chunk_gla.py to integrationbc8840d/current prep, SHA7930fdb308d2c81e8f39407110240b854ff3dc530e4a4c5891fd7d962b44b414. Candidate SHA5df6ef88f202f1b3241be2de4c79ea1b2f6430ed1b34de3739a0f6d237cf7125 adds one constexpr min(B_VCHUNK,ceil(V/BV)) and replaces two static loop bounds. All108 original configurations, autotune key, grid, arithmetic, masks, four initialized accumulators, API, tests and benchmark remain unchanged. The isolated modules extract the real exp2/kernel bytes, remove only autotune/heuristic wrappers and retain JIT. Extracted bodies are AST-identical; reverting the three changes restores the exact original kernel AST.

Both phases used independent empty task caches, original CI Python3.12.15/Torch2.11.0+cu130 package resolution, FlagTree0.7/Triton3.6/core385c1089, ptxasc960a4f2, existing original USE_FLAGCX=OFF and IEEEFP32. No Torch/Gems compute or kernel launch occurred. TLE imports Torch transitively; receipts show cuda.is_initialized false, DriverConfig._active/_default None, no /dev/nvidia devices and no GPU file descriptors before/after compilation. Docker runc,2CPUs/16GiB,no device requests; each phase had its own original30min diagnostic limit and completed normally. Earlier setup failures are preserved separately and are not successful compile measurements.

Specialization: explicit GPUTarget(cuda,90,32), BT64/BK32/BV128/B_VCHUNK4/W2/S2; six fp32 pointers with divisibility16, scale fp32 and dynamic T i32; H=HV=4,K=V=64, no varlen, STATE_V_FIRST false. Signature, constexprs, attrs, target, options, core/ptxas/compiler Python hashes and selector configuration match across both receipts.

The original package marker is an empty file (SHA e3b0...). Its TLE capability check ordinarily auto-detects an active backend and cannot do so without devices. For this CPU diagnostic only, the existing in-memory triton._flagtree_backend.FLAGTREE_BACKEND constant is explicitly set to nvidia on both phases. The marker/library files, dispatch functions and math lowering are not patched. Standard CUDABackend and the identical standard tle.load function/source hash are asserted. Future GPU validation must use normal automatic detection with no override or offline-cache reuse. This extra selector is part of the disclosed diagnostic boundary.

No public package/production repository was changed by publication preparation. original-sanitized-manifest.json records both hashes for every included stream; only owned pipe identifiers are replaced. Raw archives remain immutable. No private IP, GPU UUID or external workload data is included.

See [source attribution and MIT license](SOURCE-ATTRIBUTION.md) for the original extracted-kernel copyright and terms.
