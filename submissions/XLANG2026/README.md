# XLANG2026: MiniCPM5-2B inference throughput on MetaX C500 and Iluvatar BI-V150

Source changes for the two platform configurations of the S2 Track 2 submission. Both platforms keep the
competition serving parameters, the official benchmark script and the model unchanged; only `vllm-plugin-FL`
and `FlagGems` sources change.

| | Base (plugin) | Base (FlagGems) | Compiler |
| --- | --- | --- | --- |
| MetaX C500 | `13eb9be6` | image `4df52d91` (v5.3.5) | vendor Triton 3.0.0+metax3.7.0.7 |
| Iluvatar BI-V150 | `13eb9be6` | image `4df52d91` (v5.3.5) | FlagTree **0.7.1a2+iluvatar3.6** (official re-verification build) |

## Results

Total tokens/s, `RUNS=4, SKIP_FIRST=1`, official test cases `[[4096,1024,64,256],[16384,1024,64,128]]`.

| Platform | 4k (tok/s) | 16k (tok/s) | vs official baseline | Mean TTFT 4k/16k (ms) | MATH-500 L3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MetaX C500 | **12451.84** | **11097.29** | **2.45x / 1.58x** (mean 1.94x) | 1616.56 / 17980.89 | 96.19% (101/105) |
| Iluvatar BI-V150 (parent commit) | 5292.46 | 4618.41 | 2.61x / 5.05x (mean 3.37x) | 4050.56 / 111878.60 | 98.1% (103/105) |
| Iluvatar BI-V150 (this build, FlagTree 0.7.1a2) | 5045.20 | 4401.06 | 2.49x / 4.81x (mean 3.21x) | 4149.48 / 115735.63 | 95.24% / 98.10% (two runs) |

Official baselines: C500 `5089.645 / 7029.675` tok/s, BI-V150 `2028.01 / 915.15` tok/s. Accuracy gate is
0.962 baseline with a 0.95 floor; TTFT must stay within baseline + 1%.

## Layout

```
submissions/XLANG2026/
├── common/FlagGems/          # files shared by both platforms
├── metax/
│   ├── vllm-plugin-FL/       # MetaX-only plugin sources
│   └── FlagGems/             # MetaX-only kernel sources
└── iluvatar/
    ├── vllm-plugin-FL/       # Iluvatar-only plugin sources
    └── FlagGems/             # Iluvatar-only kernel sources
```

Each subtree keeps the repository-relative path, so an overlay is a plain copy over the base package.

## What changed

**MetaX C500** (sampler chain, GEMM routing, host synchronisation)

- `vllm_fl/.../metax/patches/topk_topp_sampler.py`, `sampler_tail_fusion.py`: replace the three-pass
  top-p/top-k sampling tail with one fused pass (write mask, compact, renormalise), removing the
  repeat/interleave round trips. Measured 4.086 ms -> 0.381 ms per step on the sampling chain.
- `linear_nn_repack.py`, `mcblas_mm.py`: repack linear weights to a single nn layout so the MACA
  `mm_kernel` pipeline can prefetch and `mcBLAS` can be used directly for large-M prefill; a shape gate
  falls back to the original nt layout when the tile pattern does not apply.
- `flash_attn.py`, `mm_autotune_warmup.py`, `slow_step_logger.py`: keep the vendor attention path on the
  official launch configuration, make the start-up GEMM warmup optional (default off) and gate the slow
  step logger.
- `FlagGems`: fused sampling kernels (`top_p_sample.py`, `sampler_fused.py`), `_metax/ops/mm.py` routing.

**Iluvatar BI-V150** (vendor attention, launch config, decode kernel)

- `impl/vendor_flash_attn.py`: page-aligned vendor prefill attention with two-gather de-paging and a
  varlen `out=` path; page size gate (`page % 4096 == 0`) keeps the original path otherwise.
- `impl/decode_attn_fast.py`, `decode_attn_pin.py`: segmented paged-decode attention, base-2 online
  softmax, work unit = (seq, kv_head, segment); launch-config pinning for the decode graphs.
- `impl/mm_silu_mul.py`, `sampler_tail_fusion.py`, `iluvatar.py`: gated MLP fusion with dual accumulators,
  fused sampling tail, backend registration.
- `FlagGems`: `fused/paged_decode_attention.py`, `_iluvatar/fused/mm_silu_mul.py`, `_iluvatar/ops/mm.py`,
  `utils/libentry.py` (autotune cache), `utils/models/sql.py` (config cache key).

**Common**

`FlagGems/pyproject.toml` adds a `fallback_version = "5.3.5"` so `pip install --no-build-isolation .` works on an unpacked tree without git metadata. `FlagGems/fused/__init__.py` registers the new fused ops; `matmuladd.py`, `gumbel_max_sample.py`,
`top_k_top_p.py` carry the `flagtune` strategy hook and the sampling kernels used by both sides.

## How to apply

Start from the official image (vLLM 0.24.0, FlagGems 5.3.5, FlagTree as above). Pick **one** platform.

```bash
# 1. shared files
cp -r submissions/XLANG2026/common/FlagGems/.  /workspace/FlagGems/

# 2a. MetaX C500
cp -r submissions/XLANG2026/metax/vllm-plugin-FL/. /workspace/vllm-plugin-FL/
cp -r submissions/XLANG2026/metax/FlagGems/.       /workspace/FlagGems/

# 2b. Iluvatar BI-V150 (requires FlagTree 0.7.1a2+iluvatar3.6)
cp -r submissions/XLANG2026/iluvatar/vllm-plugin-FL/. /workspace/vllm-plugin-FL/
cp -r submissions/XLANG2026/iluvatar/FlagGems/.       /workspace/FlagGems/

# 3. install both packages, no extra environment variables
cd /workspace/vllm-plugin-FL && pip install --no-build-isolation .
cd /workspace/FlagGems       && pip install --no-build-isolation .
```

Serving and benchmarking use the competition commands verbatim; the only addition is the official
`--no-enable-prefix-caching` used for the performance re-check.

## Verification

- Unit tests: `pytest tests/test_paged_decode_attention.py tests/test_libentry.py`
  (FlagGems) and `pytest tests/unit_tests/models/test_mm_silu_mul.py` (plugin).
- Numerical: paged decode kernel vs PyTorch reference, max error `3.05e-5` (A20) and `3.905e-3`
  (A48/A60); sampler vs reference with TV distance inside the sampling noise band.
- Batch invariance: single prompt, partial batch and full batch produce identical greedy tokens on
  BI-V150 (9 shapes).
- End-to-end: 4k and 16k four-round runs on both platforms, 256/256 and 128/128 requests successful;
  MATH-500 Level 3 twice per platform.
- `SHA256SUMS` lists every file in this directory.
