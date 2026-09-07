# GLM5-Next kernel provenance

The kpool implementation is vendored from the supplied
`glm-5-next-adapt-0808` vLLM reference tree:

- reference commit: `a9b39d6` plus `vllm-glm5next-0808.patch`
- original paths:
  - `vllm/models/glm5next/nvidia/ops/kpool_compress.py`
  - `vllm/model_executor/layers/sparse_attn_indexer_kpool.py`
- license: Apache-2.0

Before vendoring, the equivalent Hugging Face Transformers and SGLang
implementations in the same handoff bundle were checked. The plugin keeps the
reference formulas and cache lifecycle; only import paths and vLLM 0.24
metadata compatibility reads differ.

The upstream vLLM repository was re-checked at commit
`bd6536071cec4dcd8cf91c0e2aa04aec83fc1c37` (2026-08-10). Its Kimi K3 KDA,
KDA metadata, and latent-MoE tail files are byte-identical to the copies in the
supplied `a9b39d6` reference tree. Kimi K3 confirms the bounded-gate wrapper and
`eager_break_during_capture` pattern; its latent-MoE tail is not a KV tail cache
and was not transplanted into the GLM kpool implementation.

The first-layer mHC broadcast specialization was ported on 2026-08-27 from the
newer local `vllm-glm5next-ref` handoff snapshot:

- original paths:
  - `vllm/model_executor/kernels/mhc/tilelang.py:mhc_pre_broadcast_tilelang`
  - `vllm/model_executor/kernels/mhc/tilelang_kernels.py:mhc_pre_big_fuse_broadcast_with_norm_tilelang`
  - `vllm/models/deepseek_v4/nvidia/model.py:finalize_mhc_broadcast_weights`
- snapshot base commit: `a9b39d6dcbe9970a33b5d1ceb9848ba975167b1b`
- source file SHA256:
  - wrapper: `c8ce81539c779436eb2b9fbba84738f3114bf3ac0a149f2b606f7d38bd1067ce`
  - TileLang kernel: `03aeb3f7c0eabfe8391006ff80f8d4cfedba7cf3d0a168e0dee2d678d3e3bd24`
  - model integration: `d9573aa0a926cce853c856cc81cb991b02388f55c9d98f1fd4cfe915fc234a6a`
- license: Apache-2.0

The TileLang function body is AST-identical to the reference. Plugin changes
are limited to ownership-safe imports, explicit NVIDIA/DeepGEMM dispatch, and a
generic materialized-broadcast fallback when the specialization is unavailable.
