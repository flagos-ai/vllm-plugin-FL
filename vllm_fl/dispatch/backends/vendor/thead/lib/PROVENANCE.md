# T-Head PPU native extension provenance

The optional native bundle is not tracked in Git and is not included in the
Python wheel. A deployment that opts into these kernels must provision all
three files in one directory and set the absolute path in
`VLLM_FL_THEAD_NATIVE_LIB_DIR`. Without a complete bundle, the plugin emits one
warning per process and continues through its configured FlagGems/reference
fallbacks. ABI or dependency load failures remain fatal.

The bundle validated for the environment below was copied byte-for-byte from:

- image tag: `egslingjun-registry.cn-wulanchabu.cr.aliyuncs.com/egslingjun/inference-xpu-pytorch:26.04-v2.1.0-vllm0.23.0-torch2.10-cu130-20260710`
- immutable digest: `sha256:fb48fe779ca7b91741b0248cf6cda4ab3785c69dd91dfe88890c7cafdc2996ac`
- source directory: `/usr/local/lib/python3.12/site-packages/vllm/`
- source vLLM: `0.23.0+v0.2.0.ppu2.1.0`
- source Torch: `2.10.0`
- source PPU SDK: `2.1.0`

| File | Bytes | SHA-256 | Intended use |
|---|---:|---|---|
| `_C.abi3.so` | 143446208 | `10054e23f66011437ad1e33078ad67832393585d368e877448b4976952243d90` | Indexer top-k and dynamic INT8 quantization; future verified core kernels |
| `_C_stable_libtorch.abi3.so` | 130695032 | `286f17d0ee7144d695d005c8b6f3c9003ef2ac83c5ff22cb1abef6184b904bab` | MLA query concat and BF16 paged-cache insertion |
| `_moe_C.abi3.so` | 165806288 | `8fb2361800d359b5d4c287c1beb6b02337dd66092a249909a21f79238be0b172` | Candidate native MoE routing, permutation, reduction and expert kernels |

On 2026-09-03 all three libraries loaded during early PPU platform registration
in the target `hy4` container (vLLM 0.24, Torch 2.10, PPU SDK 2.1.0). Loading
before `_C_ops_registry` is mandatory because late loading duplicates fallback
schemas. Focused PPU
tests verified eager and graph replay for dynamic INT8 quantization, decode
top-k, `moe_sum`, `topk_softmax`, `concat_mla_q`, and
`concat_and_cache_mla`. An operator is not enabled merely because its binary is
present: every wrapper still requires contract-specific numerical and graph
coverage before registration.
