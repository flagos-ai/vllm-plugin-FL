# Qwen3.8-Flash-Next / Qwen4 vendor provenance

This directory records the kernel ownership used by the Day0 model commit.
The pinned source is the filesystem extraction of
`vllm/vllm-openai:qwen38-flash-next`:

* image digest: `sha256:bd995759b5b8ac51062e04c9e4d7c91c382d1ba377bb787e24dca2ccb39925e9`
* source root: `/root/qwen4/day0_work/upstream_qwen38_image_source_20260826`
* official runtime: vLLM `0.1.dev20073+g8e685d198`, PyTorch `2.13.0+cu130`
* target runtime: plugin main's vLLM `0.24.0` ABI

All official Python sources carry the upstream SPDX notices:
`Apache-2.0` and `Copyright contributors to the vLLM project`.  The local
files retain those notices.  The source hashes below are SHA-256 hashes of the
unmodified extraction; the vendored file hashes are intentionally different
when the v0.24 namespace/header adapter is applied.

| capability | official source | source SHA-256 | plugin owner | status / local change |
|---|---|---|---|---|
| HC math | `nvidia/ops/hc.py` | `4917215beeadb3265096f29c8080897c7a1af683b92ff403fdd3d0bb969f51d5` | `vendor/official/hc.py` + `common/hyperconnection.py` | kernels are vendored; `vendor/vllm024/dispatch.py` selects them only for explicit NVIDIA CUDA/Triton `QWEN4_HC_BACKEND=official`; pure-torch/current H100 path remains the explicit default fallback |
| QSA math | `nvidia/ops/qsa.py` | `c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f` | `vendor/official/qsa.py` (via `gpu/ops/qsa.py`) | official MQA/split-K/store/compression kernels are the sole runtime owner; `gpu/ops/qsa.py` is an import-only v0.24 facade |
| QSA pre-indexer | `nvidia/ops/qsa_pre_indexer.py` | `e93fd5f12a101ffd68b4959eccb3e127f737413a5b422819bee3f49e595e6477` | `vendor/official/qsa_pre_indexer.py` + `gpu/indexer_qsa.py` | official fused kernel is selected when its shape gate matches; indexer only adapts the v0.24 config/cache metadata ABI |
| QSA cache metadata | `common/qsa_cache.py` | `e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8` | `common/qsa_cache.py` | official circular-buffer metadata producer/consumer is retained; the only local code is a `CircularBufferSpec` compatibility fallback and v0.24 torch token-map fallback |
| PLE math/helper | `common/ple.py` | `ab0d4075367c4a8526eed2bdbfe1c5ea3e8accfda84d5f50b8263b771f5613aa` | `common/ple.py` | exact Apache-2.0 helper vendor; `gpu/ple_layer.py` is a v0.24 adapter that keeps native eager state I/O and graph-safe local state kernels |
| PLE layer | `nvidia/ple_layer.py` | `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `gpu/ple_layer.py` | adapted to v0.24 Mamba state/Conv ABI; newer FP8/offload-only APIs are not imported |
| GDN | `qwen_gdn_linear_attn.py` | `81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0` | `patches/gdn_packed_decode.py` | the snapshot has a Python selector but no actual C++/CUDA source/object for fused decode; it is **not claimed as vendor**. v0.24 packed recurrent Triton remains, with the existing FP32-beta semantic patch and explicit risk |

The principal adapted-file hashes in this commit are:

| local file | upstream source/hash | local SHA-256 | boundary |
|---|---|---|---|
| `common/hyperconnection.py` | `common/hyperconnection.py` / `cf2028bbd7dceb33f78393be62e3c936c29aff7255dc185f211d1e57cb007393` | `cd7cef3d899b7c185a6c91d670bfbc9562a14de7a88222334fb1abedd86d6280` | renamed/portable HC math plus explicit official-HC opt-in |
| `common/qsa_cache.py` | `common/qsa_cache.py` / `e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8` | `9a5d79cc4b4c1a4833e5c41223213e7f4ae16ec60589373a5baf8d3dd24b9418` | official metadata producer with v0.24 `CircularBufferSpec` and token-map fallbacks |
| `gpu/indexer_qsa.py` | `nvidia/indexer_qsa.py` / `e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066` | `168b644625e688fe1b33d6b28a116caa5bf145ec2f0a7ec051f67d351b6a5981` | v0.24 cache/rope/config adapter; official fused pre-indexer is selected by its shape gate |
| `gpu/ple_layer.py` | `nvidia/ple_layer.py` / `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `05df0511d8bc071a1319dbebcb1444a5ab9cd0b825dfcc96cded92d646f5da36` | v0.24 Mamba/Conv state adapter, explicit invalid-index masks |
| `gpu/ops/qsa.py` | `nvidia/ops/qsa.py` / `c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f` | `7bcd1e03c5a74b290676968a297d3a1b293d632c5f7d1eabd73f257e945e3048` | import-only v0.24 facade; no self-developed QSA math kernel remains |
| `gpu/ops/ple_state.py` | `nvidia/ple_layer.py` state path / `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `5ff4cff6791b74d3bb33c60d83404d022bc345b23991a6d1d7ba8254faa40d23` | graph-safe local stride-aware state I/O; eager native path retained |

The three files in `vendor/official/` have local hashes
`f558029d79d38ea927b7f3e7750f7ffa72566ec60f1f4b68752e5c93c274d262`,
`dfe8661a53764ba0449a7537c1303fddb4db094959b7f87f9d41cbd64928f3b2`, and
`3e2bce15ebc1d7685f17010ae1b88344769bfc95b121c0674f0f14a2143a8912` for
`hc.py`, `qsa.py`, and `qsa_pre_indexer.py`, respectively.  Their only
source-level additions are provenance headers; the QSA modules are active
through the vLLM 0.24 adapter, while HC remains opt-in.

## Layer boundaries

`vendor/official/` is source/math provenance.  `vendor/vllm024/` owns only
the ABI gate and dispatch decision.  The model `common/` and `gpu/` files own
v0.24 cache/metadata/layout adaptation and semantic correctness fixes.  No
module imports the official image's installed newer `vllm` package.

The common attention graph remains outside this model commit.  QSA itself uses
the official image's fused metadata/pre-indexer path; `common/qsa_cache.py`
keeps only the v0.24 ABI fallback for the missing `CircularBufferSpec` and
`token_to_req_indices` symbols.  Invalid request/block/position indices are
explicitly marked invalid and may use a safe zero sentinel for a masked gather;
the masked result is never exposed as a valid request or cache slot.
