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
| QSA reference math | `nvidia/ops/qsa.py` | `c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f` | `vendor/official/qsa.py` | retained reference source; not selected by the current runtime |
| QSA runtime math | reference QSA semantics plus local kernels | runtime source hash from `qsa_runtime_status()` | `gpu/ops/qsa.py`, `gpu/indexer_qsa.py`, `gpu/qsa.py` | local metadata, MQA, stable TopK, sparse GQA, compression and cache-write composition; device/shape/capture gates select implementations within these entry points |
| QSA pre-indexer reference | `nvidia/ops/qsa_pre_indexer.py` | `e93fd5f12a101ffd68b4959eccb3e127f737413a5b422819bee3f49e595e6477` | `vendor/official/qsa_pre_indexer.py` | reference-only; the local indexer does not call this fused vendor kernel |
| QSA cache metadata | `common/qsa_cache.py` | `e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8` | `common/qsa_cache.py` and `gpu/ops/qsa.py` | local v0.24 owner/metadata adaptation, shared compressed metadata producer and local graph-safe Triton preparation |
| PLE math/helper | `common/ple.py` | `ab0d4075367c4a8526eed2bdbfe1c5ea3e8accfda84d5f50b8263b771f5613aa` | `common/ple.py` | exact Apache-2.0 helper vendor; `gpu/ple_layer.py` is a v0.24 adapter that keeps native eager state I/O and graph-safe local state kernels |
| PLE layer | `nvidia/ple_layer.py` | `a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7` | `gpu/ple_layer.py` | adapted to v0.24 Mamba state/Conv ABI; newer FP8/offload-only APIs are not imported |
| GDN | `qwen_gdn_linear_attn.py` | `81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0` | `patches/gdn_packed_decode.py` | the snapshot has a Python selector but no actual C++/CUDA source/object for fused decode; it is **not claimed as vendor**. v0.24 packed recurrent Triton remains, with the existing FP32-beta semantic patch and explicit risk |

The adapted files continue to change after the initial import. Obtain their
current hashes from the checked-out source and deployment manifest, not an old
static table. `QSAIndexer.runtime_status()` records the exact compression
callable selected by the same shape/feature gate used in execution, plus the
local metadata, MQA, TopK, selection, attention and cache-store entry points.
Each identity contains the callable name, source path and SHA-256. The no-instance
`qsa_runtime_status()` lists both compression candidates and does not claim a
shape-selected branch. These reports describe selection, not proof of a GPU
launch; backend gates inside a callable still require trace/dispatch evidence.

The compatibility API `qwen4_qsa_pre_indexer_status()` reports `enabled=False`
for the vendor pre-indexer and includes the local runtime identities. No setting
in this adapter enables the vendor QSA reference modules. HC retains its separate
explicit opt-in.

The three files in `vendor/official/` have local hashes
`f558029d79d38ea927b7f3e7750f7ffa72566ec60f1f4b68752e5c93c274d262`,
`dfe8661a53764ba0449a7537c1303fddb4db094959b7f87f9d41cbd64928f3b2`, and
`3e2bce15ebc1d7685f17010ae1b88344769bfc95b121c0674f0f14a2143a8912` for
`hc.py`, `qsa.py`, and `qsa_pre_indexer.py`, respectively.  Their only
source-level additions at import time were provenance headers. QSA modules are
reference-only in the current runtime; HC remains opt-in. Verify deployment
hashes separately rather than treating these import-time hashes as live evidence.

## Layer boundaries

`vendor/official/` is source/math provenance.  `vendor/vllm024/` owns only
the ABI gate and dispatch decision.  The model `common/` and `gpu/` files own
v0.24 cache/metadata/layout adaptation and semantic correctness fixes.  No
module imports the official image's installed newer `vllm` package.

The common attention graph remains outside this model's QSA ownership. QSA uses
local metadata preparation and local compression/indexer/attention composition;
`common/qsa_cache.py` provides the v0.24 ABI adaptation. Invalid
request/block/position indices are explicitly marked invalid and may use a safe
zero sentinel for a masked gather; masked results are never valid cache slots.

## Optimization ownership

The local QSA Triton kernels and existing PLE state/fusion kernels need FlagGems
ownership for device portability, numerical contracts and kernel optimization.
Metadata composition, graph scheduling and dispatch need FlagTree ownership.
This review iteration adds no device kernel: PLE hashing now composes existing
PyTorch operations over flattened token windows. Its reference/graph tests cover
EOS boundaries, request reorder, chunking and graph padding. Portability beyond
actually tested platforms remains unverified.
