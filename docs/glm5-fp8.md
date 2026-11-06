# GLM-5.3 Flash indexer precision

The GLM indexer selects its activation and cache format from
`current_platform.supports_fp8()` before cache allocation and compiler tracing.
NVIDIA SM89+ supports native FP8; A100/SM80 and unknown capabilities select BF16.
Moore Threads MP31+ and GCU capability 4+ retain their native FP8 path.
The native DeepGEMM candidate additionally requires NVIDIA SM90+ and its ABI.

| FP8-dependent path | Platform without FP8 | Existing implementation reused |
|---|---|---|
| Query Hadamard + FP8 quantization | Hadamard BF16 output, unit scale | FlagGems-vllm `hadamard128` |
| Per-token key quantization/cache | Store BF16 keys | `qsa_store_cache_rows` |
| Prefill K-pool compression/cache | FP32 softmax/reduction, BF16 materialization and rotation/cache | PyTorch softmax/sum; `hadamard128`, `qsa_store_cache_rows` |
| Decode K-pool tail update/cache | Ordered BF16 tail writes and compression | Same existing primitives; paged tail state retained |
| Gather quantized keys/scales | Gather BF16 pages, unit scale | PyTorch gather/indexing/copy |
| Prefill FP8 MQA logits | BF16 Q/K GEMM, FP32 weighted ReLU reduction | FlagGems-vllm `router_gemm` (BF16 x BF16 -> FP32), PyTorch ReLU/sum |
| Decode FP8 paged MQA logits | Read BF16 pages, then the same BF16 GEMM/reduction | Same existing primitives, respecting page padding and valid lengths |

No device kernels are added. INT8 quantization/GEMM primitives exist, but the
pinned operator library has no complete matching INT8 K-pool/cache/MQA chain.
BF16 therefore avoids introducing an INT8 quantization, scale or cache ABI.
The public BF16 paged-MQA kernel specializes head counts 32/64, dimension 128
and dense 64-entry pages. The adapter uses the existing BF16 GEMM/reduction
for both prefill and decode, including other head counts, 32-entry pages and
padded physical strides that the specialized kernel does not cover.

The main MLA cache, KDA state, vision attention and unquantized MoE already use
BF16/FP32. FP8 checkpoint conversion is outside this change: the loader continues
to reject quantized checkpoints. The BF16 indexer covers the supported zero-RoPE,
K-pool-compressed GLM configuration; other layouts fail during validation.

The indexer cache uses 256 bytes per 128-wide BF16 key instead of 132 bytes for
FP8 plus its FP32 scale (about 1.94x before page padding). The cache spec and
profiling workspace use the selected dtype. BF16 removes quantization error but
can change near-tied TopK choices; whole-model accuracy/performance equivalence
has not been established. Composite execution may be slower than fused FP8.

CI and the GLM extra use the same FlagGems-vllm portability pin
`80a8e9716abed1bac3fc0ba5fd4e9f650ae3f989`, from operator PR #941. It includes the common main
math shim, optional constexpr/optimizer-hint compatibility, software FP8 cache
writes for HY4 on A100, and synchronized wall-clock tuning when Kunlunxin event
timing returns zero. Kunlunxin reuses the public FlagGems Hadamard transform
with the same FP32 butterfly order and existing FP32-accumulating GEMM for mHC
projection. Both mHC pre and post autotuners retain successful event timings;
kernel errors propagate without another execution. Stock Triton does not require
FlagTree-private imports or optional autotuning settings. Other vendors retain
their native autotuning API without Kunlunxin-only callback arguments. HC4 mHC pre
declares its hidden dimensions at compile time; numeric operations are unchanged.
Kunlunxin HC4 autotuning uses shape-bounded tiles, preserving other vendors and the original model-size
candidates. Kunlunxin cache stores skip invalid rows before
vector memory operations, preserving sentinel slots and physical page padding.

Kunlunxin vision attention selects existing PyTorch SDPA math operations before
execution because its FA2 compiler rejects select predicates. The adapter keeps
packed sequence boundaries on device and chunks queries to bound intermediate
memory. Other platforms retain public FlagGems-vllm FA2. This is a correctness
path with no new device kernel; its performance is not qualified.

The MetaX and Kunlunxin plain-Triton TopK paths fail during compiler lowering.
Their BF16 indexers select existing PyTorch TopK/scatter/sort operations before execution,
under the common reference policy. Row bounds, short-row padding, valid `-inf`
ties and MTP offsets are preserved; output indices use a canonical order.
This correctness path has not been performance-qualified on C550 or Kunlunxin.
