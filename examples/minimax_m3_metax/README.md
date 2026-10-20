# MiniMax M3 BF16 on MetaX C550

This example enables the MiniMax M3 optimizations in the installed FL package.
Model/TP orchestration lives in FL; pure operator changes are normal source PRs
in FlagGems and FlagGems-vllm, pinned here by public integration commits. Other model profiles keep the
normal FL entry points and dispatch policy.

## Environment

The target is two nodes, each with 8 × MetaX C550 (64 GiB), using TP16 and the
original BF16 checkpoint. The serving configuration uses 32 CPU cores per node.

| Component | Version |
|---|---|
| Python | 3.12.13 |
| MACA SDK/runtime | 3.8.1.3 |
| Host KMD | 3.8.1 |
| MetaX Torch | 2.10.0+metax3.8.1.0 |
| Triton API / FlagTree | 3.6.0 / 0.6.1+metax3.6 |
| vLLM | 0.24.0, public commit ee0da84ab9e04ac7610e28580af62c365e898389 |
| FL branch | main, with this PR |
| FlagGems | validated a7620cc1 base + [c7a255b6 integration](https://github.com/cshujie131-bot/FlagGems/commit/c7a255b6bb24d3e5b3c1d40aa392e590d84c29d6) |
| FlagGems-vllm | validated b5c2c224 base + [ff12e0bf integration](https://github.com/cshujie131-bot/FlagGems-vllm/commit/ff12e0bfa9e124827739ae2e4ce7d7592c7e65b1) |
| Vendor FlashAttention | 2.6.3+metax3.8.1.0torch2.10 |

The FlagGems-vllm release reports 0.1.0 in package metadata; the public commit is
the source identity. Do not infer its revision from that metadata string.

Use the matching vendor Torch, MACA/MCCL, FlagTree and FlashAttention binaries.
The code here does not build or redistribute those components and is not a
source-only recipe for a complete vendor image. Keep the matching host verbs
stack and Broadcom RDMA provider available. Vendor GEMMs and dense attention
remain where required; the vllm_metax framework is not loaded. FlagCX is not used.

## Prepare dependencies and install

Run in an isolated vendor environment on both nodes. Keep its Torch, compiler,
NumPy and SDK versions; the commands below do not resolve or upgrade dependencies.

~~~sh
git clone https://github.com/vllm-project/vllm.git /work/vllm
git -C /work/vllm checkout ee0da84ab9e04ac7610e28580af62c365e898389
VLLM_TARGET_DEVICE=empty python -m pip install --no-deps --no-build-isolation /work/vllm

python -m pip install --no-deps --no-build-isolation \
  -r examples/minimax_m3_metax/requirements.txt

python -m pip install --no-deps --no-build-isolation -e .
~~~

The dependency lock records public commits and their source PRs. These are
selective backports onto the verified library bases, not upgrades to the latest
whole library. [FG #7320](https://github.com/flagos-ai/FlagGems/pull/7320) and
[FGV #960](https://github.com/flagos-ai/FlagGems-vllm/pull/960) contain the qualified
Linear/router improvements; [FGV #961](https://github.com/flagos-ai/FlagGems-vllm/pull/961)
contains explicit numerical modes and [FGV #962](https://github.com/flagos-ai/FlagGems-vllm/pull/962)
contains independent cache selection. They are not merged yet.

No source patch, snapshot or patch installer is required. The launcher checks the
installed VCS commit recorded by pip, or the commit and clean state of a retained
local Git checkout for offline installs. It rejects a different source tree
shadowing the installed package. It selects FlagTree explicitly, and starts
Python with `-P` so the working directory cannot accidentally shadow installed
packages. Build requirements for the public vLLM source include setuptools-rust
and semantic-version; install them in the isolated build environment without
replacing the vendor Torch. The Python-only empty-device build is used here; it
does not provide vendor binaries. Old images with modified vLLM Python files must
install the clean source above. Version/source checks reject a mismatched ABI.

Use the BF16 checkpoint from
[ModelScope MiniMax/MiniMax-M3](https://www.modelscope.cn/models/MiniMax/MiniMax-M3).
The two nodes must see the same complete checkpoint and tokenizer snapshot:

~~~sh
modelscope download --model MiniMax/MiniMax-M3 --local_dir /models/MiniMax-M3
~~~

Mount weights read-only. Do not substitute a quantized checkpoint or reduce TP to
fit one node. With containers, expose /dev/dri, /dev/mxcd and /dev/infiniband,
use host networking/IPC and allow locked memory. Set the vendor library search
paths for the matching MCCL and verbs provider before starting Python.

## Launch

config.json is the single model/optimization configuration, passed through
VLLM_FL_CONFIG. It includes the accepted Linear tables, MoE launch policy,
attention shape rules, dispatch policy and standard runtime/communication
variables. The launcher clears obsolete optimization switches before starting a
fresh Python process.

The imported vLLM kernels are compiled with FlagTree; source reuse does not
change their compiler or switch them to native binary operators.
Supply the installed FlagTree root containing triton/; it is used as the compiler
import prefix. The example prints its command by default. Add --execute only when
the GPUs and ports are available.

~~~sh
# Node 0
python examples/minimax_m3_metax/serve.py \
  --model /models/MiniMax-M3 --flagtree-root "$FLAGTREE_ROOT" \
  --cache-dir /work/m3-cache --node-rank 0 --master-addr "$MASTER_ADDR" \
  --port 18050 --master-port 29750 --execute

# Node 1
python examples/minimax_m3_metax/serve.py \
  --model /models/MiniMax-M3 --flagtree-root "$FLAGTREE_ROOT" \
  --cache-dir /work/m3-cache --node-rank 1 --master-addr "$MASTER_ADDR" \
  --port 18050 --master-port 29750 --execute
~~~

Each node gets separate Triton, Inductor, vLLM, FG and FGV caches. An incompatible
cache identity or an unknown nonempty directory is rejected; nothing is deleted
and no existing service is stopped. No compiler caches or timing databases are
distributed. Fresh caches require warmup/compilation.

Defaults: TP16, BF16 main/index KV, block128, maxlen131072, maxseq64, MBT8192,
gpu-memory-utilization0.96, standard vLLM Scheduler, watermark0, no prefix cache
and no async scheduling. The accepted FULL_AND_PIECEWISE graph configuration is
retained, with one capture warmup; the model's Inductor compilation mode is NONE.
There is no custom KV admission scheduler in this profile.

The communication example uses four MCCL channels, P2P cap4 and a 1MiB buffer.
tuning-ar512k.cfg retains the tested message-size rule. Set the socket interfaces,
HCA names and GID index in config.json for the actual RDMA fabric. An ACTIVE port
alone does not establish that traffic is using it: check MCCL NET/IB/GDRDMA logs
and port counters during the smoke.

## Review map

[Changes against pinned upstream sources](UPSTREAM_DIFF.md) identifies reused
code, direct upstream imports and the actual M3 deltas. It does not use an older PR as
the comparison baseline.

| Area | Main files / public library PR | What to review |
|---|---|---|
| Configuration/startup | vllm_fl/minimax_m3_config.py; patches/minimax_m3_metax/ | Typed immutable policy, effective blacklists, scoped initialization and cache identity |
| Model | vllm_fl/models/minimax_m3/ | Changed methods and explicit M3 bindings; loading and unchanged model code remain in vLLM |
| MSA | ops/minimax_m3/attention.py; *_policy.py; native_index.py; native_sparse.py | TP/owner checks and explicit launch arguments; unchanged native JIT bodies are imported from vLLM |
| TP16 companion kernels | ops/minimax_m3/index_score.py; long_prefill.py; long_decode.py; moe_gemv.py | Query/head packing, selected-page traversal, page splitting and routed GEMV; base and long paths share templates |
| Linear | ops/minimax_m3/linear.py; aten.py; FG #7320 | FL owns all 35 legacy / 28 small-M rules and ordered shapes; FG owns the reused GEMM and GEMV computation |
| Gate and MoE | ops/minimax_m3/router.py; moe.py; experts.py; library_bindings.py; FGV #960 | Shared expert pipeline and FGV GEMMs/alignment/reduction; no duplicate backend selector |
| Norm/activation/QKV | ops/minimax_m3/numerics.py; FGV #961 | Explicit Gemma, division-form SwiGLU, BF16 staged norm/RoPE and sequential reduction modes |
| Vision | ops/minimax_m3/vision.py; vision_chunk.py | Shared partial-RoPE dispatch from #560, patch chunking and retained FlagGems varlen call |
| In-place mask | ops/minimax_m3/aten.py | Public registrar selects generic masked_fill_; library exports/defaults stay intact; masked_fill stays excluded |
| Vendor/communication | dispatch/backends/vendor/metax/; patches/minimax_m3_metax/compat.py; tuning-ar512k.cfg | Existing vendor backend plus scoped vLLM 0.24 ABI hooks and accepted MCCL transport |

Independent numerical kernels receive explicit tensors and metadata. FGV does
not infer MiniMax TP groups or import FL. Namespace migration alone is not a
performance claim. MoE alignment uses the existing no-TLE four-stage interface;
MoE GEMM, ordered sum and grouped top-k reuse FGV implementations after direct
chain comparisons. Sigmoid materialization remains explicit in FL. No global
tuner, compiler setting or library export is replaced.


Vision RoPE now follows the native M3 attention class through the shared
apply_rotary_emb dispatch from #560. Its FGV implementation compiles with
FlagTree; the duplicate M3 RoPE op and model override are removed. The separate
text QKV/norm/RoPE/cache-insertion fusion is retained.

The entry has real runtime hooks for the matched Torch/vLLM/MCCL ABI. They are
centralized, guarded and idempotent, and only the explicit M3 profile installs
them. Model hooks replace only named bindings inside vLLM's M3 modules, after checking
their public source revision. General router and MoE factories stay unchanged. The M3 worker explicitly
disables the shared global MoE hooks because its local router/expert factory
already supplies that integration; the common default remains enabled.
Vendor binary access reuses the existing FL vendor backend. The remaining ABI
compatibility lives alongside it, with the M3 entry controlling installation. No installed
vLLM source is rewritten and no fake module facade is installed by this entry.

## Validation and measurements

Model evaluation uses ordinary thinking and final answers:

| Task | Accuracy |
|---|---:|
| GPQA Diamond | 181/198, 91.41% |
| ERQA | 214/400, 53.50% |

GPQA includes the five budget-limited empty final answers in its denominator;
reasoning text is not used to recover intermediate choices.

Retained formal serving measurements on two 8-C550 nodes, TP16, BF16 KV, MBT8192:
256 requests, concurrency64, O1024; one warmup plus one formal round.

| Input tokens | Output tok/s | Total tok/s |
|---:|---:|---:|
| 1024 | 683.47 | 1366.94 |
| 4096 | 349.67 | 1748.37 |
| 16384 | 125.90 | 2140.37 |
| 65536 | 39.28 | 2553.05 |

The formal run IDs are short-formal-best-20260921/run02 and run04 for 1K/4K,
and long-shape-specialization-20260920/optimized-formal/run02 and run04 for
16K/64K. These are retained measurements with Triton3.6.0, not a new benchmark of
the publication layout. The current deployment keeps FlagTree enabled.

The accepted deployment passed 104 operator checks, 27 MSA cases, 26 MoE-chain
cases, 198 service regressions and four chat/image checks. The refactored and
previous outputs matched under the established per-operator criteria.

The shared vision RoPE entry passed 25 standalone BF16 cases on C550 with
FlagTree, including three graph/eager checks. The accepted FP32-reference
tolerance is rtol=0.01 and atol=0.01; input and unrotated tails stay exact.
Old/new relative L2 was at most 9.84e-6; this is numerical agreement, not a
claim of bitwise equality or a new full-model accuracy run.

This ownership update passed 100 operator fixtures, 27 MSA cases (including
three changing-input graph checks), 26 exact MoE-chain comparisons, eight
additional shared-workspace chains, and actual
ATen registration/routing checks. Standalone library PRs have their own
correctness and complete-call timing evidence. The five template-reuse timing
controls stayed within 0.6% in Graph GPU time. The 26 CPU launch/configuration
tests and six OOT scope tests also pass. The model evaluation and serving
tables above are retained measurements; they were not rerun for this cleanup.

The installed entry is checked with clean public vLLM and pinned public library
commits. TP16 smoke passed 12/12 fixed-answer cases on two 8-C550 nodes: short text,
4K/8K/16K boundaries, mixed requests through 64K, and an image. All 11 text
answers match the accepted responses; there were no empty/truncated responses,
preemptions or runtime errors. All 16 ranks completed graph capture and binding
audits. MCCL used NET/IB/GDRDMA and all four HCA port counters grew on both nodes.


~~~sh
python tests/unit_tests/test_minimax_m3_packaging.py -v
~~~
