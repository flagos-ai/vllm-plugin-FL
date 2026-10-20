# Changes against pinned upstream sources

Comparison bases are public revisions, not an older MiniMax PR:

- FL main: a696859b9aa528dfd5b21f5da3403fdc16bac4cd (vLLM 0.24 line).
- vLLM: ee0da84ab9e04ac7610e28580af62c365e898389.
- Verified library bases: FlagGems a7620cc191a0b42e040194622c5758b22a7a25dc; FGV b5c2c2249319277ccda4d41aa0f65164bd9b2a19.
- Performance qualification also used unmodified current FlagGems 860538c69bfb6457d3c2eb858feca35c352a00cf and FGV fa78ddee4c3e78ce5047213ff0ef02acff34b2d5.

## Framework differences

| Area | Upstream reused | Actual FL difference |
|---|---|---|
| MoE pipeline | Shared FL backend selection and TritonExpertsFL.apply | Explicit per-call context/assignment/activation hooks with unchanged defaults; local M3 subclass supplies them |
| Construction | vLLM FusedMoE and RoutedExperts constructors | Router/expert constructor arguments and local quant-method subclass; no copied constructor or global factory replacement |
| Model | vLLM M3 text/vision classes | Changed norm/QKV/MoE/vision methods and named model-scoped bindings only |
| Vision RoPE | Shared FL #560 dispatch and FGV rotary | Direct reuse; private M3 RoPE implementation/schema/attention override stay removed |
| Warmup | Native vLLM kernel_warmup and M3 warmup | Direct reuse |
| Linear | Pinned FG Linear kernel plus public GEMV/traversal primitives | M3 shape tables, explicit launch metadata and scoped ATen registrar |
| MSA | Native vLLM JIT computations | TP16/head layout guards and explicit top-k/head/stage launch settings; custom companion kernels retain accepted packed/split math |
| ATen registration | FlagGems public registrar API | Copy only the incoming registration rows to select local Linear policy and generic masked_fill_; no export/default/private tuner mutation |
| Vendor | Existing FL MetaX backend | vLLM 0.24 ABI additions and retained communication boundary, installed only by the M3 profile |

## Upstream kernels and compiler

Source origin and compiler are separate. Native index/sparse JIT kernels are
imported directly from vLLM and compile through FlagTree 0.6.1+metax3.6.
Startup checks the package version and actual triton module path. There is no
runtime copy into FGV, source exec, string replacement or fake module facade.

The Python wrappers in native_index.py/native_sparse.py are retained because
vLLM 0.24 does not expose their MetaX launch parameters per call. They unwrap
read-only JIT handles and pass explicit constants; they never mutate upstream
autotuners or function defaults. All unchanged numerical bodies are imported.

The new FL computation files contain the TP16 companion work: one index-vector/
dot implementation, one packed-prefill template (full or selected-page traversal),
one decode template (page_parts=1 or split pages), and routed expert GEMV.
Base and long wrappers share these templates. The changed traversal, query/head
layout and workspace contracts prevent direct reuse of the unmodified native
attention kernel. Their accepted arithmetic and shape guards remain intact.

## Operator source PRs and pinned delivery

| Work | Source PR | Delivery / classification |
|---|---|---|
| BF16 Linear GEMV / traversal | [FG #7320](https://github.com/flagos-ai/FlagGems/pull/7320) | Qualified pure-operator performance; FL integration backports primitives only, retaining its accepted model tables |
| BF16-load FP32 router | [FGV #960](https://github.com/flagos-ai/FlagGems-vllm/pull/960) | Functional extension for M8192 and measured benefit vs valid complete cast-plus-projection calls |
| Explicit numerical modes | [FGV #961](https://github.com/flagos-ai/FlagGems-vllm/pull/961) | Compatibility: Gemma residual, division-form clamped activation, staged BF16 QKV and sequential expert reduction |
| Independent FGV cache | [FGV #962](https://github.com/flagos-ai/FlagGems-vllm/pull/962) | Compatibility; existing cache environment remains the fallback |
| Existing MoE GEMM / alignment / top-k / rotary | Pinned FGV source | Reuse after comparison; no new performance PR for namespace migration |

The source PRs are not merged yet. dependency-lock.json and requirements.txt pin
public integration commits on the verified bases. Each integration repository
contains a short MINIMAX_INTEGRATION.md recording the selective backports.
No full-library upgrade is implied.

## Removed delivery mechanisms

The seven dependency patches, patch installer, source-import manifest and private
M3 library policy package are no longer deployment requirements. Examples contain
only configuration, launch code, dependency declarations and documentation.
The two copied native MSA source trees are replaced by normal imports. Duplicate
prefill/decode kernel bodies share the existing parameterized templates. Vendor
ABI additions live in FL's vendor backend; the model-specific sdk directory is
removed.

## Validation of this reduction

100 operator fixtures passed against FP32 references. Direct comparisons with
the accepted fixture outputs match except the old standalone router fixtures
used native Torch sigmoid: the actual service provider comparison separately
checks registered FlagGems sigmoid and matches routing IDs and weights exactly
across random, tied and nearly tied scores.

27 MSA fixtures cover page/query boundaries and contexts up to 131072; three
changing-input graph checks pass. 26 full MoE chains match the accepted
implementation bitwise, including graph replay, with timing controls around
1.00x. These reuse checks are not presented as new speedups.

Library PR evidence is separate: complete-call eager/Graph timings on C550 with
fixed dtype/compiler/input, correctness before timing, seven alternating paired
rounds and explicit fallback controls. Model accuracy/performance tables in the
README are retained measurements. The installed-entry TP16 smoke passed 12/12 fixed-answer cases, including 64K
mixed requests and an image. All 16 ranks passed capture/binding audits; there
were no runtime errors or illegal accesses. This is a functional regression, not
a new full quality evaluation or throughput benchmark.

The shared worker exposes an opt-out for generic global MoE factory/oracle
hooks. MiniMax supplies local factory classes; other models retain the default.
Numerical-mode overlap checks allow disjoint workspace byte spans while rejecting
real overlap. Both boundaries have dedicated regressions.
