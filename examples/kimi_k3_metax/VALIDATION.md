# Validation of the K3 integration

## TP32 residual removal

The follow-up cleanup removes four historical MLA files for H3/page256,
the H3 host/PV path in the shared long-context module, and H3-only prefill
eligibility/launch selection. The current K3 policy already requires TP8;
96 attention heads therefore yield 12 local heads.

The TP8 decode/backend, PP communication, worker/workspace, expert, KDA and
linear implementations, launch configuration and dependency pins are unchanged.
The shared QK/merge kernels remain byte-identical function bodies.
The generic dense fallback retains its physical-stride guard and call arguments.

Validation: 57 installed-package checks pass, including TP8 fallback/layout
and FGV callback forwarding contracts. All 56 retained local JIT function ASTs
across the FL tree match the preceding commit (34 are K3-local); 11 unreachable
H3 JIT functions are removed. H12 prefill launch equivalence was checked after
folding the old H3-only condition. No dangling import of a deleted module remains.

This cleanup did not run a new model smoke or performance/quality benchmark:
it removes paths excluded by the existing TP8 contract. The four-node Graph
smoke below belongs to the preceding structural-refactor commit.


## Current structural refactor

The runtime arithmetic, explicit launch settings and tuning candidate sets, guards, quantization/rounding and
communication order remain those of the accepted K3 path. No operator library
commit or native serving parameter was upgraded.

- 170 function/method AST comparisons are unchanged, including all 45 local
  Triton JIT functions; 61 module/class assignments retain the explicit launch
  tables, precision settings and workspace constants. The extracted small-projection launch body is also
  identical; dispatch-only call-target changes are checked separately.
- 52 unit/contract tests pass in the pinned MetaX runtime. They cover model
  policy, explicit dispatch profile, native callable identity, blacklist
  behavior, canonical FG imports, PP queue/lifetime handling and SDK reuse.
  Some native imports query accelerator capabilities, so these tests need a
  matching runtime with an idle visible GPU even though they do not load a model.
- 328 installed FL source/config files match the candidate. Another 6,681
  installed vLLM/FG/FGV source/config files match the fixed public revisions;
  only generated version metadata is additional. A final SDK-file EOF cleanup
  is separately confirmed bytecode-identical under Python 3.12.
- CUDA/MCCL wrapper checks, idempotent general-plugin registration and real
  UVA/weak-reference providers pass. K3 imports no forbidden vendor framework.
- Four-node TP8/PP4 Graph smoke passes **18/18** generation requests: short,
  4K/16K/64K text, red/blue images, repeated-query consistency, mixed semantic
  requests and eight concurrent exact-length outputs. All logprobs are finite.
  There are 18 unique response IDs, 99,936 prompt tokens and 208 output tokens;
  client and server accounting agree, with zero errors or preemptions.
- A test-only native worker extension observes CPU function boundaries without
  replacing compute methods or adding GPU operations. All 32 ranks execute
  **94 native FULL replays** (56 at padded size1, 38 at size4), 22 long eager
  prefill steps and 38 mixed-request FULL steps. Other Graph sizes are not
  claimed to have been replayed by this short smoke.
- Every rank retains one 176 MiB prefill arena. The service reports 1,207,448
  effective KV tokens, no allocator OOM, prefix caching off and Mamba mode none.
  All four temporary model containers exit normally with code0.
- Changed example/test files pass the repository's Ruff checks. A stricter
  optional name scan finds only the same existing legacy SDK diagnostic present
  in the target base; K3 uses its separately checked constructor.

Reproduce the contract checks after the pinned installation, selecting an idle
GPU through the runtime's usual visibility setting:

```sh
VLLM_FL_CONFIG="$PWD/examples/kimi_k3_metax/dispatch.yaml" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest --import-mode=importlib --confcutdir=tests/k3 -q tests/k3
```

The model smoke uses fresh private caches. Cold JIT/autotuning messages are
retained; no latency or throughput claim is made from this correctness run.
No GPU profile, full GPQA/ERQA evaluation or formal serving benchmark is repeated.
The historical results below and in README remain explicitly historical.


## Prior ownership-cleanup checks (retained)

- 110 source/default-path equivalence checks; relocated function bodies and launch settings retained.
- 29 CPU policy/launcher/PP/provenance tests, including the native root-parser regression for disabled prefix caching.
- Eight real import entries; the retired FGV K3 namespace is not loaded.
- Complete TP8 expert boundary: 12 shapes from M1 to M4096 exactly match the accepted implementation. Five Graph shapes with three changing inputs each also match exactly. Paired median wall-time differences are within about 0.5%; this is component evidence, not serving throughput.
- One reserved prefill arena per registered stream, 176 MiB, unchanged.
- Interleaved/contiguous MLA fallback: 32 output/LSE exact records, including changing-input Graphs.
- The model-selected TopK provider matches accepted weights/IDs for 24 eager records and 12 changing-input Graph replays. Registration respects the existing operator configuration and invalidates cached dispatch through public manager methods.
- Compatibility checks: 48 addmm cases and four MLA callback contracts. The native C550 addmm compilation failure is retained as negative evidence.
- vLLM Ruff 0.14.0, six repository local hooks and Mypy 1.20.2 for Python 3.12 pass. Full pre-commit bootstrap timed out fetching Git dependencies; it is not reported as passed.

The clean installed-package TP8/PP4 smoke passed **18/18** requests: 1K/4K/16K/64K
text, red/blue images, mixed concurrency, repeated-query consistency and eight
exact-length concurrent outputs. All logprobs were finite; 18 distinct API
response IDs and server increments match 99,936 prompt and 208 output tokens.
Prefix caching is disabled, Mamba cache mode is `none`, prefix hits and
preemptions are zero, and the final queue is idle. All 32 rank identities and
single 176 MiB arenas match; all four model containers exited with code 0.

The negative prefix-cache option is explicit in YAML: vLLM's root CLI parser
cannot resolve a subcommand's negative boolean action while converting a plain
`false`. The first smoke passed its functional checks but used prefix caching;
it is retained separately and is not substituted for the corrected-profile run.
The runtime package is the installed `4d746d70` source; the corrected entry
configuration is the one in this PR. No inference kernel changed during this
configuration repair.

Native cold autotuning rejected some over-budget shared-memory candidates and
selected supported configurations. These handled compile trials are distinct
from allocator OOM: no allocator-OOM evidence or fatal worker error was observed.
No new full quality evaluation or serving performance benchmark was run.

## Historical results retained

The unified backend-lowering implementation passed 146 four-node correctness
requests, including text/image and mixed-length inputs. The preceding published
entry point passed its fixed 18-request smoke. GPQA and ERQA scores in README
come from the earlier full evaluations. They are not new scores of this package.

The public operator PRs contain their own native references, scope, test commands
and standalone measurements. Source equivalence alone does not prove a new
installed entry point works; the installation smoke above is tracked separately.
