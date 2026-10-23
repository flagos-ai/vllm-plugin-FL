# DeepSeek V4 Flash W8A8 on MetaX C550

The implementation is split between FL framework code and pinned operator
libraries. This example contains launch/configuration files, not source patches
or a patch installer.

## Where to review

| Layer | Files / responsibility |
|---|---|
| MetaX local attention | `vllm_fl/dispatch/backends/vendor/metax/impl/dsv4/attention/`: TP-local blockdiag/staged kernels, split/reduce and launch policy |
| MetaX local indexer | `vllm_fl/dispatch/backends/vendor/metax/impl/dsv4/indexer/`: logits, paged reads, row scoring and exact tree top-k |
| FL communication | `vllm_fl/attention/deepseek_v4/{headpack,row_sharding,query_sharding}.py`: head exchanges, M/2 logits gather and M/4 index gather |
| FL attention interface | `vllm_fl/attention/deepseek_v4/bridge.py`: scheduler metadata and caller output buffers; standard dispatch selects the implementation |
| Runtime compatibility | `vllm_fl/patches/deepseek_v4_metax/`: startup validation, serial execution and RMS policy |
| Library/model adapters | `vllm_fl/dispatch/backends/flaggems/impl/deepseek_v4.py`, model and W8A8 quantization modules: pinned FG/FGV calls, graph boundaries, Hash routing and MCTlass binding |
| Operator libraries | The source commits below: pure operators and separately identified compatibility changes |

See [the capability and compatibility review map](../../docs/deepseek-v4-metax.md).
TP/headpack adaptations stay in FL. Library performance PRs must preserve the
operator boundary and demonstrate correctness and standalone gains against the
corresponding native library, not merely a namespace migration or vendor comparison.

## Fixed operator dependencies

`requirements.txt` and `dependency-lock.json` pin immutable source revisions:

| Library | Deployment commit | Review |
|---|---|---|
| FlagGems | [0ebb95bd](https://github.com/cshujie131-bot/FlagGems/commit/0ebb95bd47c902de9909e653f318fc57a5e315cf) | [Diff against the validated base](https://github.com/cshujie131-bot/FlagGems/compare/3123859968915e361e8452dd796dc6b27c956324...0ebb95bd47c902de9909e653f318fc57a5e315cf) |
| FlagGems-vllm | [9ae5b26e](https://github.com/cshujie131-bot/FlagGems-vllm/commit/9ae5b26e286a91c1dc1de50662373eef0715308a) | [Diff against the validated base](https://github.com/cshujie131-bot/FlagGems-vllm/compare/5f4dc53cc31b99e83967a2b19195ab1a295aeef6...9ae5b26e286a91c1dc1de50662373eef0715308a) |

The FGV integration backports these contributions onto the validated library base,
rather than upgrading unrelated operators:

- [FGV #946](https://github.com/flagos-ai/FlagGems-vllm/pull/946): MetaX MoE alignment.
- [FGV #947](https://github.com/flagos-ai/FlagGems-vllm/pull/947): packed-page KV gather,
  with its launch/config prerequisites.
- [FGV #755](https://github.com/flagos-ai/FlagGems-vllm/pull/755): the original
  contributor's MHC pre implementation. MHC post remains unchanged.

The forks also retain numerical/runtime compatibility that is **not** claimed as
new performance work: INT8 rounding, nonfinite routing safety, the existing mask
and SiLU paths, cache separation and optional compiler/TLE import handling.
Each library's `DSV4_INTEGRATION.md` identifies these changes. Upstream merge
status and the deployment pin are separate; no install-time patching is needed.

## Environment

| Component | Required runtime |
|---|---|
| Hardware | 8 × MetaX C550, 64 GiB/card, TP8 |
| vLLM | 0.20.2; a local `+empty` build suffix is accepted |
| Torch | 2.8.0+metax3.7.0.7 |
| FlagTree | 0.5.1+metax3.0 |
| Triton | Distribution 3.0.0+metax3.7.0.7; module reports 3.1.0 |
| MCTlassEx | 0.1.1+metax3.7.0.7torch2.8 |

Use the matching vendor runtime, not a stock CUDA Torch wheel. Weights and
proprietary runtime binaries are not distributed here. No `vllm_metax` import
is required. ATen uses FlagGems; communication remains MCCL, without FlagCX.

## Install and launch

In a compatible runtime:

~~~sh
python -I -m pip install --force-reinstall --no-deps --no-build-isolation \
  -r examples/deepseek_v4_metax/requirements.txt .
python -I examples/deepseek_v4_metax/serve.py \
  --model /models/DeepSeek-V4-Flash-W8A8-INT8 \
  --cache-dir /work/dsv4-cache --port 18001 --execute
~~~

Force-reinstall prevents an existing package with the same version number from
hiding the pinned source change. The launcher verifies installed Git commit
provenance before allocating model resources. Without `--execute`, it only
prints the command.

For offline hosts, transfer complete Git repositories and install via
`git+file:///...@<the same SHA>`. Preserve Git history/provenance; a ZIP snapshot
or an arbitrary same-version wheel is not an equivalent pinned installation.

Build from a clean checkout. The Docker recipe uses a fresh source directory,
excludes old build outputs and isolates residual files from a previous FL install.

A build example is also provided:

~~~sh
docker build --build-arg BASE_IMAGE=<matching-metax-runtime> \
  -f examples/deepseek_v4_metax/Dockerfile -t dsv4:fl-pinned .
docker run --rm --network host --ipc host \
  --device /dev/dri --device /dev/mxcd \
  -v /path/to/weights:/models/DeepSeek-V4-Flash-W8A8-INT8:ro \
  -v /path/to/new-cache:/work/cache \
  dsv4:fl-pinned --model /models/DeepSeek-V4-Flash-W8A8-INT8 \
  --cache-dir /work/cache --port 18001 --execute
~~~

`config.json` is the policy source; the tiny bootstrap only invokes FL's startup
module. Defaults: FP8 KV/block256, MBT4096, maxseq64, maxlen66560, gpu-mem0.85,
disable-custom-all-reduce. `FULL_DECODE_ONLY` keeps prefill eager while retaining
decode graphs and torch.compile. `--thinking` selects maxlen131072/maxseq32.

Use a fresh cache for every changed source/configuration. Triton, Inductor, vLLM,
FG and FGV caches are separated. The launcher refuses an incompatible or
unrecognized cache and never stops another service.

## Validation

The attention/indexer boundary refactor passed a fresh installed-package TP8 smoke:
78/78 requests covering short prompts, 1K/4K/16K/20K prefill, mixed batches and
64 concurrent distinct-string responses. All seven decode graph buckets were
captured; no request failed or hit its output-token limit.

The 336 installed FL Python sources matched the tested checkout without stale
modules. Dependency pins and provider bindings were unchanged; 24 Hash-table
aliases and 24 rank/role RMS settings were checked. All 37 CPU tests passed,
including registry/profile isolation and the kernel/framework/import boundaries.

Seven files moved byte-for-byte from `3e5dcc5`; 51 function/class bodies matched
after excluding imports and import-only exception handlers. M/2 scoring was
separated from its host orchestration without changing either computation or
launch arguments. Attention kernels, dependency pins and service policies were
unchanged. The loose-file top-k import fallback was removed intentionally.
This was a functional smoke, not a new quality or throughput benchmark.
Historical accuracy and performance below are retained.

| Task | Reviewed accuracy |
|---|---:|
| GPQA Diamond | 171/198, 86.36% |
| MATH-500 | 498/500, 99.60% |

Scores use final-answer extraction and mathematical-equivalence review.

Measured serving throughput: 64 requests, concurrency64, O1024;
one warmup plus three formal rounds, last formal round:

| Input tokens | Output tok/s | Total tok/s |
|---:|---:|---:|
| 1024 | 946.18 | 1892.36 |
| 4096 | 687.18 | 3435.91 |
| 16384 | 330.72 | 5622.18 |
| 65536 | 100.31 | 6520.06 |

CPU configuration/dispatch/pinning checks:

~~~sh
python tests/unit_tests/test_deepseek_v4_packaging.py -v
python tests/unit_tests/test_deepseek_v4_layer_boundaries.py -v
~~~
