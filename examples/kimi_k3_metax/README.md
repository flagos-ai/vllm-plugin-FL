# Kimi K3 on MetaX C550

TP8 × PP4, DP1, EP/SP off, on four 8-GPU C550 nodes. Original MXFP4/E8M0
weights remain packed. Main GEMM/attention inputs are BF16; sensitive arithmetic
and recurrent state retain their accepted precision. This is not an INT8 conversion.

## Review map

Read [the file-by-file guide](../../docs/kimi-k3-metax.md) for implementation origins.

| Responsibility | Reading entry |
| --- | --- |
| Model and vision components | `vllm_fl/models/kimi_k3*.py` |
| Attention metadata, shapes and TP layout | `vllm_fl/attention/kimi_k3/` |
| vLLM projection / activation interfaces | `vllm_fl/ops/kimi_k3/` |
| Packed MXFP4 expert interface | `vllm_fl/quantization/mxfp4/kimi_k3.py` |
| Qualified local compute | `vllm_fl/dispatch/backends/vendor/metax/impl/kimi_k3/` |
| FG/FGV adapters | `vllm_fl/dispatch/backends/flaggems/impl/kimi_k3*.py` |
| Workspace, PP and Graph lifecycle | `vllm_fl/worker/kimi_k3*.py`, `vllm_fl/distributed/kimi_k3_pp.py` |
| Version / runtime hooks | `vllm_fl/patches/kimi_k3/` |
| Model-scoped backend policy | `dispatch.yaml`, through the existing FL configuration mechanism |

The refactor preserves computation, shape guards, launch settings, fallbacks and
dependency revisions. Generic backend defaults are retained; K3 differences
are explicit capabilities. Native model loading, processors and the forward
skeleton remain inherited. Existing KDA numerical compatibility, including
local FP64 accumulation, is retained without a new performance claim.

## Public source dependencies

The requirements files and `dependency-lock.json` pin complete source commits. No
source patch, source snapshot or patch installer is required at installation.

| Package | Pinned source | Review |
| --- | --- | --- |
| vLLM | [c88c603e](https://github.com/cshujie131-bot/vllm/commit/c88c603e1f3d3bde7ad18bd7299cb3d502f7b7a7) | Default-preserving component/runner interfaces; explicit non-CUDA CUSTOM MLA validation; twelve Python files, no scheduler algorithm change |
| FlagGems | [6efd6dab](https://github.com/cshujie131-bot/FlagGems/commit/6efd6dab4a4a8a78167262d0841ff0d728a2d1d3) | [Argmax #7300](https://github.com/flagos-ai/FlagGems/pull/7300); [addmm compatibility Draft #7302](https://github.com/flagos-ai/FlagGems/pull/7302) |
| FlagGems-vllm | [1d4b69b3](https://github.com/cshujie131-bot/FlagGems-vllm/commit/1d4b69b355a8a42deb0edd485ef9d9a94b2477b8) | [TopK #950](https://github.com/flagos-ai/FlagGems-vllm/pull/950), [SiTU #951](https://github.com/flagos-ai/FlagGems-vllm/pull/951), [alignment #952](https://github.com/flagos-ai/FlagGems-vllm/pull/952), [MLA interface #953](https://github.com/flagos-ai/FlagGems-vllm/pull/953) |

The reviews are unmerged at publication time. The integration commits backport
onto the validated bases, rather than upgrading unrelated operators. Each fork
contains `K3_INTEGRATION.md`. Argmax/addmm are correctness or compatibility work;
the MLA callable is an interface change. Only the three qualified operator
optimizations claim standalone gains against their corresponding native boundary.

TopK explicitly selects FP32 sigmoid → input-dtype materialization before bias
and selection; default FGV tanh-based sigmoid is not numerically interchangeable.
Alignment keeps FL's compact-buffer policy and imports the library's compact
kernel. SiTU uses the library math under the original FL shape guards. Small,
unqualified and non-normal-scale fallbacks remain available where previously used.

## Environment

| Component | Required version |
| --- | --- |
| Python | 3.12.13 |
| vLLM | 0.28.0, empty backend plus the pinned generic interfaces |
| FL target | `0.4.0-dev` |
| MACA SDK/runtime/MCCL | 3.8.1.3 |
| MetaX Torch | 2.10.0+metax3.8.1.0 |
| FlagTree / Triton API | 0.7.0+metax3.6 / 3.6.0 |
| Transformers / host driver | 5.15.1 / 3.8.1 |

Matching vendor Torch, MACA, MCCL, RDMA userspace and FlagTree compiler binaries
are required. Do not substitute a stock CUDA wheel. See the
[MetaX FlagTree instructions](https://github.com/flagos-ai/FlagTree/wiki/User-manual-for-metax).
No proprietary binary or weights are distributed here. The profile uses no
`vllm_metax`, `mcoplib` or FlagCX; MACA/MCCL are hardware runtime dependencies.

## Install and launch

Use a fresh environment or image with the matching vendor runtime. A legacy
installation with untracked source overlays should be replaced by a clean
environment. Force reinstallation prevents a same-version package from hiding
a source change. FlagGems uses build isolation to honor its `setuptools<77` requirement;
vLLM uses the matching vendor build environment with setuptools 80.10.2, which
supports its PEP 639 metadata. Do not downgrade the shared build environment.
The build-only version override identifies the empty vLLM package; Git
provenance independently checks the exact source commit.

```sh
python -I -m pip install --force-reinstall --no-deps \
  -r examples/kimi_k3_metax/requirements-flaggems.txt
VLLM_TARGET_DEVICE=empty VLLM_VERSION_OVERRIDE=0.28.0+empty \
  python -I -m pip install --force-reinstall --no-deps --no-build-isolation \
  -r examples/kimi_k3_metax/requirements.txt
VLLM_VENDOR=metax MAX_JOBS=2 \
  python -I -m pip install --force-reinstall --no-deps --no-build-isolation .
```

For offline hosts, transfer complete Git repositories and install via
`git+file:///...@<the same commit>`. Directory installs or ZIP snapshots do not
provide the required dependency Git provenance. Preserve the original environment
and use a separate source checkout; do not overlay files onto a running service.

Every node needs a verified read-only path to the original K3 weights, tokenizer,
processor and chat template. Run once per node, changing rank 0 through 3:

```sh
python -I examples/kimi_k3_metax/launch.py \
  --model /models/Kimi-K3 --node-rank 0 \
  --master-addr MASTER_IP --master-port 29864 --host 0.0.0.0 --port 8000 \
  --cache /cache/k3/run-001/node-0 --net-interface RDMA_INTERFACE \
  --ib-hca HCA0:1,HCA1:1,HCA2:1,HCA3:1 --gid-index 3
```

The launcher checks installed dependency commits and rejects import shadowing
before model allocation. `--dry-run` prints the command without loading a model.
The policy remains `K3ExecutionConfig` plus `serve.yaml`; `dispatch.yaml` selects
this profile through the existing `VLLM_FL_CONFIG` setting. No new operator
environment switches are introduced. Existing transport settings use 16 channels
and 1 MiB buffers. Use a private writable compilation-cache directory per node.

Context 69,632; GPU memory 0.95; max sequences 64; MBT4096; PP layers
24/24/24/21; eager prefill; full decode Graph sizes 1/2/4/8/12/16/24/32/64;
prefix caching off (the YAML uses the explicit `no-enable-prefix-caching` flag).
Containers require `/dev/dri`, `/dev/mxcd`,
`/dev/infiniband`, host networking, sufficient shared memory and unlimited memlock.

## Validation

Current ownership checks and installation smoke are described in
[VALIDATION.md](VALIDATION.md). Historical evaluations below are retained results
of the unified accepted arithmetic, not newly measured scores of this packaging.

| Task | Accuracy |
| --- | ---: |
| GPQA Diamond | 185/198, 93.43% |
| ERQA | 266/400, 66.50% |

Historical TP8 × PP4, MBT4096, C64, output1024 serving results. Each row retains
its successful snapshot; the long-context rows include the later memory fixes:

| Input tokens | Total tok/s | Output tok/s |
| ---: | ---: | ---: |
| 1024 | 560.057 | 280.029 |
| 4096 | 1192.596 | 238.519 |
| 16384 | 1196.806 | 70.400 |
| 65536 | 1553.939 | 23.907 |
