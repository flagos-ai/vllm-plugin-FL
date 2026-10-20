# FlagGems-vllm operators

Compatible, explicit FlagGems operator imports use FlagGems-vllm directly.
No environment variable or opt-in is required. `flag_gems.enable()` continues
to own ATen registration. The
native vLLM 0.28 runner and graph mode are retained. NVIDIA CI uses FlagGems
v5.4.0 at commit `6ed2f39071ef85db26b7e64f0652a162f434959c` and retains the
baseline NVIDIA blacklist, without the migration's temporary `mm` exclusion.

## Install in the validated NVIDIA image

Start with the image and runtime checks in the [NVIDIA guide](usage/nvidia.md):

```text
harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-cuda-ci
```

In a container with this checkout mounted at `/workspace/vllm-plugin-FL`:

```bash
cd /workspace/vllm-plugin-FL
bash .github/scripts/cuda/setup.sh
```

The installer pins FlagGems-vllm **v0.2.0**, commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`. The release still reports package
metadata `0.1.0`; the installer checks the Git repository and commit in
`direct_url.json`. It installs with `--no-deps --no-build-isolation` and
checks that the existing Torch, FlagTree/Triton, FlagGems, vLLM, and NumPy
versions are preserved by that installer. NVIDIA CI first upgrades only
FlagGems to tag `v5.4.0` with `--no-deps --no-build-isolation` and verifies its
Git provenance; the remaining image runtime packages are preserved.

The pinned FlagGems-vllm package is required for the migrated providers.

## Confirm activation

The worker logs the following once per process after successfully importing
the package during FlagGems backend registration:

```text
FlagGems-vllm integration activated: package=...
```

This confirms provider activation. Per-operator dispatch policy can still
select a vendor or reference implementation. Package installation or an
export lookup alone does not prove that a model executed a particular kernel.

## Migration scope

Compatible activation, attention/cache, MLA, RoPE, MoE and fused-experts calls
use existing FlagGems-vllm exports with the existing arguments. This migration
adds no state-layout converters or composite fallback operators.

Dynamic per-token INT8 quantization retains its baseline dispatch chain:
FlagGems-vllm `scaled_int8_quant`, then the in-tree Triton implementation,
then the PyTorch reference if dispatch falls back after an error. The local
Triton kernel and the existing FlagGems-vllm entry are unchanged; neither is
part of this interface migration.

The migration changes the provider of existing explicit operator calls. It does
not enable OOT activations, change `compilation_config.custom_ops`, or add
custom compilation boundaries. Upstream vLLM and the existing dispatch policy
continue to decide which operator entry points execute. Native activation
formulas in graph mode are not forced through FlagGems-vllm.

See [interfaces retained from FlagGems](flaggems-vllm-unsupported-ops.md) for
missing or incompatible APIs. Device detection, runtime helpers, and the
validated Ascend vendor paths retain their existing implementations.

## CI and local verification

Every platform's setup installs the pinned package. Platform CI enable/disable
settings are preserved; a disabled platform remains disabled. Only NVIDIA CI
updates FlagGems to v5.4.0; other platforms retain their vendor runtime versions.
CUDA E2E executes each original model/task/device group once with the direct
FlagGems-vllm providers.

Run the same tests without provider-selection environment variables:

```bash
python3 tests/run.py --platform cuda --scope unit
python3 tests/run.py --platform cuda --scope functional
```

For real models, run the [adaptation gate](../tools/adaptation-gate-cases/README.md)
in eager and graph modes. NVIDIA validation does not establish correctness on
another vendor's runtime.
