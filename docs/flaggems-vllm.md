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
bash .github/scripts/nvidia/setup.sh
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

## CosyVoice3-0.5B on MUSA

The `cosyvoice3` extra pins the FlagGems-vllm revision for this integration
review. Use the standard vLLM V1 engine with `VLLM_PLUGINS=fl`. Set
`flagos_profile` to `"cosyvoice3-0.5b"` on the exported Hugging Face config.
For existing exports, pass
`additional_config={"flagos_profile": "cosyvoice3-0.5b"}` to the engine.
The profile checks the export shape, prompt-embedding support, BF16 and
TP/PP/DP of one on an MTT S5000 with vLLM 0.28.0 and
Torch 2.11.0.post2+musa5.2.0.

For the matching configuration, `FULL_AND_PIECEWISE` is the default when
no graph mode is specified. Explicit `NONE`, `PIECEWISE` and
`--enforce-eager` remain available. The model runner checks capture support
for the selected Attention backend. Other models keep the platform defaults.

The plugin binds the fused sampler to the matching engine's sampler instance.
Set `flagos_fused_topk=False` in `additional_config` to disable fusion for that
engine. The FlagGems operator policy still applies. The private V1 EngineCore
can load the optional provider. A shared in-process engine can reuse an already
loaded provider and otherwise uses native sampling. Missing Top-K APIs retain
native sampling before any random numbers are drawn.

The fused path accepts contiguous FP32 MUSA logits, vocabulary sizes up to
8192 and normalized per-row Top-K values. Native sampling handles active
Top-P, per-request generators, FP64, processed logprobs and other layouts.
