# FlagGems-vllm fused operators

When enabled, selected FlagGems implementations dispatch fused activation,
rotary, attention,
and MoE operators through FlagGems-vllm. General ATen
registration and RMS normalization continue to use FlagGems. Existing attention
backend selection is preserved: the default service paths use upstream Triton
or DeepSeek sparse attention unless FlagGems attention is explicitly selected.

Install the pinned release source along with the plugin:

```bash
python3 -m pip install --no-deps --force-reinstall \
  'flaggems-vllm @ git+https://github.com/flagos-ai/FlagGems-vllm.git@b5c2c2249319277ccda4d41aa0f65164bd9b2a19'
python3 -m pip install --no-build-isolation -e .
```

CI installs the v0.2.0 release commit
`b5c2c2249319277ccda4d41aa0f65164bd9b2a19`. That release still declares
`0.1.0` in its package metadata and `__version__`; verify its source commit or
pip's `direct_url.json` instead of interpreting the metadata as the release tag.
The release is https://github.com/flagos-ai/FlagGems-vllm/releases/tag/v0.2.0.

Every hardware platform's CI setup calls the shared installer with its selected
Python interpreter. It verifies the source commit in `direct_url.json` and keeps
the existing Torch, Triton, FlagGems, vLLM, and NumPy versions. NVIDIA CI alone
requires FlagGems 5.4.0. Eight active platforms (CUDA, Hygon, MetaX, MUSA,
Enflame, Iluvatar, Kunlunxin, and T-Head) set `VLLM_FL_USE_FLAGGEMS_VLLM=1`
in their test containers. Ascend retains its original disabled CI configuration
pending adaptation to the current version. CUDA model gates run the same
inference/serving cases in both `0` and `1` modes with the same dependencies.
The release kernel checks remain scoped to NVIDIA CUDA.

Only existing, compatible release operators replace direct FlagGems imports.
Sunrise attention/MLA and W8A8 scaled-mm imports use the same switch; scaled-mm
uses the existing `triton_scaled_mm` export under the caller's original name.
No new operator adapters, state transformations, or composite kernels are added.
Unsupported or incompatible imports retain their original implementations; see
[Unsupported direct replacements](flaggems-vllm-unsupported-ops.md).

FlagGems-vllm integration is disabled by default, even when the package is
installed. Set `VLLM_FL_USE_FLAGGEMS_VLLM=1` before starting vLLM to enable it.
Leaving the variable unset, or setting it to `0`, selects the existing FlagGems
fused implementations and the in-tree INT8 quantization kernel. General
`flag_gems.enable()` stays enabled by default in both modes. Restart the service
when changing this variable. The enabled path does not silently substitute
the old package when FlagGems-vllm is absent or a fused kernel fails. Existing
operator dispatch policy remains responsible for backend selection. Clamped
SwiGLU uses the fused package for alpha=1 and beta=0; other parameters use
the existing native formula because the release does not implement them.

Run the fused integration checks on CUDA:

```bash
VLLM_PLUGINS=fl VLLM_FL_USE_FLAGGEMS_VLLM=1 \
    python3 -m pytest tests/functional_tests/flaggems/test_vllm_package.py
```

Follow `tools/adaptation-gate-cases/README.md` for the full Qwen3.6 eager and
graph text, image, and mixed request matrix. This main branch targets vLLM
0.24.0; model architectures and the plugin ModelRunnerFL must be compatible
with that vLLM installation. DeepSeek-V4-Flash is text-only, so the gate
image and mixed-image scenarios do not apply to it. Run both gate text
scenarios in eager and graph modes when validating that model.

The plugin enables selected fused activation CustomOps for Inductor so default
CUDA graphs execute FlagGems-vllm kernels instead of compiling the native
activation formula. Opaque Torch operators and fake implementations preserve
full graph tracing. Explicit per-operator disables and OOT whitelist/blacklist
settings remain effective; the CUDA graph mode is unchanged.

DeepSeek-V4-Flash FP8 checkpoints require `--kv-cache-dtype fp8` with this
vLLM version. The validation used tensor parallel size 4. Add `--enforce-eager`
for eager mode and omit it for the default graph mode.

GDN migration is deferred: model GDN paths and the existing plugin adapters
remain unchanged. The pinned release has kernel-dependent state layouts;
converging that public contract belongs in the operator library before a future
plugin migration.

Standard clamped SwiGLU uses the same dispatcher and selection policy as SiLU
and GELU. Its public `custom_ops` name remains `silu_and_mul_with_clamp`; the
replacement also accepts `SiluAndMulWithClamp` as an alias. Unsupported alpha
and beta values retain the upstream native formula before fused execution.
MetaX fused FA2 receives no descales for unquantized `auto` KV cache; quantized
KV cache remains unsupported. A separate MetaX opt-in check validates this path.
When MetaX's existing Triton lacks the optional TLE API, the plugin omits the
release's `persistent_topk` vendor export from its eager operator collection.
The compatibility loader verifies the pinned release's initializer hash;
attention and all remaining vendor imports execute unchanged. Installed package
files and the platform's Triton version are not modified.

NVIDIA compatibility exclusions are loaded automatically from `nvidia.yaml`.
No deployment blacklist append or CUDA/NCCL workaround is required. Validate
both modes with the original model gates when updating FlagGems or the fused
release.
