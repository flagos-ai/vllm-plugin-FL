# Qwen3.8-Flash (Qwen4) Multimodal Adaptation Report

Status: **reviewed and committed locally; not pushed.** Branch `day0-qwen4`
has parent `bdeda21` (PR #455 head). Worktree:
`/root/day0-work/vllm-plugin-FL-qwen4`.

## 1. Executive judgment

The composite multimodal `*ForConditionalGeneration` class, vision tower,
processor, config and weight mapping were already present and match the
official Qwen3.8 image source. The "language-model-only false pass" is real but
is a **coverage/declaration gap, not a missing class**: the Day0 smoke
checkpoint is served text-only and no test constructed the composite path.

Two concrete defects were found and fixed:

1. `supports_multimodal_pruning` was inherited from the Qwen3.5/Qwen3VL base.
   The pinned vLLM 0.24 sets it `False`, but other 0.24 builds (e.g. the
   runnable test image) default it `True`, which would let the runner select an
   EVS pruning path this model cannot execute (`recompute_mrope_positions`
   raises). Now declared explicitly `False` on
   `Qwen3_8FlashNextForConditionalGeneration`.
2. `apply_qwen3_8_flash_next_patches()` hard-imported the packed-GDN patch
   module, whose top-level `from vllm.model_executor.layers.fla.ops.op import
   exp` aborts **all** model/processor registration on reduced/empty builds that
   omit vLLM's FLA package. The import/call is now capability-gated like
   `_register_gdn_packed_decode_patch()`.

Additionally, upstream commit `2f391861`'s padded-stride cache metadata fix was
evaluated and the applicable part was absorbed (see §4). Real H100 execution of
the full path (real processor, real vision weights, TP8 image and video
generation) passed — see §7.

## 2. Referenced sources / provenance

| Material | Path | Notes |
|---|---|---|
| Official Qwen3.8 image model source | `/root/qwen4/day0_work/upstream_qwen38_image_source_20260826/nvidia/model.py` | Composite class, processor, vision config; matches worktree |
| Official image vLLM (newer ABI) | `/root/qwen4/day0_work/upstream_qwen38_image_vllm_exact_20260826/.../vllm` | `0.1.dev20073+g8e685d198` (provenance of vendored kernels) |
| Pinned target vLLM 0.24 reference | `/root/glm5.3/vllm-v0.24.0/vllm` | Has `AttentionSpec.indexes_kv_by_block_stride` and `MLAAttentionSpec.compress_ratio` |
| Runnable test vLLM | `/vllm-workspace/vllm/.venv` | `0.1.dev1+g61d4f5663`; **lacks** `.fla`, `indexes_kv_by_block_stride`, `compress_ratio` |
| Smoke checkpoint config (real shapes) | `/root/qwen4/day0_delivery/smoke/tiny-qwen4-exp/config.json` | `Qwen4ExpForConditionalGeneration`, `qwen4_exp`, image=248/video=249/vision_start=250/vision_end=251 |
| Upstream commit under review | `2f391861ef7e42a3c2a427783739ccb0a8dbfd49` | "fix(qwen4): make QSA cache compatible with MetaX" |
| PR | `flagos-ai/vllm-plugin-FL#455` | Head equals local `bdeda21` |

No credentials, other worktrees, `opencode.jsonc`, or GPU services were touched.
GitHub was accessed over the supplied socks5h proxy for `2f391861`/PR #455.

## 3. Audit results (required items)

| Requirement | State found | Action |
|---|---|---|
| Visual encoder/projector | `self.visual = Qwen3_VisionTransformer(config.vision_config, ...)` in `gpu/model.py`; `merger`/deepstack live inside the vLLM tower; invoked via inherited `embed_multimodal` | Confirmed; covered by tests |
| Processor + config registration | `@MULTIMODAL_REGISTRY.register_processor(Qwen3VLMultiModalProcessor, info=Qwen3_8FlashNextProcessingInfo, dummy_inputs=Qwen3VLDummyInputsBuilder)`; `qwen4_exp`/`qwen4_exp_text` in `_CONFIG_REGISTRY`; `_MULTIMODAL_MODELS`/`_TEXT_GENERATION_MODELS` split | Confirmed; serving-entry test added |
| Image/video placeholders + tokens | Inherited `get_placeholder_str` returns `<|vision_start|><|image_pad|><|vision_end|>` / `...<|video_pad|>...`; token ids from composite config | Confirmed; tested |
| Multimodal embeddings | Inherited `embed_multimodal` + plugin `embed_input_ids` deepstack/merge logic | Confirmed; merge test added |
| Weight mapping | `hf_to_vllm_mapper`: `model.visual.→visual.`, `model.language_model.→language_model.model.`, `lm_head.→language_model.lm_head.` | Confirmed; tested |
| v0.24 pruning shim | `is_multimodal_pruning_enabled=False`, `video_pruning_method=None`, `video_pruning_rate=0.0` | **Fixed class-level `supports_multimodal_pruning=False`** |
| Serving entry | `vllm.platform_plugins`→`register`, `vllm.general_plugins`→`register_model`→`apply_qwen3_8_flash_next_patches()` | **Fixed empty-build gating of GDN import** |
| M-RoPE for image+text | Outer class inherits Qwen3VL `get_mrope_input_positions`; the text-only `ForCausalLM` override discards `mm_features` (correct for text) | Confirmed; image-grid test added |

## 4. Upstream commit `2f391861` evaluation and absorption

Full diff reviewed. It has three parts:

1. `common/qsa_cache.py`: `QSAStateBackend.indexes_kv_by_block_stride()` `False→True`
   and `QSAKeyStateCache.get_kv_cache_spec()` sets `indexes_kv_by_block_stride=True`
   for v0.24 cross-layer page-size padding. **Absorbed.**
2. `tests/qwen3_8_flash_next/test_qsa_cache_layout.py`: updated expectation.
   **Absorbed**, plus two new padding tests.
3. `gpu/ops/qsa.py`: `_QSA_MIN_BLOCK_M` `8→16` in the self-developed Triton
   attention path. **Not absorbed** — parent direction is to keep the
   vendored/official QSA selection and not pull the experimental self-developed
   metadata/math path into the PR.

Reader/writer verification before enabling padding: `_store_qsa_kv_rows_kernel`
addresses with `block * stride_k_cache_block + token * stride_k_cache_token`
using the tensor's real strides; `qsa_sparse_paged_attention` and
`_compress_qsa_groups_kernel` likewise consume `stride(0/1/3)`. `bind_kv_cache`
slices the strided view and keeps the allocator storage pointer. Therefore the
side cache is stride-aware and safe under padded pages. The `FullAttentionSpec`
flag is applied only when the installed ABI actually carries the field
(`_BLOCK_STRIDE_SPEC_FLAG`), so reduced builds are unaffected.

## 5. Changes

| File | Change |
|---|---|
| `vllm_fl/models/qwen3_8_flash_next/gpu/model.py` | Added `supports_multimodal_pruning = False` on the composite class + rationale comment |
| `vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py` | `QSAStateBackend.indexes_kv_by_block_stride()`→True; ABI capability detection `_BLOCK_STRIDE_SPEC_FLAG`; conditional spec flag in `QSAKeyStateCache.get_kv_cache_spec` |
| `vllm_fl/patches/qwen3_8_flash_next.py` | Capability-gated packed-GDN import/call so registration survives empty builds |
| `tests/qwen3_8_flash_next/test_qsa_cache_layout.py` | Updated stride expectation; added `test_block_stride_flag_matches_installed_abi`, `test_side_cache_binds_padded_block_stride_view`, `test_padded_slot_mapping_decomposes_to_physical_block_and_offset` |
| `tests/qwen3_8_flash_next/test_multimodal.py` (new, 285 lines) | Genuine image+text coverage: composite config typing/tokens, serving registration, protocol surface, processor factory, image-grid M-RoPE, text false-pass witness, embedding merge, weight mapper |

Not touched: `gpu/ops/qsa.py`, `vendor/**`, config classes, `opencode.jsonc`,
any other worktree/model.

## 6. Validation

Environment: `/vllm-workspace/vllm/.venv` (vLLM `0.1.dev1+g61d4f5663`,
torch 2.13, transformers 5.15.1). Because `vllm_fl/__init__.py` hard-imports
`flag_gems` (unavailable in that image), a local-only stub at
`/tmp/opencode/fgstub` was prepended on `PYTHONPATH`. It is **not** part of the
repository.

Commands and results:

```
# focused new/changed tests + policy regression
PYTHONPATH=/tmp/opencode/fgstub:$PWD /vllm-workspace/vllm/.venv/bin/python -m pytest \
  tests/qwen3_8_flash_next/test_multimodal.py \
  tests/qwen3_8_flash_next/test_qsa_cache_layout.py \
  tests/qwen3_8_flash_next/test_runtime_policy.py -q
# 36 passed

# full Qwen3.8 unit suite (two FLA-dependent GDN modules excluded: image lacks vllm...fla)
PYTHONPATH=/tmp/opencode/fgstub:$PWD /vllm-workspace/vllm/.venv/bin/python -m pytest \
  tests/qwen3_8_flash_next/ -q \
  --ignore=.../test_gdn_packed_decode_optimized.py \
  --ignore=.../test_gdn_packed_decode_patch.py
# 172 passed, 2 failed

# unrelated integration regression
pytest tests/unit_tests/worker/test_allopt_day0_integration.py -q
# 6 passed

# ruff (repo excludes vllm_fl/**)
ruff check tests/qwen3_8_flash_next/test_multimodal.py tests/qwen3_8_flash_next/test_qsa_cache_layout.py
# All checks passed
ruff check .
# 12 errors, all pre-existing in unrelated benchmarks/ and tests/ files
```

The 2 failures are **pre-existing and environmental**, not regressions:
`tests/qwen3_8_flash_next/test_qsa_metadata_graph.py` constructs
`MLAAttentionSpec(compress_ratio=...)`, a field present in the pinned target
`/root/glm5.3/vllm-v0.24.0` but absent from the runnable test image.

## 7. Real H100 validation (executed)

Authorization: real remote H100 validation was granted for this task. All runs
used a **fresh, throwaway** container (`docker run --rm`) launched through the
`baaissh` helper; no pre-existing container was stopped or modified.

Target:
- BAAI alias `aiops-10-8-2-1`, host `p-ef-ch-su04-gpu01-h1-dell-2f-4-cm-231-1-8u-2-1`,
  8x NVIDIA H100 80GB HBM3 (all idle at start, 0 MiB used).
- Image: `harbor.baai.ac.cn/flagos-inner-models-release/...qwen3.8-max-fp8-int8-...-plugin0.3.0-vllm0.24.0-cp312-pt211-cu129-x64-580.126.20:202608100512`
  (image id `b7faa60041af`; vLLM `0.24.0`, torch `2.11.0+cu129`, transformers
  `5.12.1`, FlagGems present, `.fla` present).
- Worktree `vllm_fl` from this change was tarred (`qwen4-wt.tgz`,
  sha256 `85fb31f53866b7c4ed759f308c590046e1b9fbb2396c379a771656d5ae2f8d11`),
  transferred via `scp -O`, extracted, mounted read-only at `/workspace`, and
  prepended on `PYTHONPATH`; verified `vllm_fl.__file__ == /workspace/vllm_fl/...`.
- Real checkpoint: `/data/models/Qwen-Air-Example-CKPT-BF16`
  (`Qwen4ExpForConditionalGeneration`, `qwen4_exp`, `language_model_only=false`,
  vision depth 27 / out 2560, image=248056 video=248057 vs=248053 ve=248054).

Container runtime flags had to reproduce the known-good template exactly:
`--network=host --ipc=host --cap-add=SYS_PTRACE
--security-opt=seccomp=unconfined --security-opt=label=disable
--ulimit memlock=-1 --ulimit stack=67108864`, explicit `/dev/nvidia0..7`,
`/dev/nvidiactl`, `/dev/nvidia-uvm(-tools)`, `nvidia-caps`, the read-only
`libcuda/libnvidia-ml/libnvidia-ptxjitcompiler` mounts under `/driver`, and
`LD_LIBRARY_PATH=/driver:/usr/local/cuda/lib64`. Without
`seccomp=unconfined` torch raised CUDA error 304 (`cuda.is_available()=False`);
with it, preflight passed (`cuda=True count=8 dev0=NVIDIA H100 80GB HBM3`).
BLAS/OpenMP threads were capped to 1.

**Stage A — real processor -> vision embeddings -> M-RoPE.** In-container
`Qwen3VLProcessor.from_pretrained` produced `prompt_len=317`, 256 image tokens,
`pixel_values=(1024,1536)`, `image_grid_thw=[[1,32,32]]`. `Qwen3_VisionTransformer`
loaded all 333 `model.visual.*` tensors (0 missing) and ran on GPU:
`embeddings shape=(256, 2560) dtype=bfloat16`. M-RoPE via the inherited
Qwen3VL helper gave `shape=(3,317)`, `delta=-240`, differing from the text-only
arange, with image-axis uniques `[1, 16, 16]` (16x16 grid). The video processor
path produced `video_grid_thw=[[2,32,32]]`, 512 video tokens. **PASS.**

**Stage B — TP8 offline image+text generation.** `vllm.LLM(tensor_parallel_size=8,
dtype=bfloat16, max_model_len=8192, gpu_memory_utilization=0.85, enforce_eager=True,
limit_mm_per_prompt={"image":1})` constructed in 100.5 s; GPU KV cache
758,506 tokens; generated image-aware text in 33.8 s:
`"The user wants a one-sentence description. I need to look carefully at the
image. It's a 2x2 grid of four squares. Each square"`. **PASS.**

**Stage C — TP8 video generation.** First attempt failed with a precise input
error: `ValueError: Video metadata is required but not found in mm input`; then
`TypeError: VideoMetadata.__init__() missing 1 required positional argument:
'total_num_frames'`. Passing `multi_modal_data={"video": (video,
{"total_num_frames": 8, "fps": 2.0, "frames_indices": [0..7]})}` fixed it:
LLM constructed in 100.1 s; generated in 32.8 s:
`"The user is asking to describe this video in one sentence. Looking at the
frames, all four frames appear identical—a horizontal"`; `[stageC-video] OK`,
exit 0. **PASS.**

Note: an earlier Stage C invocation was killed by the local tool timeout while
the remote container kept running. The orphan (`f26a04fcb7ea`,
`jovial_visvesvaraya`, image `b7faa60041af`) was identified, stopped, and
removed; it was a throwaway container, not one of the pre-existing jobs.

Cleanup: no temporary/auto-generated containers remain; `nvidia-smi` reported
`0 MiB` on all 8 GPUs; the host staging directory
`/public-nvme/yjwu/qwen4-mm-validate-20260910` was deleted.

Remaining limitations:
- Runs used `enforce_eager=True` (torch.compile/CUDA graphs disabled) to isolate
  the multimodal path; graph-mode serving was not re-validated here.
- The pinned development target `/root/glm5.3/vllm-v0.24.0` is source-only, so
  the local unit tests still ran against the reduced image; the H100 container
  image is the real 0.24 ABI and passed end-to-end.

## 8. Risks

1. **QSA ownership discrepancy:** `vendor/PROVENANCE.md` states
   `gpu/ops/qsa.py` is an "import-only v0.24 facade", but the checked-out file
   is a 2361-line cross-vendor Triton implementation that does not import
   `vendor/official/qsa.py`. This predates this work and was deliberately left
   untouched, but the provenance claim should be reconciled by the parent.
2. Enabling `indexes_kv_by_block_stride=True` changes side-cache page padding
   behavior on the pinned ABI. The kernels were verified stride-aware and the
   full model constructed and generated on H100 with this flag active, but a
   mixed-page-size layout that actually triggers padding was not isolated as a
   standalone hardware test.
3. The plugin still imports `flag_gems` unconditionally at package import; any
   runtime without FlagGems cannot import `vllm_fl` at all. Out of scope here.
4. `supports_multimodal_pruning=False` is now hard-coded; if the model later
   gains genuine EVS pruning this must be revisited.

## 9. Delivery state

The model-specific commit includes modified `vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py`,
`vllm_fl/models/qwen3_8_flash_next/gpu/model.py`,
`vllm_fl/patches/qwen3_8_flash_next.py`,
`tests/qwen3_8_flash_next/test_qsa_cache_layout.py`; untracked
`tests/qwen3_8_flash_next/test_multimodal.py` and pre-existing `opencode.jsonc`.
No push was performed; local-only permission configuration `opencode.jsonc` is
intentionally excluded from the commit.
