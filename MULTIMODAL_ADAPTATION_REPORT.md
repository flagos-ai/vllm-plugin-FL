# GLM5.3-Flash Multimodal Day0 Adaptation — Audit, Fixes & H100 Validation

Status: **reviewed and committed locally in `/root/day0-work/vllm-plugin-FL-glm53`**, branch
`day0-glm53`; not pushed. The parent of this delivery commit is
`1a6d293c7fc4ecd8c740a367c3e1a83c88886929` (PR [#454](https://github.com/flagos-ai/vllm-plugin-FL/pull/454)
head).
A real single-GPU H100 validation was completed. Nothing is reported as passing unless it ran.

---

## 0. TL;DR

| Item | Outcome |
| --- | --- |
| Video prompt frame count vs encoder `grid_t` | **Bug found & fixed** (real H100: 3 == 3) |
| Video encoder token ceiling | **Bug found & fixed** (real: 37170 vs inherited 2242) |
| Vision SwiGLU clamp | **Investigated, NOT changed** — checkpoint declares `glm5_next_vision`; validated native runtime clamps; real-weight output matched native runtime **maxdiff 0.0** |
| Real H100 image E2E | **Passed** (grid `[1,22,30]`, 660 patches, 165 embed tokens) |
| Real H100 video E2E | **Passed** (grid `[3,16,16]`, 768 patches, timestamps 3 == grid_t) |
| Real-weight vision forward | **Passed** — plugin tower == validated native tower, `maxdiff 0.0`, finite |
| Full image→text generation | **Not run** — blocker: 642.7 GB BF16 weights / 8×80 GB ≈ 80.3 GB/GPU weights alone |

---

## 1. Scope and evidence sources

Target ABI: **vLLM 0.24.0** on H100. Model: GLM5.3-Flash (`model_type=glm5_next`,
arch `Glm5NextForConditionalGeneration`).

Sources read (all under `/root`; `/vllm-workspace` never accessed):

- `vllm_fl/models/glm5_next_multimodal.py`, `vllm_fl/transformers_utils/processors/glm5_next.py`,
  `vllm_fl/configs/glm5_next.py`, `vllm_fl/patches/glm5_next_v024.py`
- native vLLM 0.24 ABI: `/root/glm5.3/vllm-v0.24.0/vllm/model_executor/models/{glm4_1v,glm_ocr}.py`
- official native reference: `/root/glm5.3/vllm-glm5next-ref/vllm/models/glm5next/nvidia/*`
- older 0808 references: `/root/glm5.3/glm-5-next-adapt-0808/glm-5-next-{transformers,sgl,vllm-glm5next}.patch`
- released plugin patch: `/root/glm5.3/glm5.3-flash-flagrelease-20260827/patches/vllm-plugin-FL-1dfd969-glm5-next.patch`
- checkpoint config evidence: `/public-nvme/models/GLM-5.3-Flash-BF16/{config.json,processor_config.json}`
  and `model.safetensors.index.json`
- validated native runtime baked into `glm5-next/native-vllm024:20260826-tp8-fix1`
  (`vllm_glm5_next` package; the plugin's processor is byte-identical to it)

### Connectivity map verified (static)

- **Registration**: `apply_glm5_next_v024_patches()` registers
  `Glm5NextForConditionalGeneration` in `_VLLM_MODELS`/`ModelRegistry`; the class carries
  `@MULTIMODAL_REGISTRY.register_processor(Glm5NextMultiModalProcessor, info=Glm5NextProcessingInfo,
  dummy_inputs=Glm4vDummyInputsBuilder)`; config registry + arch convertor installed. **Connected.**
- **Weight mapping**: in the vLLM 0.24 ABI `Glm4vVisionTransformer.load_weights` uses an explicit
  stacked mapping (`attn.q/k/v→attn.qkv`, `gate_proj/up_proj→gate_up_proj`), so the self-contained
  `nn.Module` fork is safe (it does not read `self.hf_to_vllm_mapper`). Verified on real weights:
  347 checkpoint tensors → 298 params loaded.
- **MM embedding / position**: `_process_image_input`/`_process_video_input` and
  `run_dp_sharded_mrope_vision_model` match the fork forward signature. Real forward executed.
- **CUDA-graph boundary**: `encoder_cudagraph_forward(view…)` calls
  `self.visual(pixel_values, None, encoder_metadata=…)`; the fork supports it and
  `get_encoder_cudagraph_config()` strips `pos_embeds`. Static check only (not replayed on H100).

---

## 2. Findings

### Finding 1 — Vision SwiGLU clamp: INVESTIGATED, NOT A DEFECT (no change)

Initial audit against the 0808 references suggested the vision MLP/merger should be **unclamped**
(`GlmOcrVisionMlp`/`GlmOcrVisionPatchMerger = ACT2FN`, sglang and native vLLM `GlmOcrVisionTransformer`
both unclamped). That was a false positive:

- The actual checkpoint declares `vision_config.model_type = "glm5_next_vision"` (not
  `glm_ocr_vision`, which the 0808 transformers patch defaults to).
- The **validated native runtime for this exact checkpoint** (`vllm_glm5_next` in
  `glm5-next/native-vllm024:20260826-tp8-fix1`) defines `Glm5NextVisionConfig(model_type =
  "glm5_next_vision")` and **clamps** both the block MLP and the merger, with the same comment as the
  plugin.
- Real-weight H100 forward (below) shows the plugin tower with the clamp retained is **bit-identical**
  (`maxdiff 0.0`) to that validated native runtime.

Action: **no code change**; a regression test locks the clamped behavior. (`swiglu_limit=10.0`
appears in both text and vision configs; the text clamp is untouched.)

### Finding 2 — Video prompt frame count did not match the encoder `grid_t` (FIXED)

The plugin returns `_hf_processor_applies_updates=False`, so vLLM expands the prompt via
`Glm4vMultiModalProcessor._get_prompt_updates`. For non-`Glm4vProcessor`/non-GLMGA processors this
uses `Glm4vProcessingInfo._get_video_second_idx_glm46v`, whose **duration-threshold** policy
(fps 3/1/0.5, cap 640) differs from the plugin's **`fps_interval`** sampler
(`Glm5NextVideoProcessor.sample_frames`, default 2, cap 2048). Measured pure-Python mismatch:

```
 30s@30fps : fork grid_t=30, vllm ts=90   -> mismatch
100s@30fps : fork grid_t=100, vllm ts=100 -> ok
 10s@30fps : fork grid_t=10, vllm ts=30   -> mismatch
 24s@2fps  : fork grid_t=24, vllm ts=24   -> ok
  5s@25fps : fork grid_t=5,  vllm ts=15   -> mismatch
```

Fix: `glm_video_timestamp_seconds(video_processor, metadata)` in the processor module (one source of
truth calling the processor's own `sample_frames`, then `[::2]`), consumed by
`Glm5NextProcessingInfo._get_video_second_idx_glm46v`.

### Finding 3 — Video encoder ceiling used the unused placeholder `size` (FIXED)

`Glm5NextVideoProcessor` sets `size={"longest_edge": 1}` (token-budget schema), but the vLLM 0.24
base `get_mm_max_tokens_per_item` reads `video_processor.size["longest_edge"]` (`glm4_1v.py:1015`)
and computes `max_vision_tokens = 0`. Fix: override `get_mm_max_tokens_per_item` to use
`_get_video_max_pixels()` and `max_frame_count_dynamic`. Real checkpoint: **37170 vs 2242** inherited.

### Finding 4 (latent / not changed) — generic-backend `get_number_of_video_patches`

`_get_num_multimodal_tokens` calls `self.video_processor.get_number_of_video_patches`, undefined on
the plugin's video processor. Against `/root/glm5.3/vllm-v0.24.0`: this API is used only by the
generic HF backend (`vllm/model_executor/models/transformers/multimodal.py:71,222`), never by the
native model path; upstream transformers 5.5.3 (`models/glm46v/processing_glm46v.py:205`) has the
same undefined call. **No change** (off-path, documented).

---

## 3. Files changed in the model-specific commit

```
 M vllm_fl/models/glm5_next_multimodal.py             (+73)
 M vllm_fl/transformers_utils/processors/glm5_next.py (+35)
?? MULTIMODAL_ADAPTATION_REPORT.md
?? tests/unit_tests/glm5_next/test_multimodal_connectivity.py
```

- `glm5_next_multimodal.py`: `Glm5NextProcessingInfo.get_mm_max_tokens_per_item` and
  `_get_video_second_idx_glm46v`. Vision tower/activation unchanged.
- `glm5_next.py`: `glm_video_timestamp_seconds` + `__all__`.
- `test_multimodal_connectivity.py` (10 tests): static contracts (clamp retained; overrides
  present; budget uses `_get_video_max_pixels`), numeric sampler/`grid_t` alignment over 6 shapes,
  legacy-policy desync regression, `do_sample_frames=False` path.

---

## 4. Real H100 validation (executed)

### 4.1 Host / GPU preflight

- Host: `p-ef-ch-su05-gpu05-h1-dell-2f-4-cm-232-3-8u-2-90` (alias `aiops-10-8-2-90`), 8× H100 80GB,
  all idle before the run.
- Container runtime `nvidia` present, default `runc`. A plain `--gpus device=0` container did **not**
  inject `/dev/nvidia*` (known 10.8.2.* behavior); the proven pattern was used instead:
  explicit `--device /dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`,
  `/dev/nvidia-caps/nvidia-cap{0,1,2}` + read-only `/driver/lib{cuda,nvidia-ml,nvidia-ptxjitcompiler}`
  mounts and `LD_LIBRARY_PATH=/driver:...`.
- Preflight result inside the container: `/dev/nvidia0` present; `torch.cuda.is_available()=True`,
  `device_count()=1`, device name `NVIDIA H100 80GB HBM3`. GPU used: **device 0**.

### 4.2 Image / worktree

- Image: `glm5-next/native-vllm024:20260826-tp8-fix1`
  (torch 2.11.0+cu129, vLLM 0.24.0, transformers 5.12.1; `vllm._C_stable_libtorch` present,
  `deep_gemm` available; no FlagGems, no `vllm_fl`). It ships the validated native `vllm_glm5_next`
  reference runtime.
- The current worktree was archived, hashed, transferred, and extracted:
  `sha256(eb15f0b4ffd383ce1a9d3744702214d1037755e4418b40eae273c077357119a2)` verified on both ends,
  mounted read-only at `/opt/glm53` with `PYTHONPATH=/opt/glm53`. In-container checks confirmed the
  two overrides (`get_mm_max_tokens_per_item`, `_get_video_second_idx_glm46v`), the helper, and the
  retained vision clamp were the ones under test.
- Real weights: `/public-nvme/models/GLM-5.3-Flash-BF16` (mounted read-only).

### 4.3 Test 1 — real-weight vision forward (plugin vs validated native runtime)

Built both `vllm_fl...Glm5NextVisionTransformer` and `vllm_glm5_next...Glm5NextVisionTransformer` with
the real `vision_config`, loaded the 347 real `model.visual.*` tensors (1.127 GB, one shard), and ran
a forward on a synthetic grid `[[1,32,32]]` (bf16, GPU 0):

```
our loaded 298 ref loaded 298
vit attention backend: AttentionBackendEnum.TRITON_ATTN
our (256, 4096) ref (256, 4096)
maxdiff 0.0  meandiff 0.0  finite True
VISION_FORWARD_OK
```

→ plugin vision tower (merge 2×2, out_hidden 4096, clamp retained) is **bit-identical** to the
validated native runtime on the real checkpoint weights; the interpolation/position path executed.

### 4.4 Test 2 — processor → placeholder/`grid_t` (real config + tokenizer)

```
image min/max tokens 16 8000  patch_expand 1  resize_mode pad
video min/max tokens 16 30000  fps_interval 2.0  max_frame_count_dynamic 2048
image_grid_thw [1, 22, 30]  pixel_values (660, 1176)  embed tokens 165  image_token present
video_grid_thw [3, 16, 16]  pixel_values_videos (768, 1176)
sampled frames 6  grid_t 3  prompt timestamps 3        # aligned (Finding 2)
override video ceiling 37170  expected 37170  inherited-buggy 2242   # Finding 3
PROCESSOR_E2E_OK
```

Image: `pixel_values.shape[0] == grid.prod()` and placeholder token count `== grid.prod()//merge²`.
Video: prompt timestamp count `== grid_t` exactly. These use the real `processor_config.json`
(`min/max_image_tokens`, `patch_expand_factor=1`, `resize_mode=pad`, `fps_interval=2`) and tokenizer.

### 4.5 Not executed: full image/video → generation request

Blocked by memory: the BF16 checkpoint is **642.7 GB across 120 shards**; TP8 on this single 8×80 GB
node is **≈80.3 GB/GPU of weights alone** before activations/KV, so an in-memory vLLM service cannot
start here (the release used 2 nodes / TP16). Generation was therefore not attempted. This is the
documented fallback from the assignment (single-GPU minimal real vision forward + processor E2E).

### 4.6 Cleanup

All containers were ephemeral (`--rm`, image mode); no service was started. Remote archive and
extraction directory were removed; `docker ps -a | grep glm53` = none. Pre-existing exited
containers from other users were left untouched.

---

## 5. Verification run (all commands, local + H100)

Interpreter for local tests: `/root/qwen3.8_handoff/.venv-v0202-patchcheck/bin/python`
(torch 2.11.0, transformers 4.57.6). H100 runs used the native image above.

| Command | Result |
| --- | --- |
| `ruff check <3 changed files>` | **All checks passed!** |
| `ruff format --check <3 changed files>` | **3 files already formatted** |
| `pytest tests/unit_tests/glm5_next/test_multimodal_connectivity.py -q` | **10 passed** |
| `pytest test_graph_boundaries.py test_multimodal_connectivity.py -q` | **14 passed, 1 failed** (only `No module named 'vllm'`) |
| `pytest tests/unit_tests/glm5_next/ --collect-only` | **30 collected, 8 collection errors** (`flag_gems`/`vllm` absent locally) |
| H100 vision forward (real weights, plugin vs native) | **maxdiff 0.0, finite** |
| H100 processor E2E (real image + video) | **PASS** (165 image tokens; 3==3 video frames; 37170 ceiling) |

Local full-processor preprocess remains blocked by transformers 4.57 API drift
(`KeyError: do_convert_rgb`); the target stack (transformers 5.12/5.16) was exercised on the H100.

---

## 6. Risks / limitations

1. **Full generation not validated** (memory blocker, §4.5). The language model + KV cache path for
   multimodal prompts was not exercised on this node.
2. **Vision-clamp decision is reference + native-runtime based**, now corroborated by a real-weight
   bit-exact match against the validated runtime; it does not independently prove the checkpoint's
   training-time activation beyond that agreement.
3. **Request-level video overrides** (`fps`/`max_frames`) are not visible to the prompt-timestamp
   helper; alignment holds for default processor settings (same limitation as vLLM's GLM-4.6V path).
4. CUDA-graph replay of the encoder was not re-run on H100 (static boundary checks only).
5. Finding 4 (generic-HF-backend patch count) remains a latent, off-path gap.

## 7. Suggested follow-ups (main thread)

- Run a full TP16/2-node image+video generation once such a resource is assigned; assert
  `num_image_tokens == grid.prod()//merge²` and video frame counts, and compare clamped-vs-reference
  image features.
- Add `get_number_of_video_patches` if generic-transformers-backend loading is ever needed.

## 8. Artifacts

- Changed source/test paths: §3.
- Worktree archive SHA256: `eb15f0b4ffd383ce1a9d3744702214d1037755e4418b40eae273c077357119a2`
  (assembled with the two fixes + retained clamp; remote copy cleaned up).
- The model-specific commit is local only; no push was performed.
