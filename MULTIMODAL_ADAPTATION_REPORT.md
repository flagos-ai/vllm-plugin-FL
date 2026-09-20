# HY4 Multimodal Adaptation Report

- Worktree: `/root/day0-work/vllm-plugin-FL-hy4` (branch `day0-hy4`)
- Upstream PR: https://github.com/flagos-ai/vllm-plugin-FL/pull/453
- ABI target: vLLM `0.24.0`
- Status: **BLOCKED for true image-text support — no official HY4 vision artifact exists.**
  An evidence-based, testable text-only capability contract was implemented instead.
- Delivery: reviewed and committed locally in the model-specific commit; not pushed.

## 1. Executive conclusion

HY4 (`tencent/Hy4-preview`, `tencent/Hy4-preview-FP8`, `model_type=hy_v4`,
`architectures=["HYV4ForCausalLM"]`) is a **text-only** Mixture-of-Experts
language model. Neither released checkpoint contains a vision tower,
`vision_config`, image processor, projector weights, image-token ids, or any
image placeholder token, and the upstream vLLM `hy_v4` implementation is a
plain `SupportsPP` causal LM.

Because there is no architecture to implement against, fabricating a vision
encoder/projector would violate the "do not infer vision architecture from
model name" instruction. The evidence-supported change is therefore:

1. make the text-only contract **explicit and fail-fast** at config load, and
2. pin the model's multimodal capability flag so vLLM 0.24 classifies HY4 as
   text-only and rejects image requests.

This is **not** image-text inference. It prevents silent, incorrect behavior
if a vision-bearing HY4 checkpoint is ever pointed at this plugin.

## 2. Method and evidence

Repository state was audited (config, model, attention, loader, vLLM 0.24
patch, serving registration) and compared against the real checkpoint family
and the upstream native implementation.

### 2.1 Official model config (primary evidence)

Fetched over HTTPS (proxy) and parsed:

- `https://huggingface.co/tencent/Hy4-preview/raw/main/config.json`
  - 59 keys; `architectures=["HYV4ForCausalLM"]`, `model_type="hy_v4"`.
  - Multimodal-like keys (`vision*`, `image*`, `pixel*`, `mm_*`,
    `multimodal*`, `video*`, `audio*`, `projector*`): **none**.
  - Key fields: `hidden_size=6144`, `num_hidden_layers=78`,
    `n_routed_experts=256`, `hc_mult=4`, `kv_lora_rank=512`,
    `qk_rope_head_dim=64`, `index_topk=2048`, `use_mla`, `use_dsa`.
- `https://huggingface.co/tencent/Hy4-preview-FP8/raw/main/config.json`
  - 60 keys; multimodal-like keys: **none**.
  - `quantization_config = {quant_method: modelopt, quant_algo: MXFP8, ...}`.
- HF file lists for both repos contain **no** `preprocessor_config.json`,
  `processor_config.json`, vision/projector shards, or image assets
  (`assets/` images are README artwork only).
- `chat_template.jinja` contains no `image`/`vision`/`<image>` tokens.

### 2.2 Official model card / public model metadata

- `https://huggingface.co/tencent/Hy4-preview` documents a 770B-A49B text
  MoE (Gated DSA, iHC residual streams, MTP); the only occurrence of
  "multimodal" is generic text about the AngelSlim compression toolkit.
- Public model metadata lists `Vision | Not supported` (e.g.
  llmreference.com/model/hy4-preview, haimaker.ai tencent/hy4-preview).
- Hugging Face search for `Hy4-VL`: no results. Tencent's only
  image-text-to-text family is `tencent/HunyuanOCR`, which is unrelated to
  `hy_v4`.
- Official repo `Tencent-Hunyuan/Hy4-preview` contains only
  `README*`, `LICENSE`, `assets/`, `finetune/` — no modeling code and no
  vision module.

### 2.3 Upstream native vLLM implementation

`/root/hy4/worktrees/vllm-hy4-native-opt/vllm/model_executor/models/hy_v4.py`
(native vLLM HY4; read-only inspection):

- Class bases: `nn.Module, SupportsPP, HYV4MixtureOfExperts`.
- No `SupportsMultiModal`, `get_multimodal_embeddings`,
  `get_input_embeddings`, processor, or image path.
- Registry: `"HYV4ForCausalLM": ("hy_v4", "HYV4ForCausalLM")` only.

So the released ABI does not define a multimodal HY4 model.

### 2.4 Local `/root/hy4` adaptation records

`/root/hy4/hy4_reference/`, `/root/hy4/hy4_transformers/`,
`/root/hy4/hy4_vllm_plugin/`, and `/root/hy4/delivery/` were searched for
`vision`, `image`, `multimodal`, `pixel`, `projector`, `connector`,
`preprocessor`: **no HY4-specific multimodal code, config, or processor**.
(The initial access to `/root/hy4` was permission-denied; after the parent
authorized `/root`, it was inspected read-only and no changes were made.)

### 2.5 Comparison of the current plugin against multimodal requirements

| Aspect | Current HY4 plugin | A multimodal HY4 would require |
|---|---|---|
| Config | text-only `HYV4Config` (no vision keys) | `vision_config`, image token ids, image size |
| Processor | none (tokenizer only) | `BaseMultiModalProcessor` + `ProcessingInfo` + dummy inputs |
| Vision encoder | absent | released vision tower/weights + TP sharding |
| Projector/connector | absent | released `mm_projector` weights + activation/merge rule |
| Image placeholder | absent | defined placeholder token + token-expansion rules |
| Multimodal embeddings | not implemented | `SupportsMultiModal`, `get_multimodal_embeddings` |
| Weight loading | text + routed-expert loader only | vision/projector tensor mapping + sharding |
| Serving registration | `SupportsPP` text LM only | multimodal registry entry + mm limits/profiling |

None of the right-hand column artifacts exist upstream, so the implementation
is blocked at the evidence gate.

## 3. Implemented changes (diff summary)

Two source files and one focused test are included in the model-specific commit.

### `vllm_fl/configs/hy_v4.py`
- Added `_MULTIMODAL_KEY_PREFIXES` / `_MULTIMODAL_KEY_EXACT` and
  `_unsupported_multimodal_keys(mapping)`.
- `HYV4Config.__init__` now rejects a checkpoint whose kwargs advertise a
  vision tower (`vision_config`, `image_token_id`, `mm_projector`,
  `multi_modal_projector`, `pixel_values`, ...) with a clear `ValueError`
  instead of silently dropping multimodal parameters.
- Added `HYV4Config.supports_multimodal = False`.

### `vllm_fl/models/hy_v4.py`
- Added `HYV4ForCausalLM.supports_multimodal = False` with a comment stating
  that HY4 preview is text-only, so vLLM 0.24's
  `supports_multimodal()`/registry classifies HY4 as text-only.

### `tests/unit_tests/patches/test_hy_v4_multimodal_contract.py` (new, 11 tests)
- Official Hy4 config keyset contains no multimodal keys.
- `HYV4Config` constructs without multimodal attributes.
- Parametrized rejection of `vision_config`, `image_token_id`,
  `image_token_index`, `vision_start_token_id`, `mm_projector`,
  `multi_modal_projector`, `pixel_values`.
- `HYV4ForCausalLM` is not multimodal per the real vLLM 0.24
  `supports_multimodal()` helper.
- Detector ignores unrelated text keys.

No changes were made to attention, HC/MoE, sparse MLA, W8A8/MXFP8 loading,
the vLLM 0.24 patch, or serving registration. Text path and TP16/empty-build
compatibility are untouched.

## 4. Validation

### 4.1 Static analysis (executed)

- `ruff check vllm_fl/configs/hy_v4.py vllm_fl/models/hy_v4.py tests/unit_tests/patches/test_hy_v4_multimodal_contract.py`
  → **All checks passed!** (ruff 0.16.3)
- `ruff format --check` → config and test files already formatted; the model
  file has pre-existing format drift at lines ~930–995 that is present in
  `HEAD` and unrelated to this diff, so it was intentionally **not**
  reformatted (avoids touching unrelated/concurrent code).

### 4.2 Focused correctness tests (executed, CPU-only)

Harness: disposable stock image
`vllm/vllm-openai:v0.24.0-cu129-ubuntu2404` (Python 3.12.3, vLLM **0.24.0**,
Torch 2.11.0+cu129, pytest 9.1.1), run **without GPUs**. `flag_gems` was
import-stubbed only so `vllm_fl/__init__` can import; no FlagGems op or GPU
kernel was exercised.

| Command (inside container) | Result |
|---|---|
| `pytest tests/unit_tests/patches/test_hy_v4_multimodal_contract.py tests/unit_tests/patches/test_hy_v4_v024.py` | **25 passed** (11 new + 14 existing) |
| `pytest tests/unit_tests/ops/test_hy_v4_hc.py tests/unit_tests/patches/test_hy4_hc_n8_projection.py` | 11 skipped (CUDA-only modules; no GPU by design) |
| `HYV4Config.from_dict(<real tencent/Hy4-preview config.json>)` | accepted: `model_type=hy_v4`, `78` layers, `vocab_size=120832`, `supports_multimodal=False`, no `vision_config` |

The existing 14 HY4 unit tests passing confirms the config/model-class change
does not regress the text path. `is_vllm_024()` / patch registration were not
re-invoked during these tests (unit scope), but no patched symbol changed.

### 4.3 Not executed (and why)

- No GPU kernel/serving/benchmark: forbidden by task constraints and would
  require real 790GB weights.
- HC/ops correctness tests: CUDA-marked, skipped without a GPU.
- End-to-end multimodal rejection over the OpenAI API: impossible to test
  meaningfully because no image-capable HY4 checkpoint exists; would also need
  a GPU. Rejection is covered at the registry/contract level instead.

## 5. Risks and limitations

- The guard matches config keys by well-known prefixes/names. It runs on
  constructor `kwargs`, which is the path used by `from_pretrained`/
  `from_dict`; multimodal keys arriving as explicit constructor parameters or
  set post-construction are not inspected (no such parameters exist today).
- Prefix matching could reject a future text-only key beginning with
  `image_`/`pixel_`/`video_`/`audio_`; this is intentional fail-fast and easily
  narrowed when an official multimodal spec appears.
- `supports_multimodal=False` is redundant with the default `getattr(...,
  False)` but makes the contract explicit and guards against future base-class
  changes; it does not itself enable or disable any vLLM feature.
- This change does **not** make HY4 accept images. Any image request is still
  rejected (now deterministically, and a vision-bearing checkpoint fails at
  config load with a precise message).
- No GPU/numerical evidence was produced; the text/W8A8 TP16 behavior is
  unchanged by construction (no touched runtime path) and by the passing
  existing unit tests, not by a new GPU run.

## 6. Concrete gaps to unblock real image-text support

An official multimodal HY4 must first exist. Required artifacts, in priority
order:

1. A released multimodal HY4 checkpoint (e.g. `Hy4-preview-VL` or equivalent)
   with `vision_config`, image token ids, and an image processor config.
2. The vision tower definition: encoder class, depth, hidden size, patch
   size, attention/ROPE scheme, and a safetensors index listing the vision
   tensors.
3. The projector/connector spec: module type, input/output dims, activation,
   and token-merge/expansion rules (how many LLM tokens per image patch).
4. The placeholder/token scheme in `chat_template.jinja` and tokenizer.
5. The official `modeling_*.py` / vLLM implementation, or fixed
   prompt-token-ids with expected embeddings/logits for a golden check.

Once (1)–(5) exist, the work is a real implementation: `SupportsMultiModal`,
`get_multimodal_embeddings`/`get_input_embeddings`, a
`BaseMultiModalProcessor`/`ProcessingInfo`/dummy-inputs builder, image-token
expansion, vision/projector weight loading with TP sharding, multimodal
profiling and serving registration — then GPU correctness and empty-build
TP16 validation.

## 7. Files touched

```
 vllm_fl/configs/hy_v4.py                                  | 37 +++++++++++++++++++++
 vllm_fl/models/hy_v4.py                                   |  6 +++++
 tests/unit_tests/patches/test_hy_v4_multimodal_contract.py | new (11 tests)
```

The model-specific commit is local only; nothing was pushed, force-pushed, or
applied to shared refs or other worktrees. No GPU service was started or occupied.
