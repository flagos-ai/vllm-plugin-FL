# Ascend-specific patches

This directory contains runtime monkey-patches that adapt upstream vLLM model
code for Huawei Ascend NPUs. These patches are applied automatically when the
`fl` plugin is loaded on an Ascend platform (`VLLM_FL_PLATFORM=ascend`).

## Patch list

| File | Target | Purpose |
|------|--------|---------|
| `patch_mamba_config.py` | `HybridAttentionMambaModelConfig.verify_and_update_config` | Aligns attention/mamba block sizes with Ascend requirements. |
| `patch_multimodal_merge.py` | `vllm.model_executor.models.utils.merge_multimodal_embeddings` | In-place merge of multimodal embeddings on NPU. |
| `patch_qwen3_5.py` | `vllm.model_executor.models.qwen3_next.Qwen3NextAttention.forward` | Fuses the Q/K/V split, RMSNorm and M-RoPE of Qwen3.5/Qwen3.6 full-attention layers into a single Triton kernel (`torch.ops.vllm.triton_split_qkv_rmsnorm_mrope`). |

## Adding a new patch

1. Create `patches/patch_<name>.py`.
2. Expose an idempotent `patch_<name>()` function.
3. Register the function in `../patch.py::apply_ascend_patches()`.
4. Update this README with the new entry.
