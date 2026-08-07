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
| `patch_qwen3_6_gdn.py` | `vllm.model_executor.models.qwen3_next.Qwen3NextGatedDeltaNet._forward_core` / `get_state_shape`, `vllm.model_executor.layers.layernorm.GemmaRMSNorm.forward_oot`, `vllm_fl.worker.model_runner.ModelRunnerFL._reshape_kv_cache_tensors`, `vllm.v1.attention.backends.gdn_attn.GDNAttentionMetadataBuilder.build` | Routes the Qwen3.5/Qwen3.6 GDN core computation to the AscendC fused kernels (`torch.ops._C_ascend.npu_causal_conv1d_custom`, `npu_fused_gdn_gating`, `npu_recurrent_gated_delta_rule`) and GemmaRMSNorm to `npu_gemma_rms_norm` / `npu_add_rms_norm_bias`, mirroring vllm-ascend's `AscendGatedDeltaNetAttention`. Stores the ssm state in the kernel-native `(Hv, Dv, Dk)` layout via `get_state_shape`. Fresh (all-zero `initial_state`) prefill batches run the fused PTO/Bisheng megakernel (`vllm_fl/dispatch/backends/vendor/ascend/impl/pto_chunk_gdn`, vllm-ascend PR #8872, 6 GDN stages in one launch); the PTO-vs-Triton decision and chunk counting are made from CPU-side metadata flags attached by the builder wrap, so no per-layer device syncs are introduced. Other prefill batches keep the Triton `chunk_gated_delta_rule` with transposes at the boundary. Also regroups the mamba KV-cache views into dense per-state tensors (the AscendC kernels address state caches assuming dense layout; the Triton path is layout-agnostic and unaffected). The CANN custom-op environment is bootstrapped automatically (`ASCEND_CUSTOM_OPP_PATH` pointed at the packaged `_cann_ops_custom` vendor dir); the patch falls back to the Triton path only when the `_C_ascend` bindings/op package are unavailable, `VLLM_FL_DISABLE_ASCENDC_GDN=1` is set, or `VLLM_FL_DISABLE_PTO_GDN=1` disables just the PTO megakernel. |
| `patch_qwen3_mtp.py` | `vllm.v1.worker.utils.bind_kv_cache`, `Qwen3NextMultiTokenPredictor.forward`, `Qwen3_5MultiTokenPredictor.forward`, `SpeculativeConfig.hf_config_override`, `MRotaryEmbedding.forward_native`, `Qwen3NextMTP.load_weights` | Enables Multi-Token Prediction (MTP) on Ascend by allowing multiple attention layers to share a layer index in KV-cache binding and by forcing the MTP drafter to use local token embeddings on the last PP rank. Maps `qwen3_5`/`qwen3_5_moe` draft configs to a real `Qwen3NextConfig` (supplying `decoder_sparse_step`, `mlp_only_layers`, etc.) so that MoE checkpoints such as Qwen3.6-35B-A3B can load the upstream `Qwen3NextMTP` drafter. Splits packed `experts.gate_up_proj`/`experts.down_proj` MoE weights into 2-D per-expert `gate_proj`/`up_proj`/`down_proj` slices expected by upstream `FusedMoE.make_expert_params_mapping`. Also makes M-RoPE's fallback native path graph-safe for dynamic token counts. |

## Adding a new patch

1. Create `patches/patch_<name>.py`.
2. Expose an idempotent `patch_<name>()` function.
3. Register the function in `../patch.py::apply_ascend_patches()`.
4. Update this README with the new entry.
