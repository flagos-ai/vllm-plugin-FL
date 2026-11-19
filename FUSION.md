# PR-style CP layout fusion

This worktree branches from `b632393` on `prstyle-fused-layout`. The original worktree at `/vllm-workspace/vllm-plugin-FL-prstyle` remains on `prstyle-comm-layout-analysis`.

The optimization keeps dense MoE tokens in collective rank order through routing and experts, removing sequence restoration followed by rank packing. Ragged inputs and already-reduced expert backends retain their previous handling. Full attention packs K/V together with a strided Triton copy, exchanges them in one collective, and fuses restoration plus native-head selection into contiguous final K/V buffers. GDN retains sequence order and feeds its reduce-scatter result directly to normalization. Per-forward metadata and bounded cross-forward plan caches reduce repeated CPU work and device synchronization.

The Triton kernels copy values without changing their arithmetic. Decode still executes the native TP model. The plan cache includes query boundaries and prefix lengths, with at most eight entries.

Validation, measured results, traces and reproduction scripts:

- `/vllm-workspace/validation/qwen36_fused_layout_64k_20260929/RESULTS.md`
- `/vllm-workspace/validation/qwen36_fused_layout_64k_20260929/IMPLEMENTATION.md`
- `/vllm-workspace/validation/qwen36_fused_layout_64k_20260929/HANDOFF.md`

Run the layout tests from this worktree with the plugin disabled for model patching:

```bash
VLLM_PLUGINS=fl_qwen_cp VLLM_FL_QWEN_CP=0 python3 -m pytest tests/unit_tests/cp -q
```

The CUDA tests cover exact K/V values and native head ownership, including noncontiguous storage, padding, empty ranks and KV head replication. Additional tests cover MoE token order and reduction, request boundaries and cache invalidation across different prefixes.
