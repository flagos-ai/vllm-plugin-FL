"""Thin vLLM bridge to the FlagGems Qwen GDN integration."""

from __future__ import annotations

import sys


def install_vllm_gdn_bridge() -> None:
    """Install the version-adapted FlagGems GDN runtime lazily."""
    # vLLM 0.24 split the generic GDN base from the Qwen implementation and
    # moved the latter into the mamba.gdn package. The existing FlagGems
    # integration patches the old module path and class name, so expose a
    # narrow compatibility alias without changing either project globally.
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn

    gdn.GatedDeltaNetAttention = gdn.QwenGatedDeltaNetAttention
    sys.modules.setdefault(
        "vllm.model_executor.layers.mamba.gdn_linear_attn", gdn
    )

    from flag_gems.integrations.vllm import install_qwen_gdn

    install_qwen_gdn()

    # ChunkGatedDeltaRule.forward_native in 0.24 can provide a preallocated
    # output buffer.  The existing CPU implementation returns the same tensor
    # instead, so bridge the new optional ABI and preserve both behaviours.
    flag_gems_chunk_gated_delta_rule = gdn.fla_chunk_gated_delta_rule

    def chunk_gated_delta_rule_024(*args, core_attn_out=None, **kwargs):
        output, final_state = flag_gems_chunk_gated_delta_rule(*args, **kwargs)
        if core_attn_out is not None:
            output_flat = output.squeeze(0).reshape(-1)
            target_flat = core_attn_out.reshape(-1)
            target_flat[: output_flat.numel()].copy_(output_flat)
        return output, final_state

    gdn.fla_chunk_gated_delta_rule = chunk_gated_delta_rule_024

    # vLLM 0.24 selects a new CPU-only GDN implementation in ``__init__``.
    # That path bypasses the FlagGems functions installed above and explicitly
    # rejects speculative metadata.  Keep using the generic Qwen GDN core on
    # Apple ARM: despite the historical ``forward_cuda`` name, the patched
    # implementation dispatches only CPU tensors to FlagGems/libtriton_jit.
    # Install this after FlagGems so we retain its projection setup wrapper.
    original_init = gdn.QwenGatedDeltaNetAttention.__init__

    def install_arm_cpu_forward(self, *args, **kwargs) -> None:
        original_init(self, *args, **kwargs)
        self._forward_method = self.forward_cuda

    gdn.QwenGatedDeltaNetAttention.__init__ = install_arm_cpu_forward

    # vLLM 0.24 passes an additional token-count argument to this hook.
    # FlagGems intentionally disables it because the accelerator JIT warmup
    # is neither needed nor valid for the Apple CPU runtime.
    def no_op_prefill_warmup(self, *args, **kwargs) -> None:
        return None

    gdn.QwenGatedDeltaNetAttention._warmup_prefill_kernels = (
        no_op_prefill_warmup
    )


__all__ = ["install_vllm_gdn_bridge"]
