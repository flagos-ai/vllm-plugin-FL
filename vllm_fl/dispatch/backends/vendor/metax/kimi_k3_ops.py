"""K3-specific VENDOR capabilities; no generic MetaX implementation is replaced."""
from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl


def register_builtins(registry):
    from vllm_fl.attention.kimi_k3.backend import attention_backend
    from .impl.kimi_k3.experts import tp8_experts, grouped_moe, activations
    from .impl.kimi_k3.attention import mla_tp8, prefill_mma, prefill_kernel, cache_gather, mla_bmm
    from .impl.kimi_k3.kda.interface import chunk_kda_with_fused_gate
    from .impl.kimi_k3.linear import linear, latent, latent_up, small_projection
    from .impl.kimi_k3.vision import apply_rope
    from vllm_fl.attention.kimi_k3.decode import flash_mla_with_kvcache

    implementations = {
        "attention_backend": attention_backend,
        "kimi_k3_prepare_experts": tp8_experts.prepare,
        "kimi_k3_tp8_experts": tp8_experts.run,
        "kimi_k3_grouped_experts": grouped_moe.grouped_experts,
        "kimi_k3_situ_out": activations.situ_out,
        "kimi_k3_mla_tp8": mla_tp8.run,
        "kimi_k3_prefill_mma": prefill_mma.run,
        "kimi_k3_prefill_attention": prefill_kernel.attention,
        "kimi_k3_cache_gather": cache_gather.gather,
        "kimi_k3_query_projection": mla_bmm.project_query,
        "kimi_k3_chunk_kda": chunk_kda_with_fused_gate,
        "kimi_k3_linear": linear.linear,
        "kimi_k3_latent_projection": latent.project_grouped,
        "kimi_k3_latent_up": latent_up.accumulate_latent_up,
        "kimi_k3_small_projection": small_projection.run,
        "kimi_k3_vision_rope": apply_rope,
        "kimi_k3_mla_decode": flash_mla_with_kvcache,
    }
    for name, fn in implementations.items():
        identity = "vendor.metax.kimi_k3." + name
        prior = registry.get_implementation(name, identity)
        if prior is not None:
            if prior.fn is not fn:
                raise RuntimeError(f"Conflicting K3 provider: {name}")
            continue
        registry.register_impl(OpImpl(
            op_name=name, impl_id=identity, kind=BackendImplKind.VENDOR,
            fn=fn, vendor="metax", priority=BackendPriority.VENDOR + 10,
        ))
