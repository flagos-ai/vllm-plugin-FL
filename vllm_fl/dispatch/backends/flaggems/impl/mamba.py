# SPDX-License-Identifier: Apache-2.0
"""Public GDN adapters; framework tensors and cache lifetimes stay upstream."""

import inspect

from vllm_fl.dispatch.types import BackendImplKind, BackendPriority, OpImpl

_api = None


def uses_speculative_convolution():
    from vllm.config import get_current_vllm_config
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "ascend":
        return False
    return get_current_vllm_config().speculative_config is not None


def _initialize():
    global _api
    if _api is None:
        import flaggems_vllm

        implementation = getattr(flaggems_vllm, "causal_conv1d_update", None)
        if not callable(implementation):
            raise RuntimeError("MTP requires FlagGems-vllm causal_conv1d_update")
        signature = inspect.signature(implementation)
        required = {
            "conv_state",
            "num_accepted_tokens",
            "query_start_loc",
            "max_query_len",
        }
        if not required.issubset(signature.parameters):
            raise RuntimeError("Incompatible speculative convolution API")
        signature.bind(
            None,
            None,
            None,
            **dict(
                bias=None,
                activation=None,
                conv_state_indices=None,
                num_accepted_tokens=None,
                query_start_loc=None,
                max_query_len=-1,
                pad_slot_id=-1,
                block_idx_last_scheduled_token=None,
                initial_state_idx=None,
                validate_data=False,
            ),
        )
        _api = implementation
    return _api


def is_available():
    if not uses_speculative_convolution():
        return False
    _initialize()
    return True


def causal_conv1d_update(
    x,
    conv_state,
    weight,
    bias=None,
    activation=None,
    conv_state_indices=None,
    num_accepted_tokens=None,
    query_start_loc=None,
    max_query_len=-1,
    pad_slot_id=-1,
    block_idx_last_scheduled_token=None,
    initial_state_idx=None,
    validate_data=False,
):
    """Mutate x and conv_state; return the filtered x with its original dtype.

    Reuse upstream layouts: varlen x is [tokens, channels]; state is a
    [slots, channels, history] view with channel stride one. Accepted counts
    include the bonus token; history contains K-1 plus speculative positions.
    """
    if x.ndim not in (2, 3) or x.shape[1] < 1024:
        raise NotImplementedError(
            "Ascend speculative convolution requires at least 1024 channels"
        )
    if (
        conv_state.ndim != 3
        or conv_state.stride(-2) != 1
        or weight.ndim != 2
        or weight.shape[0] != x.shape[1]
        or conv_state.shape[1] != x.shape[1]
    ):
        raise ValueError(
            "Speculative convolution requires matching channel-major views"
        )
    return _initialize()(
        x,
        conv_state,
        weight,
        bias=bias,
        activation=activation,
        conv_state_indices=conv_state_indices,
        num_accepted_tokens=num_accepted_tokens,
        query_start_loc=query_start_loc,
        max_query_len=max_query_len,
        pad_slot_id=pad_slot_id,
        block_idx_last_scheduled_token=block_idx_last_scheduled_token,
        initial_state_idx=initial_state_idx,
        validate_data=validate_data,
    )


def register_causal_conv1d_operator(registry):
    causal_conv1d_update._is_available = is_available
    registry.register_impl(
        OpImpl(
            op_name="causal_conv1d_update",
            impl_id="default.flaggems_vllm",
            kind=BackendImplKind.DEFAULT,
            fn=causal_conv1d_update,
            priority=BackendPriority.DEFAULT,
            allow_execution_fallback=False,
        )
    )


_chunk_api = None


def chunk_available():
    global _chunk_api
    if not uses_speculative_convolution():
        return False
    if _chunk_api is None:
        import flaggems_vllm

        implementation = getattr(flaggems_vllm, "chunk_gated_delta_rule_fwd", None)
        if not callable(implementation):
            raise RuntimeError("MTP requires FlagGems-vllm chunk_gated_delta_rule_fwd")
        inspect.signature(implementation).bind(
            q=None,
            k=None,
            v=None,
            g=None,
            beta=None,
            scale=1.0,
            initial_state=None,
            output_final_state=True,
            cu_seqlens=None,
        )
        _chunk_api = implementation
    return True


def gdn_chunk_prefill(
    q,
    k,
    v,
    g,
    beta,
    initial_state,
    output_final_state,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_offsets=None,
    use_qk_l2norm_in_kernel=True,
    core_attn_out=None,
):
    """Map upstream [N,H,V,K] states to the library's [N,H,K,V] contract.

    Caller chunk_indices/offsets are backend-specific scheduling hints. Like
    upstream's FlashInfer path, this backend rebuilds its own schedule from
    cu_seqlens, retaining the original sequence boundaries and tensor order.
    """
    import torch

    if q.shape[-1] != 128 or v.shape[-1] != 128:
        raise NotImplementedError("Ascend GDN chunk prefill requires K=V=128")
    if use_qk_l2norm_in_kernel:
        from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd

        q, k = l2norm_fwd(q), l2norm_fwd(k)
    state = (
        initial_state.transpose(-1, -2).contiguous()
        if initial_state is not None
        else None
    )
    lengths = cu_seqlens.to(torch.long) if cu_seqlens is not None else None
    result = _chunk_api(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=q.shape[-1] ** -0.5,
        initial_state=state,
        output_final_state=output_final_state,
        cu_seqlens=lengths,
    )
    output = result[1].to(q.dtype)
    final_state = (
        result[3].transpose(-1, -2).contiguous() if result[3] is not None else None
    )
    if core_attn_out is not None:
        core_attn_out.reshape(-1)[: output.numel()].copy_(output.reshape(-1))
    return output, final_state


def register_gdn_chunk_operator(registry):
    gdn_chunk_prefill._is_available = chunk_available
    registry.register_impl(
        OpImpl(
            op_name="gdn_chunk_prefill",
            impl_id="default.flaggems_vllm",
            kind=BackendImplKind.DEFAULT,
            fn=gdn_chunk_prefill,
            priority=BackendPriority.DEFAULT,
            allow_execution_fallback=False,
        )
    )
