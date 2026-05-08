# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.mamba.gdn_linear_attn import fused_gdn_gating

from vllm_fl.ops.fla import FusedRecurrentGatedDeltaRuleOp

logger = init_logger(__name__)


class FusedRecurrentGatedDeltaRuleFL(FusedRecurrentGatedDeltaRuleOp):
    """Adapter that routes vLLM 0.19 GDN decode through vllm_fl FLA."""

    def __init__(
        self,
        inplace_final_state: bool = True,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> None:
        # Decode patches instantiate this adapter from model forward, where
        # vLLM's CustomOp config context is no longer active.
        torch.nn.Module.__init__(self)
        self._forward_method = self.forward_native
        self.inplace_final_state = inplace_final_state
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        logger.info_once(
            "Using FL OOT FusedRecurrentGatedDeltaRule adapter for GDN decode.",
            scope="local",
        )

    def forward_sigmoid_gating(
        self,
        A_log: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        beta: float = 1.0,
        threshold: float = 20.0,
        scale: float | None = None,
        initial_state: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g, beta_out = fused_gdn_gating(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            beta=beta,
            threshold=threshold,
        )
        return self.forward_native(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta_out,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
        )

    def forward_packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        out: torch.Tensor,
        ssm_state_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, k, v = _unpack_mixed_qkv(mixed_qkv, initial_state)
        cu_seqlens = torch.arange(
            mixed_qkv.shape[0] + 1,
            device=mixed_qkv.device,
            dtype=ssm_state_indices.dtype,
        )
        recurrent_out, final_state = self.forward_sigmoid_gating(
            A_log=A_log,
            a=a,
            b=b,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            scale=scale,
            initial_state=initial_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
        )
        out.copy_(recurrent_out.squeeze(0).unsqueeze(1))
        return out, final_state

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)


_PATCHED = False
_OPS: dict[tuple[bool, bool], FusedRecurrentGatedDeltaRuleFL] = {}


def _get_op(
    inplace_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
) -> FusedRecurrentGatedDeltaRuleFL:
    key = (inplace_final_state, use_qk_l2norm_in_kernel)
    if key not in _OPS:
        _OPS[key] = FusedRecurrentGatedDeltaRuleFL(
            inplace_final_state=inplace_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )
    return _OPS[key]


def _unpack_mixed_qkv(
    mixed_qkv: torch.Tensor,
    initial_state: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hv, v_dim, k_dim = initial_state.shape[-3:]
    qkv_dim = mixed_qkv.shape[1]
    qk_dim = qkv_dim - hv * v_dim
    if qk_dim <= 0 or qk_dim % 2 != 0:
        raise ValueError(f"Invalid mixed_qkv last dim={qkv_dim} for HV={hv}, V={v_dim}.")

    q_dim = qk_dim // 2
    if q_dim % k_dim != 0:
        raise ValueError(f"Invalid Q dim={q_dim}; expected divisible by K={k_dim}.")

    num_q_heads = q_dim // k_dim
    q_raw, k_raw, v_raw = torch.split(mixed_qkv, [q_dim, q_dim, hv * v_dim], dim=-1)
    q = q_raw.view(1, mixed_qkv.shape[0], num_q_heads, k_dim).contiguous()
    k = k_raw.view(1, mixed_qkv.shape[0], num_q_heads, k_dim).contiguous()
    v = v_raw.view(1, mixed_qkv.shape[0], hv, v_dim).contiguous()
    return q, k, v


def _fl_fused_sigmoid_gating_delta_rule_update(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    ssm_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if is_kda:
        raise NotImplementedError("FL GDN recurrent replacement does not handle KDA.")

    return _get_op(
        inplace_final_state,
        use_qk_l2norm_in_kernel,
    ).forward_sigmoid_gating(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
        q=q,
        k=k,
        v=v,
        beta=beta,
        threshold=threshold,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        num_accepted_tokens=num_accepted_tokens,
    )


def _fl_fused_recurrent_gated_delta_rule_packed_decode(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _get_op(
        True,
        use_qk_l2norm_in_kernel,
    ).forward_packed_decode(
        mixed_qkv=mixed_qkv,
        a=a,
        b=b,
        A_log=A_log,
        dt_bias=dt_bias,
        scale=scale,
        initial_state=initial_state,
        out=out,
        ssm_state_indices=ssm_state_indices,
    )


def patch_gdn_recurrent_decode() -> None:
    global _PATCHED
    if _PATCHED:
        return

    import vllm.model_executor.layers.fla.ops as fla_ops
    import vllm.model_executor.layers.fla.ops.fused_recurrent as fused_recurrent
    import vllm.model_executor.layers.fla.ops.fused_sigmoid_gating as fused_sigmoid
    import vllm.model_executor.layers.mamba.gdn_linear_attn as gdn_linear_attn

    fused_sigmoid.fused_sigmoid_gating_delta_rule_update = (
        _fl_fused_sigmoid_gating_delta_rule_update
    )
    fla_ops.fused_sigmoid_gating_delta_rule_update = (
        _fl_fused_sigmoid_gating_delta_rule_update
    )
    gdn_linear_attn.fused_sigmoid_gating_delta_rule_update = (
        _fl_fused_sigmoid_gating_delta_rule_update
    )

    fused_recurrent.fused_recurrent_gated_delta_rule_packed_decode = (
        _fl_fused_recurrent_gated_delta_rule_packed_decode
    )
    fla_ops.fused_recurrent_gated_delta_rule_packed_decode = (
        _fl_fused_recurrent_gated_delta_rule_packed_decode
    )
    gdn_linear_attn.fused_recurrent_gated_delta_rule_packed_decode = (
        _fl_fused_recurrent_gated_delta_rule_packed_decode
    )

    _PATCHED = True
    logger.info_once(
        "Using FL FusedRecurrentGatedDeltaRuleOp for GDN decode recurrent update.",
        scope="local",
    )
