# Copyright (c) 2025 BAAI. All rights reserved.

import warnings
from typing import Optional, Union
import os
import torch

from vllm.model_executor.custom_op import CustomOp
from flag_gems.fused.FLA import fused_recurrent_gated_delta_rule_fwd


class FusedRecurrentFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        inplace_final_state: bool = True,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        o, final_state = fused_recurrent_gated_delta_rule_fwd(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            beta=beta.contiguous(),
            scale=scale,
            initial_state=initial_state,
            inplace_final_state=inplace_final_state,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=ssm_state_indices,
            num_accepted_tokens=num_accepted_tokens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

        return o, final_state
    

@CustomOp.register("fused_recurrent_gated_delta_rule")
class FusedRecurrentGatedDeltaRuleOp(CustomOp):
    def __init__(
        self,
        inplace_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> None:
        r"""
        Args:
        inplace_final_state (Optional[bool]):
                Whether to inplace the final state of shape `[N, H, K, V]`. Default: `False`.
        """
        super().__init__()
        self.inplace_final_state = inplace_final_state
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor = None,
        scale: float = None,
        initial_state: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        ssm_state_indices: torch.Tensor | None = None,
        num_accepted_tokens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        `Args:
            q (torch.Tensor):
                queries of shape `[B, T, H, K]`.
            k (torch.Tensor):
                keys of shape `[B, T, H, K]`.
            v (torch.Tensor):
                values of shape `[B, T, HV, V]`.
                GVA is applied if `HV > H`.
            g (torch.Tensor):
                g (decays) of shape `[B, T, HV]`.
            beta (torch.Tensor):
                betas of shape `[B, T, HV]`.
            scale (Optional[int]):
                Scale factor for the RetNet attention scores.
                If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
            initial_state (Optional[torch.Tensor]):
                Initial state of shape `[N, HV, K, V]` for `N` input sequences.
                For equal-length input sequences, `N` equals the batch size `B`.
                Default: `None`.
            inplace_final_state: bool:
                Whether to store the final state in-place to save memory.
                Default: `True`.
            cu_seqlens (torch.LongTensor):
                Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
                consistent with the FlashAttention API.
            ssm_state_indices (Optional[torch.Tensor]):
                Indices to map the input sequences to the initial/final states.
            num_accepted_tokens (Optional[torch.Tensor]):
                Number of accepted tokens for each sequence during decoding.

        Returns:
            o (torch.Tensor):
                Outputs of shape `[B, T, HV, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, HV, K, V]`.
        """
        if cu_seqlens is not None and q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if scale is None:
            scale = k.shape[-1] ** -0.5
        else:
            assert scale > 0, "scale must be positive"
        if beta is None:
            beta = torch.ones_like(q[..., 0])
        o, final_state = FusedRecurrentFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            self.inplace_final_state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            self.use_qk_l2norm_in_kernel,
        )
        return o, final_state
