# Copyright (c) 2025 BAAI. All rights reserved.

import warnings
from typing import Optional, Union
import os
import torch

from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.fla.ops.l2norm import l2norm_fwd
from vllm.model_executor.layers.fla.ops.utils import input_guard
from flag_gems.fused.FLA import chunk_gated_delta_rule_fwd



class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        return o.to(q.dtype), final_state
    

@CustomOp.register("chunk_gated_delta_rule")
class ChunkGatedDeltaRuleOp(CustomOp):
    def __init__(
        self,
        output_final_state: bool = False,
        use_qk_l2norm_in_kernel: bool = False,
    ) -> None:
        r"""
        Args:
        output_final_state (Optional[bool]):
                Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        """
        super().__init__()
        self.output_final_state = output_final_state
        self.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        self.suppress_level = int(os.getenv("GDN_RECOMPUTE_SUPPRESS_LEVEL", "0"))

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float = None,
        initial_state: torch.Tensor = None,
        cu_seqlens: torch.LongTensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
        r"""
        Args:
            q (torch.Tensor):
                queries of shape `[B, T, H, K]``.
            k (torch.Tensor):
                keys of shape `[B, T, H, K]`.
            v (torch.Tensor):
                values of shape `[B, T, H, V]`.
            g (torch.Tensor):
                (forget) gating tensor (in log space!) of shape `[B, T, H]`.
            beta (torch.Tensor):
                betas of shape `[B, T, H]`.
            scale (Optional[int]):
                Scale factor for the RetNet attention scores.
                If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
            initial_state (Optional[torch.Tensor]):
                Initial state of shape `[N, H, K, V]` for `N` input sequences.
                For equal-length input sequences, `N` equals the batch size `B`.
                Default: `None`.
            cu_seqlens (torch.LongTensor):
                Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
                consistent with the FlashAttention API.

        Returns:
            o (torch.Tensor):
                Outputs of shape `[B, T, H, V]`.
            final_state (torch.Tensor):
                Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.
        """
        assert q.dtype == k.dtype == v.dtype
        assert q.dtype != torch.float32, (
            "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
        )
        assert len(beta.shape) == 3, (
            "beta must be of shape [B, T, H] if head_first=False, or [B, H, T] otherwise."
        )

        if q.shape[1] < q.shape[2]:
            warnings.warn(
                f"Input tensor shape suggests potential format mismatch: seq_len ({q.shape[1]}) < num_heads ({q.shape[2]}). "
                "This may indicate the inputs were passed in head-first format [B, H, T, ...] "
                "Please verify your input tensor format matches the expected shape [B, T, H, ...].",
                stacklevel=2,
            )
        if cu_seqlens is not None:
            if q.shape[0] != 1:
                raise ValueError(
                    f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                    f"Please flatten variable-length inputs before processing."
                )
            if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
                raise ValueError(
                    f"The number of initial states is expected to be equal to the number of input sequences, "
                    f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
                )
        if scale is None:
            scale = k.shape[-1] ** -0.5
        o, final_state = ChunkGatedDeltaRuleFunction.apply(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            self.output_final_state,
            cu_seqlens,
            self.use_qk_l2norm_in_kernel,
        )
        return o, final_state
