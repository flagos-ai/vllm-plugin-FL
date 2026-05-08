# Copyright (c) 2026 BAAI. All rights reserved.

import torch

from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.mamba.gdn_linear_attn import ChunkGatedDeltaRule
from vllm_fl.ops.fla.chunk import ChunkGatedDeltaRuleOp

logger = init_logger(__name__)


class ChunkGatedDeltaRuleFL(ChunkGatedDeltaRule):
    """OOT adapter that routes vLLM 0.19 GDN prefill through vllm_fl FLA."""

    def __init__(self) -> None:
        CustomOp.__init__(self)
        self._forward_method = self.forward_native
        self._chunk_gated_delta_rule = ChunkGatedDeltaRuleOp()
        logger.info_once(
            "Using FL OOT ChunkGatedDeltaRule adapter for GDN prefill.",
            scope="local",
        )

    def forward_native(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._chunk_gated_delta_rule.forward_native(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        )

    def forward_cuda(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_oot(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)
