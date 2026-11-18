# SPDX-License-Identifier: Apache-2.0
import torch

from vllm.models.hy_v4.nvidia.hc import HYV4HCLayer

from vllm_fl.ops.hy4_flash_kernels import ihc_post, ihc_pre


class HYV4FlashHCLayer(HYV4HCLayer):
    """Keep inherited parameters and route the H100 iHC pre/post computation."""

    def pre(self, hidden_states: torch.Tensor):
        if not self.enable_ihc:
            return super().pre(hidden_states)
        pre = self.hc_pre
        reduced, gates = ihc_pre(
            hidden_states,
            pre.hc_fn.weight,
            pre.hc_scale,
            pre.hc_base,
            pre.magnitude,
            pre.hc_eps,
            pre.layernorm_epsilon,
        )
        return reduced, gates, hidden_states

    def post(self, output_with_bias, residual, post_gates):
        if not self.enable_ihc:
            return super().post(output_with_bias, residual, post_gates)
        assert post_gates is not None
        return ihc_post(output_with_bias, residual, post_gates)
