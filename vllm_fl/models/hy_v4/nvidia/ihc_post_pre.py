# SPDX-License-Identifier: Apache-2.0
"""Default-off orchestration for the one intra-layer post/pre boundary."""

from vllm_fl.utils import use_flaggems_op

from torch.nn.modules import module as nn_module


def maybe_ihc_post_pre(owner, output, residual, post_gates):
    if not use_flaggems_op("ihc_post_pre", default=False):
        return None
    from .hc import HYV4HCLayer, HYV4HCPostLayer, HYV4HCPreLayer
    from .triton_ihc import triton_ihc_supported

    before, after = owner.hc_attn_layer, owner.hc_mlp_layer
    if (
        type(before) is not HYV4HCLayer
        or type(after) is not HYV4HCLayer
        or not (before.enable_ihc and after.enable_ihc)
    ):
        return None
    post_layer, pre_layer = before.hc_post, after.hc_pre
    if (
        type(post_layer) is not HYV4HCPostLayer
        or type(pre_layer) is not HYV4HCPreLayer
        or post_layer.hpc_op is not None
        or pre_layer.hpc_op is not None
        or post_gates is None
        or not triton_ihc_supported(output)
    ):
        return None
    if (
        getattr(post_layer.forward, "__func__", None) is not HYV4HCPostLayer.forward
        or getattr(pre_layer.forward, "__func__", None) is not HYV4HCPreLayer.forward
        or getattr(before.post, "__func__", None) is not HYV4HCLayer.post
        or getattr(after.pre, "__func__", None) is not HYV4HCLayer.pre
        or getattr(after.prepare_input, "__func__", None)
        is not HYV4HCLayer.prepare_input
    ):
        return None
    if (
        nn_module._global_forward_hooks
        or nn_module._global_forward_pre_hooks
        or post_layer._forward_hooks
        or post_layer._forward_pre_hooks
        or pre_layer._forward_hooks
        or pre_layer._forward_pre_hooks
    ):
        return None
    if any(
        getattr(module, name, None)
        for module in (nn_module, post_layer, pre_layer)
        for name in (
            "_global_backward_hooks",
            "_global_backward_pre_hooks",
            "_backward_hooks",
            "_backward_pre_hooks",
        )
    ):
        return None
    from flaggems_vllm.ops.ihc_post_pre import can_ihc_post_pre, ihc_post_pre

    args = (
        output,
        residual,
        post_gates,
        pre_layer.hc_fn.weight,
        pre_layer.hc_scale,
        pre_layer.hc_base,
        pre_layer.magnitude,
        pre_layer.hc_eps,
        pre_layer.layernorm_epsilon,
    )
    if not can_ihc_post_pre(*args):
        return None
    return ihc_post_pre(*args)
