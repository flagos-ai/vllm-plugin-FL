# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Keep image routing IDs aligned with native naive DP/EP token dispatch."""

from functools import wraps
from inspect import signature

from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_ep_group
from vllm.distributed.parallel_state import GroupCoordinator
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.models.deepseek_v41.common.mm_preprocess import IMAGE_SENTINEL_BASE_ID

logger = init_logger(__name__)


def _eager_ep_fence():
    # MACA lazy device-image loading performs a synchronous SDMA copy. Peers
    # must not queue the next EP collective until local modules are prepared.
    # Native DP execution synchronizes eager/graph mode across ranks. During
    # capture FULL skips this host barrier; replay contains no Python barrier.
    if get_forward_context().cudagraph_runtime_mode == CUDAGraphMode.NONE:
        logger.info_once(
            "V4.1 eager EP preparation fences enabled; FULL graph capture skips host fences"
        )
        get_ep_group().barrier()


def _wrap_forward_impl(original):
    @wraps(original)
    def forward(
        self, hidden_states, router_logits, shared_experts_input, input_ids=None
    ):
        if (
            type(self._quant_method).__name__ != "MetaXCompressedTensorsW4A8Method"
            or not self.do_naive_dispatch_combine
        ):
            return original(
                self, hidden_states, router_logits, shared_experts_input, input_ids
            )
        image_route = self.router.image_sentinel_lo == IMAGE_SENTINEL_BASE_ID
        draft_route = (
            self.router.image_sentinel_lo == 0
            and self.router.global_num_experts == 128
            and self.moe_config.hidden_dim == 5120
        )
        scoped = self.router.bias_vl is not None and (image_route or draft_route)
        if not scoped:
            return original(
                self, hidden_states, router_logits, shared_experts_input, input_ids
            )
        # These combinations already fail the model's initial-adaptation contract;
        # reject before a gate, staging write or collective starts.
        if (
            self.moe_config.pcp_size != 1
            or self.moe_config.is_sequence_parallel
            or self._fse_fuse_gate
        ):
            raise NotImplementedError(
                "V4.1 image-ID EP dispatch requires PCP1, no SP or fused shared gate"
            )
        if image_route and (
            input_ids is None
            or input_ids.ndim != 1
            or input_ids.shape[0] != hidden_states.shape[0]
        ):
            raise ValueError(
                "V4.1 EP routing requires one local raw token ID per hidden row"
            )
        if image_route:
            logger.info_once(
                "V4.1 W4A8 EP dispatch transports raw routing IDs with hidden states and logits"
            )
        self.routed_experts._ensure_moe_quant_config_init()
        overlapping = False
        if self._shared_experts is not None:
            overlapping = self._shared_experts.maybe_forward_async(shared_experts_input)
        if self.gate is not None:
            router_logits, _ = self.gate(hidden_states)
        with self._sequence_parallel_context():
            _eager_ep_fence()
            if image_route:
                hidden_states, router_logits, extra = (
                    get_ep_group().dispatch_router_logits(
                        hidden_states, router_logits, False, extra_tensors=[input_ids]
                    )
                )
                input_ids = extra[0]
            else:
                hidden_states, router_logits = self._maybe_dispatch(
                    hidden_states, router_logits
                )
            shared_output, output = self._apply_quant_method(
                hidden_states=hidden_states,
                router_logits=router_logits,
                shared_experts_input=shared_experts_input,
                input_ids=input_ids,
                shared_experts_overlapping=overlapping,
            )
            _eager_ep_fence()
            return self._maybe_combine(shared_output, output)

    return forward


def prepare():
    original = MoERunner._forward_impl
    if getattr(original, "_fl_v41_ep_ids", False):
        return []
    if list(signature(original).parameters) != [
        "self",
        "hidden_states",
        "router_logits",
        "shared_experts_input",
        "input_ids",
    ]:
        raise RuntimeError("unsupported native MoERunner forward contract")
    if (
        "extra_tensors"
        not in signature(GroupCoordinator.dispatch_router_logits).parameters
    ):
        raise RuntimeError("native EP dispatcher lacks extra tensor transport")
    if not {
        "hidden_states",
        "router_logits",
        "shared_experts_input",
        "input_ids",
        "shared_experts_overlapping",
    }.issubset(signature(MoERunner._apply_quant_method).parameters):
        raise RuntimeError("unsupported native MoE apply contract")
    replacement = _wrap_forward_impl(original)
    replacement._fl_v41_ep_ids = True
    return [(MoERunner, "_forward_impl", replacement)]
