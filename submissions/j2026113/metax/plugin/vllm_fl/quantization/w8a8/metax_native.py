"""C500 W8A8 feasibility adapter; not a standalone competition innovation.

Keeps the framework candidate registry and the existing quantization contract.
The native implementation is explicit, so failures never substitute another op.
"""

import torch
from w8a8_submission_route_r1239 import launch
from vllm.logger import init_logger
from vllm.model_executor.kernels.linear import Int8ScaledMMLinearKernel
from vllm.model_executor.layers.quantization.utils import replace_parameter

logger = init_logger(__name__)




@torch.library.custom_op("fl_c500::dynamic_w8a8_linear", mutates_args=())
def c500_dynamic_w8a8_linear(x: torch.Tensor, weight: torch.Tensor,
                           weight_scale: torch.Tensor,
                           bias: torch.Tensor | None = None) -> torch.Tensor:
    from vllm_fl.dispatch.backends.flaggems.impl.quantization import (
        dynamic_per_token_quant_int8_flaggems_triton,
    )
    if x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("C500 feasibility adapter requires 2D BF16 activations")
    m, k = x.shape
    n = weight.shape[0]
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m == 0:
        return out
    a, sa = dynamic_per_token_quant_int8_flaggems_triton(x.contiguous())
    if bias is not None:
        raise ValueError("The MiniCPM5 C500 W8A8 route requires bias-free linear layers")
    launch(a, weight, sa, weight_scale, out, m, n, k)
    return out


@c500_dynamic_w8a8_linear.register_fake
def _fake_c500_linear(x, weight, weight_scale, bias=None):
    return x.new_empty((x.shape[0], weight.shape[0]))


class C500W8A8FeasibilityKernel(Int8ScaledMMLinearKernel):
    @classmethod
    def is_supported(cls, compute_capability=None):
        import flag_gems
        if flag_gems.runtime.device.vendor_name != "metax":
            return False, "requires MetaX"
        if "C500" not in torch.cuda.get_device_name():
            return False, "currently validated for C500 only"
        return True, None

    @classmethod
    def can_implement(cls, config):
        if not config.is_channelwise or config.is_static_input_scheme or not config.input_symmetric:
            return False, "requires channelwise weights and dynamic symmetric token INT8"
        return True, None

    def process_weights_after_loading(self, layer):
        w_name, s_name, i_name, z_name, azp_name = self.layer_param_names
        weight = getattr(layer, w_name)
        scale = getattr(layer, s_name)
        if weight.dtype != torch.int8 or weight.ndim != 2:
            raise ValueError("expected serialized INT8 [N,K]")
        if scale.numel() != weight.shape[0]:
            raise ValueError("expected one FP32 scale per output channel")
        replace_parameter(layer, w_name, torch.nn.Parameter(weight.detach().contiguous(), requires_grad=False))
        replace_parameter(layer, s_name, torch.nn.Parameter(
            scale.detach().reshape(-1, 1).float().contiguous(), requires_grad=False))
        setattr(layer, i_name, None)
        setattr(layer, z_name, None)
        setattr(layer, azp_name, None)
        logger.info_once("C500 W8A8 feasibility adapter selected; dynamic-token INT8, Triton scaled GEMM")

    def apply_weights(self, layer, x, bias=None):
        w, s, _, _, _ = self._get_layer_params(layer)
        shape = x.shape
        y = c500_dynamic_w8a8_linear(x.reshape(-1, shape[-1]), w, s, bias)
        return y.reshape(*shape[:-1], w.shape[0])


def register_c500_feasibility_kernel(registry):
    from vllm.platforms import PlatformEnum
    candidates = registry.setdefault(PlatformEnum.OOT, [])
    if C500W8A8FeasibilityKernel not in candidates:
        candidates.insert(0, C500W8A8FeasibilityKernel)

