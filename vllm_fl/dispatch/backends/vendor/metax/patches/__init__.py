from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False


def _install_metax_fast_triton_ops():
    import logging as _logging
    import torch as _torch
    from flag_gems.fused.metax_fast_ops import (
        fast_rms_forward,
        fast_rope_forward,
        fast_silu_and_mul,
    )
    _log = _logging.getLogger(__name__)
    _done = {"rope": False, "rms": False, "silu": False}

    try:
        from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
        _orig_rope_cuda = RotaryEmbedding.forward_cuda

        def _metax_rope_cuda(self, positions, query, key=None):
            if key is not None and self.is_neox_style and self.rotary_dim == self.head_size == 128:
                if not _done["rope"]:
                    _done["rope"] = True
                    _log.warning("E469 metax pure-Triton RotaryEmbedding.forward_cuda active")
                self.cos_sin_cache = self.cos_sin_cache.to(positions.device)
                cos, sin = self.cos_sin_cache.chunk(2, dim=-1)
                return fast_rope_forward(query, key, cos, sin, positions, False, True)
            return _orig_rope_cuda(self, positions, query, key)

        RotaryEmbedding.forward_cuda = _metax_rope_cuda
        RotaryEmbedding.forward_oot = _metax_rope_cuda
    except Exception as _e:
        _log.warning("E469 rope patch skipped: %s", _e)

    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
        _orig_rms_cuda = RMSNorm.forward_cuda

        def _metax_rms_cuda(self, x, residual=None):
            if x.shape[-1] == 2048 and self.weight is not None:
                if not _done["rms"]:
                    _done["rms"] = True
                    _log.warning("E469 metax pure-Triton RMSNorm.forward_cuda active")
                return fast_rms_forward(x, residual, self.weight, self.variance_epsilon)
            return _orig_rms_cuda(self, x, residual)

        RMSNorm.forward_cuda = _metax_rms_cuda
        RMSNorm.forward_oot = _metax_rms_cuda
    except Exception as _e:
        _log.warning("E469 rms patch skipped: %s", _e)

    try:
        from vllm.model_executor.layers.activation import SiluAndMul
        _orig_silu_cuda = SiluAndMul.forward_cuda

        def _metax_silu_cuda(self, x):
            if x.shape[-1] % 2 == 0:
                if not _done["silu"]:
                    _done["silu"] = True
                    _log.warning("E469 metax pure-Triton SiluAndMul.forward_cuda active")
                return fast_silu_and_mul(x)
            return _orig_silu_cuda(self, x)

        SiluAndMul.forward_cuda = _metax_silu_cuda
        SiluAndMul.forward_oot = _metax_silu_cuda
    except Exception as _e:
        _log.warning("E469 silu patch skipped: %s", _e)


_install_metax_fast_triton_ops()
