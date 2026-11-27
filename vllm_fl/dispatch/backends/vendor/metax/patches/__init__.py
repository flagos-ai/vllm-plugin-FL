# MACA command dispatch mode, chosen by vllm_fl._metax_runtime_env at plugin
# import (VLLM_FL_MX_DIRECT_DISPATCH).  Report what this process runs with,
# before the patches below (some of them need a device).
import os as _os

from vllm.logger import init_logger as _init_logger
from vllm_fl import _metax_runtime_env as _mx_runtime_env

_init_logger(__name__).info(
    "MetaX: MACA_DIRECT_DISPATCH=%s (%s)",
    _os.environ.get("MACA_DIRECT_DISPATCH", "unset, runtime default"),
    _mx_runtime_env.STATE,
)

from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import fused_sampler  # noqa: F401 -- fused top-p sampling (VLLM_FL_METAX_FUSED_SAMPLER)
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False
