from . import accelerator_compat
from . import functorch_config_patch
from . import fix_standalone_compile
from . import pynccl_wrapper
from . import cuda_wrapper
from . import utils_patch
from . import chunk_delta_h
from . import topk_topp_sampler
from . import sampler_tail_fusion
from . import linear_nn_repack
from . import slow_step_logger
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention
from . import mm_autotune_warmup  # noqa: F401 — keep last: warmup is optional

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu
iu.has_triton_kernels = lambda: False
