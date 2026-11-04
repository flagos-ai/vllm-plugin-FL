# Import order preserves the existing vendor initialization dependencies.
# isort: off
from . import accelerator_compat as accelerator_compat
from . import functorch_config_patch as functorch_config_patch
from . import fix_standalone_compile as fix_standalone_compile
from . import pynccl_wrapper as pynccl_wrapper
from . import cuda_wrapper as cuda_wrapper
from . import utils_patch as utils_patch
from . import chunk_delta_h as chunk_delta_h
from . import topk_topp_sampler
from . import gdn_linear_attn  # noqa: F401 — register MacaGatedDeltaNetAttention
# isort: on

topk_topp_sampler.install_metax_sampling_patches()

# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu  # noqa: E402

iu.has_triton_kernels = lambda: False
