# --------------------------------------------------
# MetaX C550 does not support third-party Triton kernels (Triton upgrade required).
# Disable them so FLA decode ops (fused_recurrent_gated_delta_rule etc.) fall back
# to the non-Triton path handled by mcoplib, producing correct output.
# TODO: remove when MetaX Triton support is available.
import vllm.utils.import_utils as iu

# Every module here applies its patch as an import side effect; the names are
# deliberately unused.
from . import (
    accelerator_compat,  # noqa: F401 — torch 2.8 accelerator memory APIs
    chunk_delta_h,  # noqa: F401 — maca shmem-safe chunk_delta_h
    cuda_wrapper,  # noqa: F401 — maca cudart wrapper
    fix_standalone_compile,  # noqa: F401 — torch 2.8 standalone compile hotfix
    functorch_config_patch,  # noqa: F401 — torch 2.8 functorch config key shim
    gdn_linear_attn,  # noqa: F401 — register MacaGatedDeltaNetAttention
    pynccl_wrapper,  # noqa: F401 — MCCL instead of NCCL; traceable all_reduce
    topk_topp_sampler,  # noqa: F401 — pytorch topk_topp fallback
    utils_patch,  # noqa: F401 — find mccl library
    vllm024_compat,  # noqa: F401 — vLLM 0.24.0 shims (load_ptr, penalties, UVA)
)

iu.has_triton_kernels = lambda: False
