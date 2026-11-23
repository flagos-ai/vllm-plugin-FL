# Copyright (c) 2026 BAAI. All rights reserved.

"""Compilation configuration for Ascend graph execution."""

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig


def _configure_ascend_compilation(vllm_config: "VllmConfig") -> None:
    """Configure ordinary Ascend FULL decode graphs.

    V1 uses the eager FX backend; V2 captures the full model directly without
    torch.compile, using its own graph manager and stable input buffers.

    Ascend compiler options may be omitted or explicitly disabled with False.
    The six recognized rc1 compiler options have no implementation here; reject
    true values, non-booleans and unknown keys, including in eager mode.
    """
    from vllm.config import CompilationMode, CUDAGraphMode

    additional_config = vllm_config.additional_config
    if additional_config is None:
        additional_config = {}
    if not isinstance(additional_config, dict):
        raise TypeError("additional_config must be a dict")
    ascend_compilation = additional_config.get("ascend_compilation_config", {})
    if not isinstance(ascend_compilation, dict):
        raise TypeError("additional_config.ascend_compilation_config must be a dict")
    disabled_compiler_options = {
        "enable_npugraph_ex",
        "enable_static_kernel",
        "fuse_norm_quant",
        "fuse_qknorm_rope",
        "fuse_allreduce_rms",
        "fuse_muls_add",
    }
    for name, enabled in ascend_compilation.items():
        if name not in disabled_compiler_options:
            raise ValueError(f"Unknown ascend_compilation_config option: {name!r}")
        if not isinstance(enabled, bool):
            raise TypeError(f"ascend_compilation_config.{name} must be bool")
        if enabled:
            raise NotImplementedError(
                f"{name}=true is not supported by FL basic Ascend graph support"
            )

    compilation = vllm_config.compilation_config
    model_config = vllm_config.model_config
    enforce_eager = bool(model_config is not None and model_config.enforce_eager)
    if (
        model_config is None
        or enforce_eager
        or compilation.cudagraph_mode == CUDAGraphMode.NONE
    ):
        compilation.mode = CompilationMode.NONE
        compilation.cudagraph_mode = CUDAGraphMode.NONE
        return
    if compilation.cudagraph_mode != CUDAGraphMode.FULL_DECODE_ONLY:
        raise NotImplementedError(
            "FL basic Ascend graph support requires NONE or FULL_DECODE_ONLY"
        )
    # Match the runner's upstream switch (including vLLM's cached env value).
    # BreakableCUDAGraphWrapper bypasses our runner-owned graph task lifecycle.
    from vllm.envs import VLLM_USE_BREAKABLE_CUDAGRAPH

    if VLLM_USE_BREAKABLE_CUDAGRAPH:
        raise NotImplementedError(
            "VLLM_USE_BREAKABLE_CUDAGRAPH=1 is not supported by FL Ascend graphs; "
            "ordinary FULL_DECODE_ONLY graphs require the runner-owned runtime"
        )
    if vllm_config.parallel_config.use_ubatching:
        raise NotImplementedError(
            "FL basic Ascend FULL decode graphs do not support ubatching"
        )
    if compilation.mode not in (
        None,
        CompilationMode.NONE,
        CompilationMode.VLLM_COMPILE,
    ):
        raise NotImplementedError(
            "FL Ascend graph compilation supports VLLM_COMPILE or NONE"
        )
    if os.environ.get("ASCEND_LAUNCH_BLOCKING", "0") == "1":
        raise ValueError("ACL graph is incompatible with ASCEND_LAUNCH_BLOCKING=1")
    if (
        getattr(compilation, "backend", None) == "inductor"
        and compilation.mode != CompilationMode.NONE
    ):
        raise ValueError(
            "Ascend graph execution requires compilation backend='eager'; "
            "the Inductor backend is not supported on torch_npu."
        )
    compilation.mode = (
        CompilationMode.NONE
        if getattr(vllm_config, "use_v2_model_runner", False)
        else CompilationMode.VLLM_COMPILE
    )
    compilation.backend = "eager"
    compilation.use_inductor = False
    compilation.use_inductor_graph_partition = False
    compilation.splitting_ops = []
    compilation.cudagraph_num_of_warmups = 1
    compilation.pass_config.fuse_norm_quant = False
    compilation.pass_config.fuse_act_quant = False
    compilation.pass_config.fuse_attn_quant = False
