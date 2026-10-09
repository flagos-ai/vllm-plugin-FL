# Copyright (c) 2026 BAAI. All rights reserved.


def patch_triton_compile_hooks() -> None:
    """Allow vLLM 0.28's JIT monitor to observe Ascend compilations.

    FlagTree 0.6.2a1's hook formatter reads this CUDA option unconditionally,
    but NPUOptions omits it. NPU launches never use cooperative CUDA grids;
    provide the hook metadata without changing the NPU compilation options.
    """
    from triton.backends.ascend.compiler import NPUOptions

    if not hasattr(NPUOptions, "launch_cooperative_grid"):
        NPUOptions.launch_cooperative_grid = False
