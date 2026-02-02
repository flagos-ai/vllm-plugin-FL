"""
torch.compile integration for vLLM-FL.

Provides graph-level optimizations through PyTorch 2.0's compiler.
"""

from .torch_compile_integration import (
    CompileConfig,
    CompileMode,
    enable_compile_mode,
    compile_function,
    CompiledModelWrapper,
    DecodePhaseCompiler,
    compile_region,
    patch_vllm_for_compile,
    get_compile_config,
)

__all__ = [
    'CompileConfig',
    'CompileMode',
    'enable_compile_mode',
    'compile_function',
    'CompiledModelWrapper',
    'DecodePhaseCompiler',
    'compile_region',
    'patch_vllm_for_compile',
    'get_compile_config',
]
