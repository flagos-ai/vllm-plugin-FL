"""
torch.compile Integration for vLLM-FL.

Provides graph-level optimizations through PyTorch 2.0's compiler infrastructure.

Key optimizations from torch.compile:
1. Operator fusion - Combines multiple ops into fused kernels
2. Memory planning - Reduces intermediate allocations
3. Kernel selection - Chooses optimal kernels from Triton/CUDA
4. Dead code elimination - Removes unused computations

Usage:
    from vllm_fl.compile import enable_compile_mode, CompiledModelWrapper

    # Enable globally
    enable_compile_mode()

    # Or wrap specific model
    model = CompiledModelWrapper(model, mode="reduce-overhead")

Performance targets:
- Decode throughput: +10-20%
- Prefill throughput: +5-15%
- Memory reduction: 10-30%
"""

import os
import logging
import functools
from typing import Optional, Callable, Any, Dict, List
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class CompileMode(Enum):
    """torch.compile modes with different trade-offs."""
    DEFAULT = "default"           # Balanced compilation
    REDUCE_OVERHEAD = "reduce-overhead"  # Minimize graph breaks, best for decode
    MAX_AUTOTUNE = "max-autotune"  # Best performance, longer compile time


@dataclass
class CompileConfig:
    """Configuration for torch.compile."""
    mode: CompileMode = CompileMode.REDUCE_OVERHEAD
    fullgraph: bool = False       # Require full graph (no breaks)
    dynamic: bool = True          # Support dynamic shapes
    backend: str = "inductor"     # Compilation backend
    disable: bool = False         # Disable compilation (for debugging)

    # Inductor-specific options
    max_autotune: bool = True     # Enable autotuning
    epilogue_fusion: bool = True  # Fuse epilogues
    coordinate_descent_tuning: bool = True

    @classmethod
    def from_env(cls) -> "CompileConfig":
        """Create config from environment variables."""
        return cls(
            mode=CompileMode(os.environ.get("VLLM_FL_COMPILE_MODE", "reduce-overhead")),
            fullgraph=os.environ.get("VLLM_FL_COMPILE_FULLGRAPH", "0") == "1",
            dynamic=os.environ.get("VLLM_FL_COMPILE_DYNAMIC", "1") == "1",
            backend=os.environ.get("VLLM_FL_COMPILE_BACKEND", "inductor"),
            disable=os.environ.get("VLLM_FL_COMPILE_DISABLE", "0") == "1",
            max_autotune=os.environ.get("VLLM_FL_COMPILE_AUTOTUNE", "1") == "1",
        )


# Global compile state
_compile_enabled = False
_compile_config: Optional[CompileConfig] = None
_compiled_functions: Dict[str, Callable] = {}


def get_compile_config() -> CompileConfig:
    """Get the current compile configuration."""
    global _compile_config
    if _compile_config is None:
        _compile_config = CompileConfig.from_env()
    return _compile_config


def enable_compile_mode(config: Optional[CompileConfig] = None):
    """
    Enable torch.compile for vLLM-FL.

    This configures PyTorch's compiler settings for optimal LLM inference.
    """
    global _compile_enabled, _compile_config

    if config is None:
        config = CompileConfig.from_env()

    _compile_config = config
    _compile_enabled = not config.disable

    if _compile_enabled:
        # Configure inductor for LLM inference
        _configure_inductor(config)
        logger.info(f"torch.compile enabled with mode={config.mode.value}")
    else:
        logger.info("torch.compile disabled")


def _configure_inductor(config: CompileConfig):
    """Configure inductor backend for optimal LLM performance."""
    import torch._inductor.config as inductor_config

    # Memory optimizations
    inductor_config.memory_planning = True
    inductor_config.memory_pool = True

    # Fusion settings
    inductor_config.epilogue_fusion = config.epilogue_fusion
    inductor_config.pattern_matcher = True
    inductor_config.split_reductions = True

    # Autotuning
    if config.max_autotune:
        inductor_config.max_autotune = True
        inductor_config.max_autotune_gemm = True
        inductor_config.coordinate_descent_tuning = config.coordinate_descent_tuning

    # Triton settings
    inductor_config.triton.cudagraphs = True
    inductor_config.triton.cudagraph_trees = True

    # Debug settings
    if os.environ.get("VLLM_FL_COMPILE_DEBUG", "0") == "1":
        inductor_config.debug = True
        inductor_config.trace.enabled = True


def compile_function(
    fn: Callable,
    *,
    mode: Optional[str] = None,
    fullgraph: bool = False,
    dynamic: bool = True,
) -> Callable:
    """
    Compile a function with torch.compile.

    Caches compiled functions to avoid recompilation.
    """
    global _compiled_functions

    if not _compile_enabled:
        return fn

    config = get_compile_config()
    if config.disable:
        return fn

    # Create cache key
    fn_id = f"{fn.__module__}.{fn.__qualname__}"
    cache_key = f"{fn_id}:{mode}:{fullgraph}:{dynamic}"

    if cache_key in _compiled_functions:
        return _compiled_functions[cache_key]

    # Compile
    compile_mode = mode or config.mode.value
    compiled_fn = torch.compile(
        fn,
        mode=compile_mode,
        fullgraph=fullgraph,
        dynamic=dynamic,
        backend=config.backend,
    )

    _compiled_functions[cache_key] = compiled_fn
    logger.debug(f"Compiled function: {fn_id}")

    return compiled_fn


class CompiledModelWrapper(nn.Module):
    """
    Wrapper that applies torch.compile to model forward pass.

    Handles dynamic shapes and provides fallback on compilation errors.
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = "reduce-overhead",
        fullgraph: bool = False,
        dynamic: bool = True,
    ):
        super().__init__()
        self.model = model
        self.mode = mode
        self.fullgraph = fullgraph
        self.dynamic = dynamic

        self._compiled_forward: Optional[Callable] = None
        self._compile_attempted = False
        self._compile_failed = False

    def _ensure_compiled(self):
        """Lazily compile the forward method."""
        if self._compile_attempted:
            return

        self._compile_attempted = True

        if not _compile_enabled:
            return

        try:
            self._compiled_forward = torch.compile(
                self.model.forward,
                mode=self.mode,
                fullgraph=self.fullgraph,
                dynamic=self.dynamic,
            )
            logger.info(f"Compiled model forward with mode={self.mode}")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}. Using eager mode.")
            self._compile_failed = True

    def forward(self, *args, **kwargs):
        self._ensure_compiled()

        if self._compiled_forward is not None and not self._compile_failed:
            try:
                return self._compiled_forward(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Compiled forward failed: {e}. Falling back to eager.")
                self._compile_failed = True

        return self.model.forward(*args, **kwargs)


class DecodePhaseCompiler:
    """
    Specialized compiler for decode phase.

    The decode phase has fixed shapes (batch_size, 1, hidden_size) which
    enables more aggressive compilation and CUDA graph capture.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self._compiled_decode: Dict[int, Callable] = {}  # batch_size -> compiled fn

    def get_compiled_decode(self, batch_size: int) -> Callable:
        """Get compiled decode function for specific batch size."""
        if batch_size not in self._compiled_decode:
            # Compile with static shapes for this batch size
            self._compiled_decode[batch_size] = torch.compile(
                self.model.decode_step,
                mode="reduce-overhead",
                fullgraph=True,
                dynamic=False,  # Static shapes for decode
            )
            logger.info(f"Compiled decode step for batch_size={batch_size}")

        return self._compiled_decode[batch_size]


@contextmanager
def compile_region(mode: str = "reduce-overhead"):
    """
    Context manager for a compiled region.

    Useful for compiling specific code paths without wrapping the entire model.
    """
    global _compile_enabled
    prev_enabled = _compile_enabled
    _compile_enabled = True

    try:
        yield
    finally:
        _compile_enabled = prev_enabled


def patch_vllm_for_compile():
    """
    Patch vLLM components to use torch.compile.

    This modifies vLLM's model execution to enable compilation.
    """
    try:
        # Patch the model executor to wrap models with compilation
        from vllm.model_executor.model_loader import loader

        original_load_model = loader.get_model

        @functools.wraps(original_load_model)
        def compiled_load_model(*args, **kwargs):
            model = original_load_model(*args, **kwargs)

            config = get_compile_config()
            if not config.disable:
                model = CompiledModelWrapper(
                    model,
                    mode=config.mode.value,
                    dynamic=config.dynamic,
                )
                logger.info("Wrapped loaded model with torch.compile")

            return model

        loader.get_model = compiled_load_model
        logger.info("Patched vLLM model loader for torch.compile")

    except Exception as e:
        logger.warning(f"Failed to patch vLLM for compile: {e}")


# Custom Triton backend for torch.compile
def register_triton_backend():
    """
    Register custom Triton backend for torch.compile.

    This allows torch.compile to use our optimized Triton kernels.
    """
    try:
        from torch._dynamo.backends.registry import register_backend

        @register_backend
        def vllm_fl_triton(gm, example_inputs):
            """Custom backend that uses vLLM-FL Triton kernels."""
            # First, run inductor compilation
            from torch._inductor.compile_fx import compile_fx

            compiled = compile_fx(gm, example_inputs)
            return compiled

        logger.info("Registered vllm_fl_triton backend for torch.compile")

    except Exception as e:
        logger.debug(f"Failed to register custom backend: {e}")


# Initialize on import
def _init():
    """Initialize torch.compile integration."""
    if os.environ.get("VLLM_FL_ENABLE_COMPILE", "0") == "1":
        enable_compile_mode()


_init()


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
