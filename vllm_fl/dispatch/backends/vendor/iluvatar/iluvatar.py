# Copyright (c) 2026 BAAI. All rights reserved.

"""
ILUVATAR backend implementation.

This backend provides operator implementations for Iluvatar GPUs.
Iluvatar uses a CUDA-compatible architecture.
"""

from __future__ import annotations

import logging
from typing import Optional, Union

import torch

from vllm_fl.dispatch.backends.base import Backend

logger = logging.getLogger(__name__)


def patch_triton_language_for_iluvatar() -> None:
    """Add make_tensor_descriptor stub to triton.language for triton < 3.3.

    triton JIT's DependencyFinder walks all @triton.jit function bodies at
    cache_key computation time (first compile, not import). Even dead-code
    branches guarded by ``USE_TD: tl.constexpr = False`` cause AttributeError
    when ``tl.make_tensor_descriptor`` does not exist, because DependencyFinder
    resolves ``tl.*`` attributes to compute a stable kernel hash.

    The _Stub below is a plain Python callable + hashable object that satisfies
    attribute lookup. It raises at call-time so accidental USE_TD=True usage is
    caught immediately.

    TODO: Remove once minimum supported triton version is >= 3.3.
    """
    try:
        import triton.language as tl

        if hasattr(tl, "make_tensor_descriptor"):
            return  # triton >= 3.3, nothing to do

        class _TensorDescriptorStub:
            """Stub for tl.make_tensor_descriptor — callable and hashable."""

            def __call__(self, *args, **kwargs):
                raise RuntimeError(
                    "tl.make_tensor_descriptor is not available on triton < 3.3. "
                    "Ensure VLLM_TRITON_ATTN_USE_TD=0 (the default on iluvatar)."
                )

            def __hash__(self):
                return hash("_tl_make_tensor_descriptor_stub")

            def __repr__(self):
                return "<tl.make_tensor_descriptor stub for triton<3.3>"

        tl.make_tensor_descriptor = _TensorDescriptorStub()
        logger.info(
            "Patched triton.language: added make_tensor_descriptor stub "
            "(triton < 3.3 detected; USE_TD=False assumed on iluvatar)."
        )
    except Exception as e:
        logger.warning("Failed to patch triton.language for iluvatar: %s", e)


def patch_triton_chained_or_for_iluvatar() -> None:
    """Rewrite chained boolean 'or' chains in vllm triton kernels.

    Iluvatar's triton version (3.2.x) raises UnsupportedLanguageConstruct for
    chained boolean operators like ``A or B or C`` inside @triton.jit functions.
    Rewrites the affected source file in-place (idempotent).

    Must be called from the main process BEFORE Worker subprocesses start,
    so the patched file is on disk when Workers import the module.

    Applies only when triton < 3.3 — later versions support chained boolean
    operators natively.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    import re
    import importlib.util
    import pathlib
    import sys

    # Only needed for triton < 3.3.
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot determine triton version, "
            "applying patch defensively: %s", e
        )

    _MARKER = "# _iluvatar_chained_or_patched"

    spec = importlib.util.find_spec("vllm.v1.attention.ops.triton_attention_helpers")
    if spec is None or spec.origin is None:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: "
            "vllm.v1.attention.ops.triton_attention_helpers not found, skipping."
        )
        return

    fpath = pathlib.Path(spec.origin)
    try:
        src = fpath.read_text()
    except Exception as e:
        logger.warning("patch_triton_chained_or_for_iluvatar: cannot read %s: %s", fpath, e)
        return

    if _MARKER in src:
        return  # already patched

    # Replace: A or B or C  →  (A or B) or C
    _OPERAND = r'(?:not\s+)?(?:\([^)]*\)|\w+)'
    pattern = re.compile(
        r'(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')\s+or\s+(' + _OPERAND + r')'
    )

    def _rewrite(m: re.Match) -> str:
        return f"({m.group(1)} or {m.group(2)}) or {m.group(3)}"

    new_src, count = re.subn(pattern, _rewrite, src)
    if count == 0:
        return  # nothing to patch

    new_src += f"\n{_MARKER}\n"
    try:
        fpath.write_text(new_src)
    except Exception as e:
        logger.warning(
            "patch_triton_chained_or_for_iluvatar: cannot write %s: %s", fpath, e
        )
        return

    # Clear pycache so Python and triton both see the patched source.
    pycache = fpath.parent / "__pycache__"
    if pycache.exists():
        import shutil
        try:
            shutil.rmtree(pycache)
        except Exception:
            pass  # non-fatal

    # Evict from sys.modules so this process reimports the patched source.
    sys.modules.pop("vllm.v1.attention.ops.triton_attention_helpers", None)

    logger.info(
        "patch_triton_chained_or_for_iluvatar: rewrote %d chained-or expression(s) in %s",
        count, fpath,
    )


def patch_triton_perf_model_for_iluvatar() -> None:
    """
    Patch triton.ops.matmul_perf_model.get_clock_rate_in_khz for iluvatar.

    On iluvatar, neither `ixsmi` nor `libnvidia-ml.so` is available, so the
    default implementation crashes. We replace it with a fixed value that
    matches typical iluvatar BI-V150 SM clock (1500 MHz = 1500000 kHz).

    The function is decorated with @functools.lru_cache so assigning a new
    callable to the module attribute is sufficient — callers import the name
    directly, so we also patch the reference in the testing helper.

    TODO: Remove once iluvatar triton natively handles missing nvsmi/nvml.
    """
    try:
        import triton.ops.matmul_perf_model as _mpm

        if getattr(_mpm, '_iluvatar_clock_patched', False):
            return

        import functools

        @functools.lru_cache()
        def _iluvatar_get_clock_rate_in_khz():
            # BI-V150 SM clock ~1500 MHz; used only for perf-model heuristics.
            return 1500 * 1e3

        _mpm.get_clock_rate_in_khz = _iluvatar_get_clock_rate_in_khz
        _mpm._iluvatar_clock_patched = True
        logger.info(
            "Patched triton.ops.matmul_perf_model.get_clock_rate_in_khz "
            "for iluvatar (fixed 1500 MHz, no nvsmi/nvml available)."
        )
    except Exception as e:
        logger.warning(
            "Failed to patch triton perf model for iluvatar: %s", e
        )

def patch_triton_testing_tflops_for_iluvatar() -> None:
    """Patch triton.ops.matmul_perf_model.get_max_tensorcore_tflops for iluvatar.

    On BI-V150, torch.cuda.get_device_capability() returns a value < 8,
    which triggers `assert dtype == torch.float16` in the original flagtree
    triton implementation. BF16 models fail this assert.

    NOTE: patching triton.testing.get_max_tensorcore_tflops is insufficient
    because matmul_perf_model.py binds the function directly via
    `from ..testing import get_max_tensorcore_tflops`, so the module-level
    name in matmul_perf_model must be patched directly.

    TODO: Remove once flagtree triton handles non-NVIDIA capability correctly.
    """
    try:
        import triton.testing as _tt
        import triton.ops.matmul_perf_model as _mpm_ops
        import torch as _torch
        if getattr(_mpm_ops, "_iluvatar_tflops_patched", False):
            return
        from triton.runtime import driver as _driver

        def _get_max_tensorcore_tflops(dtype, clock_rate, device=None):
            if not device:
                device = _torch.cuda.current_device()
            num_subcores = (
                _driver.active.utils.get_device_properties(device)
                ["multiprocessor_count"] * 4)
            if dtype in [_torch.float32, _torch.int32]:
                ops_per_sub_core = 256
            elif dtype in [_torch.float16, _torch.bfloat16, _torch.int16]:
                ops_per_sub_core = 512
            else:
                ops_per_sub_core = 512  # safe fallback for unknown dtypes
            return num_subcores * clock_rate * ops_per_sub_core * 1e-9

        # Patch both the testing module and the matmul_perf_model module
        # (the latter binds the function directly at import time).
        _tt.get_max_tensorcore_tflops = _get_max_tensorcore_tflops
        _mpm_ops.get_max_tensorcore_tflops = _get_max_tensorcore_tflops
        _mpm_ops._iluvatar_tflops_patched = True
        logger.info(
            "Patched triton.ops.matmul_perf_model.get_max_tensorcore_tflops "
            "for iluvatar (bfloat16 support, no capability assert)."
        )
    except Exception as e:
        logger.warning(
            "Failed to patch triton.testing.get_max_tensorcore_tflops "
            "for iluvatar: %s", e
        )

patch_triton_chained_or_for_iluvatar()


def patch_sampler_compile_for_iluvatar() -> None:
    # Disable torch.compile on vllm sampler ops for Iluvatar.
    # flagtree triton only supports Iluvatar backend, not cuda target.
    # TODO: Remove once flagtree triton supports cuda inductor target.
    try:
        import importlib
        _tts = importlib.import_module('vllm.v1.sample.ops.topk_topp_sampler')
        if not getattr(_tts, '_iluvatar_compile_patched', False):
            _orig = _tts.compiled_random_sample
            _tts.compiled_random_sample = getattr(_orig, '__wrapped__', _orig)
            _tts._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped topk_topp_sampler.compiled_random_sample')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (topk_topp): %s', e)
    try:
        import importlib
        _lp = importlib.import_module('vllm.v1.sample.ops.logprobs')
        if not getattr(_lp, '_iluvatar_compile_patched', False):
            _orig = _lp.batched_count_greater_than
            _lp.batched_count_greater_than = getattr(_orig, '__wrapped__', _orig)
            _lp._iluvatar_compile_patched = True
            logger.info('patch_sampler_compile_for_iluvatar: unwrapped logprobs.batched_count_greater_than')
    except Exception as e:
        logger.warning('patch_sampler_compile_for_iluvatar (logprobs): %s', e)

patch_sampler_compile_for_iluvatar()


def patch_torch_inductor_for_iluvatar() -> None:
    """Patch torch._inductor GPUTarget to use the correct triton backend name.

    torch._inductor always passes device_type='cuda' to GPUTarget, but
    flagtree triton only registers an 'iluvatar' (3.6.x) or 'corex' (3.2.x)
    backend -- never 'cuda'.

    Fix: subclass GPUTarget to intercept 'cuda' and remap it to whatever
    backend name flagtree triton actually registered.

    Compatibility: safe to call when not using flagtree -- if triton
    has a 'cuda' backend or no backends at all, the patch is skipped.

    Hardware gate: Iluvatar only.
    TODO: Remove once torch._inductor or flagtree natively handles this.
    """
    try:
        import triton.backends as _tb
        _registered = list(getattr(_tb, 'backends', {}).keys())
        # If 'cuda' is already registered (standard triton), no patch needed
        if 'cuda' in _registered or not _registered:
            logger.debug('patch_torch_inductor_for_iluvatar: triton has cuda backend or no backends, skipping')
            return
        # Probe the actual target string expected by supports_target().
        # In flagtree triton 3.6.x, the registered backend name is 'iluvatar'
        # but its supports_target() checks backend == 'corex', so we must
        # discover the correct probe string at runtime.
        _target = None
        try:
            from triton.backends.compiler import GPUTarget as _GPUTarget
            for _probe in ('corex', 'iluvatar') + tuple(_registered):
                try:
                    _t = object.__new__(_GPUTarget)
                    _GPUTarget.__init__(_t, _probe, 90, False)
                    for _bname in _registered:
                        if _tb.backends[_bname].compiler.supports_target(_t):
                            _target = _probe
                            break
                except Exception:
                    pass
                if _target:
                    break
        except Exception:
            pass
        if not _target:
            _target = 'corex'  # safe default for all known flagtree versions
    except Exception:
        logger.debug('patch_torch_inductor_for_iluvatar: cannot inspect triton backends, skipping')
        return

    try:
        import torch._inductor.runtime.triton_heuristics as _th
        from triton.backends.compiler import GPUTarget as _OrigGPUTarget

        if getattr(_th, '_iluvatar_gputarget_patched', False):
            return

        _remap_target = _target

        class _IluvatarGPUTarget(_OrigGPUTarget):
            """GPUTarget wrapper that remaps 'cuda' to the flagtree backend."""
            def __new__(cls, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                return super().__new__(cls)

            def __init__(self, backend, *args, **kwargs):
                if backend == 'cuda':
                    backend = _remap_target
                super().__init__(backend, *args, **kwargs)

        _th.GPUTarget = _IluvatarGPUTarget
        _th._iluvatar_gputarget_patched = True
        logger.info(
            "Patched torch._inductor GPUTarget: 'cuda' -> '%s' (iluvatar)",
            _remap_target,
        )
    except Exception as e:
        logger.warning(
            "Failed to patch torch._inductor for iluvatar triton backend: %s", e
        )

patch_torch_inductor_for_iluvatar()


class IluvatarBackend(Backend):
    """
    Iluvatar backend for operator implementations.

    This backend uses Iluvatar libraries to provide high-performance
    operator implementations for Iluvatar GPUs.
    """

    _available: Optional[bool] = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def vendor(self) -> Optional[str]:
        return "iluvatar"

    def is_available(self) -> bool:
        """
        Check if Iluvatar hardware and libraries are available.

        This method uses the platform's vendor information to determine
        if the device is an Iluvatar GPU.
        """
        if IluvatarBackend._available is None:
            try:
                from vllm.platforms import current_platform
                # Iluvatar GPUs should be detected via vendor_name
                if hasattr(current_platform, 'vendor_name') and current_platform.vendor_name == "iluvatar":
                    IluvatarBackend._available = True
                else:
                    # Fallback: check if CUDA is available with iluvatar device
                    if torch.cuda.is_available():
                        # Try to detect Iluvatar GPU
                        # Iluvatar GPUs typically expose CUDA-compatible interface
                        # We can check device name if available
                        device_name = torch.cuda.get_device_name(0)
                        if "iluvatar" in device_name.lower():
                            IluvatarBackend._available = True
                        else:
                            IluvatarBackend._available = False

                    else:
                        IluvatarBackend._available = False
            except Exception:
                IluvatarBackend._available = False
        return IluvatarBackend._available

    # ==================== Operator Implementations ====================

    def silu_and_mul(self, obj, x: torch.Tensor) -> torch.Tensor:
        """
        SiLU activation followed by element-wise multiplication.

        Args:
            obj: The calling obj (for interface consistency)
            x: Input tensor of shape [..., 2*d]

        Returns:
            Output tensor of shape [..., d]
        """
        from .impl.activation import silu_and_mul_iluvatar

        return silu_and_mul_iluvatar(obj, x)

    def rms_norm(
        self,
        obj,
        x: torch.Tensor,
        residual: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        RMS normalization.

        Args:
            obj: The calling obj (e.g., RMSNorm layer)
            x: Input tensor
            residual: Optional residual tensor

        Returns:
            Normalized tensor, or tuple of (normalized, residual) if residual is provided
        """
        from .impl.normalization import rms_norm_iluvatar

        return rms_norm_iluvatar(obj, x, residual)

    def rotary_embedding(
        self,
        obj,
        query: torch.Tensor,
        key: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor,
        rotary_interleaved: bool = False,
        inplace: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply rotary position embedding.

        Args:
            obj: The calling obj (for interface consistency)
            query: Query tensor
            key: Key tensor
            cos: Cosine cache
            sin: Sine cache
            position_ids: Position indices
            rotary_interleaved: Whether to use interleaved rotary
            inplace: Whether to modify tensors in-place

        Returns:
            Tuple of (embedded_query, embedded_key)
        """
        from .impl.rotary import rotary_embedding_iluvatar

        return rotary_embedding_iluvatar(
            obj,
            query,
            key,
            cos,
            sin,
            position_ids,
            rotary_interleaved=rotary_interleaved,
            inplace=inplace,
        )

    def attention_backend(self, use_mla: bool = False, use_sparse: bool = False) -> str:
        """
        Get the attention backend class path for Iluvatar.

        Args:
            use_mla: Whether to use Multi-head Latent Attention (MLA)
            use_sparse: Whether to use Deepseek Sparse Attention (DSA)

        Returns:
            Fully qualified class path string
        """
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        if use_mla:
            if use_sparse:
                return AttentionBackendEnum.FLASHMLA_SPARSE.get_path()
            return AttentionBackendEnum.FLASHMLA.get_path()

        # flash_attn is not available on iluvatar. Use TRITON_ATTN (the vllm
        # default), but patch tl.make_tensor_descriptor so that triton JIT's
        # DependencyFinder can hash the kernel without AttributeError on
        # triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
        # the stub is never called.
        patch_triton_language_for_iluvatar()
        patch_triton_perf_model_for_iluvatar()
        return AttentionBackendEnum.TRITON_ATTN.get_path()
