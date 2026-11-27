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

    Applies only when triton < 3.3 — later versions handle missing nvsmi/nvml
    natively or remove triton.ops.matmul_perf_model entirely.

    TODO: Remove once minimum supported Iluvatar triton version is >= 3.3.
    """
    try:
        import triton as _triton
        _tv = tuple(int(x) for x in _triton.__version__.split(".")[:2])
        if _tv >= (3, 3):
            return

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

patch_triton_chained_or_for_iluvatar()
patch_triton_language_for_iluvatar()
patch_triton_perf_model_for_iluvatar()


def patch_unified_attention_for_iluvatar() -> None:
    """Install the BI-V150 unified-attention kernel over vLLM's.

    Thin forwarder so the call site stays next to the other Iluvatar runtime
    patches; the kernel copy and the launch-parameter choice live in
    ``kernels/unified_attention_iluvatar.py`` (that file documents both the
    corex ``exp(-inf)`` miscompilation it works around and the measured
    speedups).  Vendor-gated inside the callee and idempotent.  Raises on
    failure: vLLM's own kernel gives wrong prefill attention on this device.

    Call it from ``WorkerFL.init_device`` -- it must run in the process that
    owns the model, before ``load_model`` and before CUDA-graph capture.  It is
    deliberately *not* a module-level patch: this module is imported from
    ``PlatformFL.pre_register_and_update`` while the platform is still being
    resolved, and the kernel module touches ``current_platform`` at import.
    """
    from .kernels.unified_attention_iluvatar import (
        patch_unified_attention_for_iluvatar as _install,
    )

    _install()


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


# Per-device byte budget for the top-k/top-p kernel's per-program scratch.
# The kernel keeps one vocab-sized fp32 scratch row per program
# (``BUFFER + pid * VOCAB_SIZE``), so this is what bounds how many programs
# the policy below is willing to ask for.  32 MiB is ~0.1% of BI-V150's
# 32 GiB; ``enable_topp_scratch_policy`` below keeps it off the KV-cache
# budget.  At a 130k vocabulary it allows 64 concurrent rows.
_ILUVATAR_SAMPLER_SCRATCH_BYTES = 32 << 20

# The policy is armed only once the KV cache exists -- see
# ``enable_topp_scratch_policy``.
_ILUVATAR_TOPP_ARMED = False


def enable_topp_scratch_policy() -> None:
    """Arm the NUM_PROGRAMS policy, to be called after KV-cache allocation.

    vLLM sizes the KV cache from the *peak* torch allocation of the memory
    profiling run (``allocated_bytes.all.peak``), and that run ends in a dummy
    sampler pass.  So whatever scratch the top-p kernel caches during
    profiling is subtracted from the KV cache, and then it is still there at
    serve time: the buffer is paid for twice.  Letting the policy raise the
    program count from 16 to 64 during profiling grew that retained buffer
    from 8 to 33 MiB and cost 976 KV tokens (523,008 -> 522,032).  A 16k
    request holds 1,088 pages, so the capacity ceiling fell from 30.04 to
    29.99 requests and steady-state 16k concurrency dropped from 30 to 29 --
    which ate most of the sampler speed-up on that case.

    Keeping the policy disarmed until after ``initialize_from_config`` makes
    the profiling run allocate exactly the buffer vLLM would have
    (num_compute_units = 16, i.e. 8 MiB), so the KV cache is sized as if the
    policy did not exist, and the 33 MiB buffer is then allocated from the
    headroom that is left *after* the KV cache -- 32 GiB total against a 0.85
    utilisation target leaves GiBs of it.  ``reset_buffer_cache()`` drops the
    profile-time buffer first so the two never coexist.
    """
    global _ILUVATAR_TOPP_ARMED
    if _ILUVATAR_TOPP_ARMED:
        return
    try:
        import importlib

        _ttt = importlib.import_module('vllm.v1.sample.ops.topk_topp_triton')
        _ttt.reset_buffer_cache()
    except Exception as e:  # noqa: BLE001
        logger.warning('enable_topp_scratch_policy (reset): %s', e)
    _ILUVATAR_TOPP_ARMED = True
    logger.info(
        'enable_topp_scratch_policy: top-p scratch policy armed after '
        'KV-cache allocation (%d MiB charged to post-KV headroom)',
        _ILUVATAR_SAMPLER_SCRATCH_BYTES >> 20,
    )


def patch_topp_launch_for_iluvatar() -> None:
    """Pick NUM_PROGRAMS for vLLM's Triton top-k/top-p kernel on BI-V150.

    vLLM launches ``_topk_topp_kernel`` with
    ``NUM_PROGRAMS = min(num_compute_units(dev), batch)`` and the kernel folds
    the leftover rows into each program (``for row_id in tl.range(pid,
    BATCH_SIZE, num_programs)``).  That policy assumes a device with many more
    compute units than there are sampled rows.  BI-V150 reports 16 compute
    units, so a 64-row sampling batch is serialised 4 rows deep per program
    and a 31-row batch 2 deep.

    The kernel is not occupancy-bound: a row is a handful of streaming passes
    over its own logits row, each program owns a private scratch row
    (``BUFFER + pid * VOCAB_SIZE``) and nothing is shared or reduced between
    rows, so one program per row is both legal and much faster.  Measured on
    BI-V150 with vocab 130560: B=64 3688 -> 1065 us (x3.46), B=31 1868 -> 1008
    us (x1.85), output bit-identical (``torch.equal``) and identical kept-token
    counts, because only the program count changes.

    Policy -- keyed on runtime facts only (sampling batch size, the device's
    compute-unit count, the vocabulary size):

        rows  = scratch budget / (vocab_size * 4 bytes)
        procs = min(batch, max(num_compute_units, rows))

    i.e. one program per row up to what the scratch budget allows, never
    fewer programs than vLLM would have used.  It is expressed by raising the
    "compute units" that ``apply_top_k_top_p_triton`` sizes its launch and its
    scratch buffer from, so vLLM's own kernel, buffer cache and launch code
    stay in charge; nothing about the kernel body changes.

    Safe to call repeatedly and from any process; it has no effect on other
    vendors.  A thread-local carries the per-call budget so the override is
    scoped to the sampler call and every other caller of
    ``num_compute_units`` keeps the real value.
    """
    try:
        import functools
        import importlib
        import threading

        _ttt = importlib.import_module('vllm.v1.sample.ops.topk_topp_triton')
        if getattr(_ttt, '_iluvatar_launch_patched', False):
            return

        _real_num_compute_units = _ttt.num_compute_units
        _state = threading.local()

        def _num_compute_units_iluvatar(*args, **kwargs):
            real = _real_num_compute_units(*args, **kwargs)
            want = getattr(_state, 'procs', None)
            if want is None:
                return real
            return max(real, want)

        _orig_apply = _ttt.apply_top_k_top_p_triton

        @functools.wraps(_orig_apply)
        def _apply_top_k_top_p_triton_iluvatar(logits, k, p, *args, **kwargs):
            procs = None
            try:
                if _ILUVATAR_TOPP_ARMED and logits.ndim == 2 \
                        and logits.shape[0] > 0:
                    batch, vocab = logits.shape
                    row_bytes = max(1, vocab * logits.element_size())
                    rows = max(1, _ILUVATAR_SAMPLER_SCRATCH_BYTES // row_bytes)
                    procs = min(int(batch), int(rows))
            except Exception:  # noqa: BLE001 - never break sampling
                procs = None
            _state.procs = procs
            try:
                return _orig_apply(logits, k, p, *args, **kwargs)
            finally:
                _state.procs = None

        _ttt.num_compute_units = _num_compute_units_iluvatar
        _ttt.apply_top_k_top_p_triton = _apply_top_k_top_p_triton_iluvatar
        try:
            _tts = importlib.import_module('vllm.v1.sample.ops.topk_topp_sampler')
            if getattr(_tts, 'apply_top_k_top_p_triton', None) is _orig_apply:
                _tts.apply_top_k_top_p_triton = _apply_top_k_top_p_triton_iluvatar
        except Exception as e:  # noqa: BLE001
            logger.warning('patch_topp_launch_for_iluvatar (rebind): %s', e)
        _ttt._iluvatar_launch_patched = True
        logger.info(
            'patch_topp_launch_for_iluvatar: top-p NUM_PROGRAMS = '
            'min(batch, max(compute_units, %d MiB / (vocab * 4)))',
            _ILUVATAR_SAMPLER_SCRATCH_BYTES >> 20,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning('patch_topp_launch_for_iluvatar: %s', e)


patch_topp_launch_for_iluvatar()


def install_il_sampler() -> None:
    """R9/T9: the eager sampler + logits chain outside the CUDA graph.

    Three pieces of that chain measured above their bandwidth floor on
    BI-V150 -- the temperature divide (FlagGems' rank-2 broadcast kernel at
    189 GB/s against a 537 GB/s floor), the top-p kernel's launch shape
    (num_warps never set, so Triton's default 4), and the four-pass sampling
    tail (softmax -> exponential_ -> div_ -> argmax, which the Gumbel-max
    identity collapses to one pass for the same q and the same token).
    See kernels/il_sampler.py for the full measurement table.

    Must run before any Sampler/TopKTopPSampler is constructed, because
    TopKTopPSampler.__init__ binds self.forward to a class attribute.
    """
    try:
        from vllm_fl.dispatch.backends.vendor.iluvatar.kernels import il_sampler
        il_sampler.install()
    except Exception as e:  # noqa: BLE001 - never break sampling
        logger.warning('install_il_sampler: %s', e)


install_il_sampler()


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
        # If 'cuda' is already registered as a backend name, no patch needed
        if 'cuda' in _registered or not _registered:
            logger.debug('patch_torch_inductor_for_iluvatar: triton has cuda backend or no backends, skipping')
            return
        # Probe the actual target string expected by supports_target().
        # Different flagtree triton versions use different conventions:
        #   - 3.2.x: backend name 'iluvatar', supports_target checks == 'cuda'
        #   - 3.6.x: backend name 'iluvatar', supports_target checks == 'corex'
        # We must discover the correct target string at runtime.
        _target = None
        try:
            from triton.backends.compiler import GPUTarget as _GPUTarget
            # Include 'cuda' in probes — some backends accept 'cuda' as target
            for _probe in ('cuda', 'corex', 'iluvatar') + tuple(_registered):
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

        # If the discovered target is already 'cuda', no remapping needed —
        # inductor naturally passes 'cuda' to GPUTarget.
        if _target == 'cuda':
            logger.info(
                "patch_torch_inductor_for_iluvatar: backend '%s' already "
                "supports target='cuda', no GPUTarget remap needed.",
                _registered,
            )
            return
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
        # default). The tl.make_tensor_descriptor stub and perf model patches
        # are already applied at module level (above), so triton JIT's
        # DependencyFinder can hash the kernel without AttributeError on
        # triton < 3.3.  The kernel itself uses USE_TD=False at runtime, so
        # the stub is never called.
        return AttentionBackendEnum.TRITON_ATTN.get_path()
