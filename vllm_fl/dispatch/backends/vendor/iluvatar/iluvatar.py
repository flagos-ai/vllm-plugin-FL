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


# _iluvatar_paged_attn_opt_hook
import logging as _logging
import os as _os
if "logger" not in globals():
    logger = _logging.getLogger(__name__)
_DECODE_WORKSPACES = {}
_ATTN_STATS = {"decode": 0, "prefill": 0, "fallback": 0}
_ATTN_FB = {}
_ATTN_CALLS = 0
_ATTN_DUMPED = False
_OPT_MOD = None
_ATTN_STATS_ON = _os.environ.get("FL_ATTN_STATS", "0") not in ("0", "false", "False")


def _attn_stat(branch, fb_reason=None):
    if not _ATTN_STATS_ON:
        return
    global _ATTN_CALLS
    _ATTN_STATS[branch] += 1
    if fb_reason is not None:
        _ATTN_FB[fb_reason] = _ATTN_FB.get(fb_reason, 0) + 1
    _ATTN_CALLS += 1
    if _ATTN_CALLS % 200 == 0:
        logger.info(
            "[FL_ATTN_STATS] calls=%d decode=%d prefill=%d fallback=%d fb_reasons=%s sampler=%s",
            _ATTN_CALLS, _ATTN_STATS["decode"], _ATTN_STATS["prefill"],
            _ATTN_STATS["fallback"], dict(_ATTN_FB), getattr(_OPT_MOD, "_SAMPLER_STATS", {}),
        )


def patch_triton_unified_attention_for_iluvatar() -> None:
    """Route MiniCPM5-2B GQA attention to page-aligned BLOCK_Q=8 prefill and split-KV."""
    try:
        from . import paged_attn_opt as _opt
    except Exception as exc:
        logger.warning("patch_triton_unified_attention_for_iluvatar skipped: %s", exc)
        return

    if __package__ and __package__.endswith(".metax"):
        try:
            import sys as _sys
            _pkg = _sys.modules.get(__package__)
            if _pkg is not None and "MacaBackend" in globals():
                setattr(_pkg, "MacaBackend", globals()["MacaBackend"])
            import vllm._custom_ops  # noqa: F401
            import vllm_fl.dispatch.backends.vendor.metax.patches  # noqa: F401
        except Exception:
            pass

    globals()["_OPT_MOD"] = _opt
    _opt.patch_iluvatar_sampler()
    try:
        import vllm.v1.attention.ops.triton_unified_attention as _tua
    except Exception as exc:
        logger.warning("patch_triton_unified_attention_for_iluvatar skipped: %s", exc)
        return
    if getattr(_tua, "_iluvatar_paged_attn_patched", False):
        return

    _orig_unified = _tua.unified_attention

    def _unified_attention_opt(
        q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
        softmax_scale, causal, window_size, block_table, softcap, *args, **kwargs
    ):
        # Forward-compatible shim: vLLM 0.24.0 passes extra keyword args
        # (use_alibi_sqrt, kv_quant_mode, k/v_scale_cache, chunk_lookback, use_td).
        # Anything we do not explicitly handle is captured in *args/**kwargs and
        # forwarded verbatim to the original on the fallback path.
        def _fallback(reason="?"):
            _attn_stat("fallback", reason)
            return _orig_unified(
                q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size, block_table, softcap,
                *args, **kwargs,
            )

        if _ATTN_STATS_ON and not _ATTN_DUMPED:
            globals()["_ATTN_DUMPED"] = True
            try:
                logger.info(
                    "[FL_ATTN_KWARGS] mq=%s mk=%s qshape=%s seqused=%s kshape=%s kstride=%s "
                    "mm_prefix=%s kv_quant=%s q_descale=%s k_descale=%s v_descale=%s "
                    "k_scale_cache=%s chunk_lookback=%s causal=%s window=%s",
                    max_seqlen_q, max_seqlen_k, tuple(q.shape), tuple(seqused_k.shape),
                    tuple(k.shape), tuple(k.stride()), kwargs.get("mm_prefix_range") is not None,
                    getattr(kwargs.get("kv_quant_mode"), "name", kwargs.get("kv_quant_mode")),
                    kwargs.get("q_descale") is not None, kwargs.get("k_descale") is not None,
                    kwargs.get("v_descale") is not None, kwargs.get("k_scale_cache") is not None,
                    kwargs.get("chunk_lookback"), causal, window_size,
                )
            except Exception:
                pass

        # Only handle the plain causal BF16/FP16/FP32 GQA case. Fall back for
        # PrefixLM, KV/logits quantization, and chunked-attention lookback, whose
        # masking/scaling our kernels do not implement.
        # NOTE: vLLM 0.24.0 ALWAYS passes k_descale/v_descale as (1.0) scale tensors
        # even for bf16/fp16, so descale non-None is NOT a quantization signal.
        # Quantization is gated solely by kv_quant_mode (== NONE for unquantized).
        if kwargs.get("mm_prefix_range") is not None:
            return _fallback("mm_prefix_range")
        _km = kwargs.get("kv_quant_mode")
        if _km is not None and getattr(_km, "name", None) not in (None, "NONE"):
            return _fallback("kv_quant_mode=%s" % getattr(_km, "name", None))
        if kwargs.get("k_scale_cache") is not None or kwargs.get("v_scale_cache") is not None:
            return _fallback("kv_scale_cache")
        if kwargs.get("chunk_lookback", -1) not in (None, -1):
            return _fallback("chunk_lookback=%s" % kwargs.get("chunk_lookback"))
        if not _opt.can_use_specialized_attention(
            q, k, v, causal=causal, window_size=window_size,
            alibi_slopes=kwargs.get("alibi_slopes"), softcap=softcap,
            qq_bias=kwargs.get("qq_bias"), sinks=kwargs.get("sinks"),
            output_scale=kwargs.get("output_scale"),
        ):
            return _fallback("specialized_guard")
        dev_key = (q.device.type, q.device.index)
        ws = _DECODE_WORKSPACES.get(dev_key)
        if ws is None:
            ws = _opt.SplitKVWorkspace(
                max_batch=64, q_heads=q.shape[1], max_splits=8, dim=q.shape[2]
            )
            _DECODE_WORKSPACES[dev_key] = ws
        max_k = int(max_seqlen_k) if max_seqlen_k and int(max_seqlen_k) > 1 else 4096
        if int(max_seqlen_q) <= 1 and q.shape[0] == seqused_k.shape[0]:
            _attn_stat("decode")
            splits = _opt.select_num_splits(q.shape[0], max_k, kv_heads=k.shape[2])
            _opt.launch_split_kv_decode(
                q, k, v, block_table=block_table, seq_lens=seqused_k,
                workspace=ws, splits=splits, scale=softmax_scale, out=out,
            )
            return
        _attn_stat("prefill")
        _opt.launch_paged_prefill(
            q, k, v, block_table=block_table, seq_lens=seqused_k,
            cu_seqlens_q=cu_seqlens_q, scale=softmax_scale, out=out,
            block_q=16, workspace=ws, max_seqlen_k=max_k,
        )

    _tua.unified_attention = _unified_attention_opt
    _tua._iluvatar_paged_attn_patched = True
    try:
        import vllm.v1.attention.backends.triton_attn as _ta
        if hasattr(_ta, "unified_attention"):
            _ta.unified_attention = _unified_attention_opt
    except Exception:
        pass
    logger.info(
        "Patched unified_attention with Iluvatar BI-V150 page-aligned BLOCK_Q=16 + split-KV (stats=%s).",
        _ATTN_STATS_ON,
    )


patch_triton_unified_attention_for_iluvatar()
