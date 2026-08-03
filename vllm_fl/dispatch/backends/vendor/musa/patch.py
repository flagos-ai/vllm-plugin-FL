# Copyright (c) 2026 BAAI. All rights reserved.

"""
MUSA-specific patches for vLLM compatibility.
"""

import logging

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_musa_patches():
    """Apply MUSA patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    patch_topk_topp_sampler()
    patch_triton_reshape_and_cache_flash()
    patch_cuda_get_device_properties()
    patch_accelerator_missing_attrs()
    patch_cuda_stream_for_musa()
    patch_inductor_triton_for_musa()
    patch_moe_topk_softmax_for_musa()
    patch_triton_mtgpu_alias_for_musa()
    patch_device_config_for_musa()


def patch_topk_topp_sampler():
    """Force PyTorch-native top-k/top-p on MUSA.

    vllm's default ``apply_top_k_top_p`` calls CUDA kernels via
    ``torch.ops._C_cache_ops`` which are not available on MUSA.  Replace
    with the pure-PyTorch fallback that vllm ships alongside.
    """
    try:
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
        import vllm.v1.sample.ops.topk_topp_sampler as _sampler_mod

        _sampler_mod.apply_top_k_top_p = apply_top_k_top_p_pytorch
        logger.info("Patched apply_top_k_top_p to use PyTorch-native path for MUSA")
    except Exception as e:
        logger.warning("Failed to patch apply_top_k_top_p for MUSA: %s", e)


def patch_triton_reshape_and_cache_flash():
    """No-op stub: reshape_and_cache_flash is handled by vllm_fl attention backends."""
    pass


def patch_cuda_get_device_properties():
    """Patch torch.cuda.get_device_capability to return a MUSA-safe value.

    Some vllm code paths query CUDA compute capability to decide which kernels
    to use.  On MUSA the call would fail or return wrong values.  We return
    (8, 0) which is sufficient to enable bf16/fp16 paths without triggering
    Ampere/Hopper-specific code that MUSA does not support.
    """
    try:
        import torch
        import torch_musa

        if getattr(torch.cuda, "_musa_device_props_patched", False):
            return

        def _get_device_capability_musa(device=None):
            return (8, 0)

        torch.cuda.get_device_capability = _get_device_capability_musa
        logger.info("Patched torch.cuda.get_device_capability for MUSA")

        _orig_get_props = torch.cuda.get_device_properties

        def _cuda_get_device_properties_musa(device, names, init_cuda=False):
            props = torch_musa.get_device_properties(device)
            return props

        import vllm.utils as _vllm_utils
        if hasattr(_vllm_utils, "cuda_get_device_properties"):
            _vllm_utils.cuda_get_device_properties = _cuda_get_device_properties_musa
            logger.info("Patched cuda_get_device_properties to use torch_musa for MUSA")

        torch.cuda._musa_device_props_patched = True
    except Exception as e:
        logger.warning("Failed to patch cuda_get_device_properties for MUSA: %s", e)


def patch_accelerator_missing_attrs():
    """Patch missing/broken torch.accelerator attributes for MUSA compatibility.

    Some vLLM modules call APIs that were added to torch.accelerator in newer
    PyTorch versions but are absent or broken on the MUSA build:

    - ``torch.accelerator.empty_cache()`` — called by gdn_linear_attn.py after
      prefill kernel warmup. The built-in impl calls
      ``_accelerator_isAllocatorInitialized()`` which asserts on MUSA.
      Delegated to ``torch_musa.empty_cache()``.

    - ``torch.accelerator.max_memory_allocated()`` — called by
      ``vllm.model_executor.model_loader.base_loader`` after model load to
      measure peak memory. Same ``_accelerator_isAllocatorInitialized()``
      assert. Delegated to ``torch_musa.max_memory_allocated()``.

    - ``torch.accelerator.device_index(index)`` — used as a context manager in
      fla/ops/utils.py to pin operations to a specific device. The MUSA
      equivalent is ``torch_musa.device(index)``.
    """
    try:
        import torch
        import torch_musa

        if getattr(torch.accelerator, "_musa_attrs_patched", False):
            return

        # Unconditionally override: the built-in impls call
        # _accelerator_isAllocatorInitialized() which asserts on MUSA.
        torch.accelerator.empty_cache = torch_musa.empty_cache
        logger.info("Patched torch.accelerator.empty_cache for MUSA")

        # max_memory_allocated is called by base_loader after model load.
        torch.accelerator.max_memory_allocated = torch_musa.max_memory_allocated
        logger.info("Patched torch.accelerator.max_memory_allocated for MUSA")

        if not hasattr(torch.accelerator, 'device_index'):
            torch.accelerator.device_index = torch_musa.device
            logger.info("Patched torch.accelerator.device_index for MUSA")

        torch.accelerator._musa_attrs_patched = True
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator attrs for MUSA: %s", e)


def patch_cuda_stream_for_musa():
    """Patch CUDA stream APIs to use MUSA equivalents.

    vllm uses several ``torch.cuda`` stream primitives internally:

    - ``vllm.utils.torch_utils.aux_stream()`` creates a background stream for
      MoE shared-expert overlap. Patched to return a ``torch_musa.Stream()``.

    - ``torch.cuda.stream(s)`` context manager used in shared_experts.py.
      Patched to delegate to ``torch.musa.stream(s)`` on MUSA.

    - ``torch.cuda.set_stream(s)`` called by vllm's current_stream bookkeeping.
      Patched to delegate to ``torch.musa.set_stream(s)`` on MUSA.

    - ``torch.cuda.current_stream(device)`` queried by the scheduler.
      Patched to delegate to ``torch.musa.current_stream(device)`` on MUSA.
    """
    try:
        import torch
        import torch.cuda as torch_cuda
        import torch_musa

        if getattr(torch_cuda, "_musa_stream_patched", False):
            return

        # aux_stream helper used by MoE shared-expert overlap.
        # MUSA does not support torch.cuda.Stream() — return None so that
        # SharedExperts falls back to the synchronous (non-overlapped) path,
        # which is guarded by ``if self._stream is not None``.
        try:
            import vllm.utils.torch_utils as _tu

            def _aux_stream_musa():
                return None

            _tu.aux_stream = _aux_stream_musa

            # shared_experts.py uses `from vllm.utils.torch_utils import aux_stream`
            # which binds the function object at import time — we must also
            # patch the local reference in every module that has done so.
            # Force-import each module so that sys.modules has them, then
            # overwrite their local binding.
            import importlib as _il
            import sys as _sys
            _modules_using_aux_stream = [
                "vllm.model_executor.layers.fused_moe.runner.shared_experts",
                "vllm.worker.model_runner_base",
                "vllm.model_executor.models.utils",
                "vllm.model_executor.layers.fused_moe.runner.moe_runner",
            ]
            for _mod_name in _modules_using_aux_stream:
                try:
                    _mod = _sys.modules.get(_mod_name) or _il.import_module(_mod_name)
                    if hasattr(_mod, "aux_stream"):
                        _mod.aux_stream = _aux_stream_musa
                except Exception as _me:
                    logger.debug("aux_stream patch: skipped %s: %s", _mod_name, _me)
            logger.info("Patched aux_stream → None for MUSA (SharedExperts sync path)")
        except Exception as e:
            logger.warning("Failed to patch aux_stream for MUSA: %s", e)

        # --- torch.cuda.stream() context manager -> torch.musa.stream() ---
        _orig_cuda_stream_ctx = torch_cuda.stream

        def _cuda_stream_ctx_musa(stream):
            if stream is None:
                import contextlib
                return contextlib.nullcontext()
            if isinstance(stream, torch_musa.Stream):
                return torch.musa.stream(stream)
            return _orig_cuda_stream_ctx(stream)

        torch_cuda.stream = _cuda_stream_ctx_musa

        # --- torch.cuda.set_stream() -> torch.musa.set_stream() ---
        _orig_set_stream = torch_cuda.set_stream

        def _set_stream_musa(stream):
            if isinstance(stream, torch_musa.Stream):
                torch.musa.set_stream(stream)
                try:
                    from vllm.utils.torch_utils import _current_stream_tls
                    _current_stream_tls.value = stream
                except Exception:
                    pass
            else:
                _orig_set_stream(stream)

        torch_cuda.set_stream = _set_stream_musa

        # --- torch.cuda.current_stream() -> torch.musa.current_stream() ---
        _orig_current_stream = torch_cuda.current_stream

        def _current_stream_musa(device=None):
            try:
                return torch.musa.current_stream(device)
            except Exception:
                return _orig_current_stream(device)

        torch_cuda.current_stream = _current_stream_musa

        torch_cuda._musa_stream_patched = True
        logger.info("Patched torch.cuda stream APIs for MUSA")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda stream APIs for MUSA: %s", e)


def patch_inductor_triton_for_musa():
    """Patch torch._inductor triton_heuristics.make_launcher for MUSA.

    flagtree 0.6.0+mthreads3.6 installs itself as the ``triton`` package but
    its mthreads backend builds ``CompiledKernel.metadata`` as a
    ``KernelMetadata`` namedtuple that does **not** include a ``cluster_dims``
    field (MUSA hardware does not support cluster launches).

    ``TritonCompileResult.make_launcher`` unconditionally accesses
    ``binary.metadata.cluster_dims`` when it takes the mthreads kernel branch,
    raising ``AttributeError`` during KV-cache profiling warmup.

    Fix: when a mthreads binary is detected (has ``metadata``, no top-level
    ``num_ctas``, no ``cluster_dims`` in metadata), rebuild the
    ``KernelMetadata`` namedtuple with a synthetic ``cluster_dims=(1,1,1)``
    before delegating to the original implementation.

    TODO: remove once flagtree mthreads adds ``cluster_dims`` to its
    ``KernelMetadata``.
    """
    try:
        import torch._inductor.runtime.triton_heuristics as _th

        if getattr(_th, "_musa_cluster_dims_patched", False):
            return

        _OrigCompileResult = _th.TritonCompileResult
        _orig_make_launcher = _OrigCompileResult.make_launcher

        def _make_launcher_musa(self):
            # TritonCompileResult uses .kernel (not .binary) in torch 2.9+
            kernel = self.kernel
            # Detect mthreads kernel: has metadata namedtuple but no cluster_dims
            if (
                kernel is not None
                and hasattr(kernel, "metadata")
                and hasattr(kernel.metadata, "_fields")
                and "cluster_dims" not in kernel.metadata._fields
            ):
                try:
                    from collections import namedtuple
                    old_meta = kernel.metadata
                    fields = old_meta._fields + ("cluster_dims",)
                    NewMeta = namedtuple(type(old_meta).__name__, fields)
                    kernel.metadata = NewMeta(*tuple(old_meta), (1, 1, 1))
                except Exception:
                    pass
            return _orig_make_launcher(self)

        _OrigCompileResult.make_launcher = _make_launcher_musa
        _th._musa_cluster_dims_patched = True
        logger.info(
            "Patched torch._inductor triton_heuristics.make_launcher "
            "for MUSA (cluster_dims fallback)")
    except Exception as e:
        logger.warning(
            "Failed to patch torch._inductor triton_heuristics for MUSA: %s",
            e)


def patch_moe_topk_softmax_for_musa():
    """Patch MoE top-k softmax for MUSA via FlagGems.

    ``torch.ops._moe_C.topk_softmax`` is a CUDA-only extension not available
    on MUSA.  vllm 0.24.0 calls it through two entry points:

      - ``vllm.model_executor.layers.fused_moe.router.fused_topk_router``
        ``vllm_topk_softmax()``
      - ``vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router``
        ``vllm_topk_softmax()``

    Rather than patching ``vllm._custom_ops`` (which suffers from circular
    import timing issues at apply_musa_patches() call time), we directly
    replace the ``vllm_topk_softmax`` function objects in both router modules
    via ``importlib.import_module``.  These modules are safe to import at
    this point since they only depend on torch and vllm internals that are
    already loaded by the time apply_musa_patches() is invoked in the worker.

    TODO: remove once MUSA ships a compiled _moe_C extension.
    """
    try:
        from vllm_fl.dispatch.backends.flaggems.impl.fused_moe import (
            topk_softmax_flaggems,
        )
    except Exception as exc:
        logger.warning(
            "patch_moe_topk_softmax_for_musa: cannot import "
            "topk_softmax_flaggems — MoE models will fail on MUSA: %s", exc)
        return

    def _topk_softmax_musa(
        topk_weights,
        topk_ids,
        token_expert_indices,
        gating_output,
        renormalize=False,
        e_score_correction_bias=None,
    ):
        # topk_softmax_flaggems modifies topk_weights/topk_ids in-place.
        # Must return (topk_weights, topk_ids) to match vllm_topk_softmax
        # signature which callers unpack as: topk_weights, topk_ids = topk_func(...)
        topk_softmax_flaggems(
            topk_weights,
            topk_ids,
            token_expert_indices,
            gating_output,
            renormalize,
        )
        return topk_weights, topk_ids

    _router_modules = [
        "vllm.model_executor.layers.fused_moe.router.fused_topk_router",
        "vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router",
    ]
    patched = []
    for mod_name in _router_modules:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "vllm_topk_softmax"):
                mod.vllm_topk_softmax = _topk_softmax_musa
                patched.append(mod_name.split(".")[-1])
        except Exception as exc:
            logger.warning(
                "patch_moe_topk_softmax_for_musa: failed to patch %s: %s",
                mod_name, exc)

    if patched:
        logger.info(
            "Patched vllm_topk_softmax for MUSA (FlagGems) in: %s",
            ", ".join(patched))


def patch_triton_mtgpu_alias_for_musa():
    """Alias triton.backends.mtgpu -> triton.backends.mthreads.

    flagtree >= 3.6 renamed the mthreads backend from ``mtgpu`` to
    ``mthreads``.  torch_musa's inductor code still imports from the old
    ``triton.backends.mtgpu`` namespace (e.g.
    ``torch_musa/_inductor/utils.py`` does
    ``from triton.backends.mtgpu.musa_testing import do_bench``).

    We insert ``mtgpu`` as an alias in ``sys.modules`` and as an attribute
    on ``triton.backends`` so both attribute access and import statements
    work transparently.  We also create a ``musa_testing`` stub module that
    exposes ``do_bench`` from ``triton.testing`` so that torch_musa inductor
    code that benchmarks kernels continues to work when flagtree no longer
    ships ``triton.backends.mtgpu.musa_testing``.

    TODO: remove once torch_musa updates its inductor code to use
    ``triton.backends.mthreads`` and the current triton.testing API.
    """
    try:
        import sys
        import types
        import triton.backends as _tb

        # ------------------------------------------------------------------ #
        # 1. Alias triton.backends.mtgpu -> triton.backends.mthreads          #
        # ------------------------------------------------------------------ #
        if not hasattr(_tb, "mtgpu"):
            if hasattr(_tb, "mthreads"):
                _tb.mtgpu = _tb.mthreads
                parent = sys.modules.get("triton.backends.mthreads")
                if parent is not None:
                    sys.modules.setdefault("triton.backends.mtgpu", parent)
                    for k, v in list(sys.modules.items()):
                        if k.startswith("triton.backends.mthreads."):
                            alias = k.replace(
                                "triton.backends.mthreads.",
                                "triton.backends.mtgpu.",
                                1,
                            )
                            sys.modules.setdefault(alias, v)
                logger.info(
                    "Aliased triton.backends.mtgpu -> triton.backends.mthreads "
                    "(torch_musa inductor compatibility)")
            else:
                logger.debug(
                    "patch_triton_mtgpu_alias_for_musa: "
                    "triton.backends.mthreads not found, skipping alias")

        # ------------------------------------------------------------------ #
        # 2. Create triton.backends.mtgpu.musa_testing stub if missing        #
        # ------------------------------------------------------------------ #
        stub_key = "triton.backends.mtgpu.musa_testing"
        if stub_key not in sys.modules:
            # Try to get do_bench from triton.testing (always present)
            try:
                from triton.testing import do_bench as _do_bench
            except ImportError:
                _do_bench = None

            stub = types.ModuleType(stub_key)
            if _do_bench is not None:
                stub.do_bench = _do_bench
            sys.modules[stub_key] = stub

            # Also expose as attribute on the mtgpu package if it exists
            mtgpu_mod = sys.modules.get("triton.backends.mtgpu")
            if mtgpu_mod is not None:
                mtgpu_mod.musa_testing = stub

            logger.info(
                "Created triton.backends.mtgpu.musa_testing stub "
                "(do_bench -> triton.testing.do_bench)")

    except Exception as exc:
        logger.warning(
            "patch_triton_mtgpu_alias_for_musa failed: %s", exc)


def patch_device_config_for_musa():
    """Patch vllm DeviceConfig so that device_type='musa' is accepted.

    vllm 0.24.0's ``DeviceConfig.__post_init__`` calls
    ``torch.device(self.device_type)`` which raises
    ``RuntimeError: Device string must not be empty`` when device_type is
    ``'musa'`` because plain ``torch.device('musa')`` is invalid without an
    index — ``torch.device('musa:0')`` works after torch_musa is imported.

    We monkey-patch ``__post_init__`` to intercept the ``'musa'`` case and
    substitute ``torch.device('musa:0')`` before the original logic runs.

    TODO: remove once vllm DeviceConfig handles non-CUDA device strings
    natively, or torch_musa registers 'musa' as a valid bare device string.
    """
    try:
        import torch
        import torch_musa  # noqa: F401 — registers musa device type with torch
        import vllm.config.device as _dc_mod

        if getattr(_dc_mod.DeviceConfig, "_musa_post_init_patched", False):
            return

        _orig_post_init = _dc_mod.DeviceConfig.__post_init__

        def _patched_post_init(self):
            # Intercept before torch.device("musa") is called (invalid).
            if getattr(self, "device", None) == "musa":
                self.device_type = "musa"
                self.device = torch.device("musa:0")
                return
            _orig_post_init(self)

        _dc_mod.DeviceConfig.__post_init__ = _patched_post_init
        _dc_mod.DeviceConfig._musa_post_init_patched = True
        logger.info(
            "Patched DeviceConfig.__post_init__ for MUSA "
            "(torch.device('musa:0') workaround)")
    except Exception as exc:
        logger.warning(
            "patch_device_config_for_musa failed: %s", exc)
