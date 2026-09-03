# Copyright (c) 2026 BAAI. All rights reserved.

"""Sunrise deferred patches for vLLM compatibility.

``apply_sunrise_patches()`` is called from ``vllm_fl.ops.custom_ops`` before
model construction. Import-time patches live in ``patches/__init__.py``.
"""

import logging
import os
import sys

import torch

logger = logging.getLogger(__name__)
_patches_applied = False


def apply_sunrise_patches():
    """Apply Sunrise/PTPU patches that must run before model construction."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    # Must run early: vLLM multi_stream_utils uses torch.cuda.stream/current_stream.
    patch_cuda_stream_for_ptpu()
    patch_flagcx_comm_lifecycle()
    patch_flagcx_stream_adapter()
    patch_flagcx_collective_hot_path()
    patch_vllm_group_coordinator_dispatch()
    patch_op_manager_fast_path()
    patch_oot_layer_fast_path()
    patch_ptpu_topk_topp_sampler()
    patch_ptpu_penalties_bincount()
    patch_distributed_runtime()
    patch_ptpu_cudagraph()
    patch_op_cls()
    patch_accelerator_empty_cache()
    patch_memory_profiling_for_plain_allocator()
    patch_fused_topk_bias_router()
    patch_hy_v3_shared_mlp_weights()
    patch_mla_gather_cache_ops()
    patch_ptpu_trunc_normal_init()
    patch_minicpmo_resampler_device()
    patch_moe_force_config()
    patch_native_moe_ops()
    patch_native_int8_routing()


def patch_native_moe_ops() -> None:
    """Give vLLM's fused-MoE path implementations that exist on PTPU.

    Deferred rather than import-time: the shims resolve through the FL dispatch
    registry, which is only populated once the backends have registered. Must
    precede ``patch_native_int8_routing``, which routes W8A8 MoE onto them.
    """
    from .patches import patch_moe_native_ops

    patch_moe_native_ops.apply_patch()


def patch_native_int8_routing() -> None:
    """Re-assert the INT8 routing that ``register_oot_ops`` can overwrite.

    ``register_oot_ops`` installs ``install_fl_w8a8_moe_selector`` and clones the
    CUDA linear kernels into the OOT slot before it calls us, so sunrise has to
    re-run the order-sensitive INT8 patches afterwards to be the last writer.
    """
    from .patches import patch_int8_native

    patch_int8_native.install_late_patches()


def patch_moe_force_config() -> None:
    """Pin the fused-MoE tile config, for bisecting tile-dependent bugs.

    vLLM derives BLOCK_SIZE_* from the token count alone and exposes no flag to
    fix them, but ``try_get_optimal_moe_config`` consults a module global before
    any heuristic, which is what this sets. BLOCK_SIZE_M also reaches
    moe_align_block_size, so the padding follows along.

    Off unless VLLM_FL_MOE_FORCE_CONFIG holds a JSON object, e.g.
    '{"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64,
      "GROUP_SIZE_M": 1, "num_warps": 8, "num_stages": 3}'
    """
    raw = os.environ.get("VLLM_FL_MOE_FORCE_CONFIG", "").strip()
    if not raw:
        return

    import json

    import vllm.model_executor.layers.fused_moe as vllm_fused_moe

    config = json.loads(raw)
    required = {
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "BLOCK_SIZE_K",
        "GROUP_SIZE_M",
        "num_warps",
        "num_stages",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(
            f"VLLM_FL_MOE_FORCE_CONFIG is missing {sorted(missing)}; got {config}"
        )

    vllm_fused_moe._config = config
    logger.warning(
        "VLLM_FL_MOE_FORCE_CONFIG is set: every fused-MoE shape will use %s. "
        "This is a debugging lever and costs throughput.",
        config,
    )


def patch_minicpmo_resampler_device() -> None:
    """Move MiniCPM-O resampler pos caches to PTPU after weight load.

    ``MiniCPMOBaseModel.load_weights`` overrides ``MiniCPMV4_5.load_weights``
    without calling ``_ensure_resampler_device()``. Resampler 2D/temporal
    position buffers are created on CPU; vision forward then fails with
    ``ptpu:0 vs cpu`` when adding ``pos_embed`` to activations.
    """
    if getattr(patch_minicpmo_resampler_device, "_done", False):
        return

    try:
        from vllm.model_executor.models.minicpmo import MiniCPMOBaseModel
    except ImportError:
        return

    if getattr(MiniCPMOBaseModel, "_sunrise_resampler_device_patched", False):
        return

    _orig_load_weights = MiniCPMOBaseModel.load_weights

    def _load_weights(self, weights):
        loaded = _orig_load_weights(self, weights)
        ensure = getattr(self, "_ensure_resampler_device", None)
        if ensure is not None:
            ensure()
        return loaded

    MiniCPMOBaseModel.load_weights = _load_weights
    MiniCPMOBaseModel._sunrise_resampler_device_patched = True
    patch_minicpmo_resampler_device._done = True  # type: ignore[attr-defined]
    logger.info(
        "Patched MiniCPMOBaseModel.load_weights to call _ensure_resampler_device"
    )


def patch_cuda_stream_for_ptpu():
    """Patch torch.cuda stream APIs for Sunrise/PTPU (non-CUDA torch).

    Upstream ``vllm.utils.multi_stream_utils`` (and shared-expert overlap) call
    ``torch.cuda.stream`` / ``current_stream`` / ``set_stream``. On PTPU those
    hit ``AssertionError: Torch not compiled with CUDA enabled``. Mirror the
    MUSA patch: delegate to ``current_platform.torch_device_fn`` (torch.ptpu).
    """
    try:
        import contextlib

        import torch.cuda as torch_cuda
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return
        if getattr(torch_cuda, "_ptpu_stream_patched", False):
            return

        device_fn = current_platform.torch_device_fn

        # --- aux_stream: return a PTPU Stream instead of torch.cuda.Stream() ---
        try:
            import vllm.utils.torch_utils as tu_mod

            def _aux_stream_ptpu():
                if tu_mod._aux_stream is None:
                    tu_mod._aux_stream = device_fn.Stream()
                return tu_mod._aux_stream

            tu_mod.aux_stream = _aux_stream_ptpu
            try:
                import vllm.model_executor.layers.fused_moe.runner.shared_experts as se_mod

                se_mod.aux_stream = _aux_stream_ptpu
            except Exception:
                pass
            logger.info("Patched vllm.utils.torch_utils.aux_stream for PTPU")
        except Exception as e:
            logger.warning("Failed to patch aux_stream for PTPU: %s", e)

        def _is_ptpu_stream(stream) -> bool:
            if stream is None:
                return False
            try:
                if isinstance(stream, device_fn.Stream):
                    return True
            except Exception:
                pass
            return type(stream).__module__.startswith("torch.ptpu")

        # --- torch.cuda.stream() -> torch.ptpu.stream() ---
        def _cuda_stream_ctx_ptpu(stream):
            if stream is None:
                return contextlib.nullcontext()
            if _is_ptpu_stream(stream):
                return device_fn.stream(stream)
            # Prefer PTPU stream ctx for unknown device streams; never call
            # the real CUDA path (raises on non-CUDA builds).
            try:
                return device_fn.stream(stream)
            except Exception:
                return contextlib.nullcontext()

        torch_cuda.stream = _cuda_stream_ctx_ptpu

        # --- torch.cuda.set_stream() -> torch.ptpu.set_stream() ---
        def _set_stream_ptpu(stream):
            if _is_ptpu_stream(stream) or stream is not None:
                device_fn.set_stream(stream)
                try:
                    from vllm.utils.torch_utils import _current_stream_tls

                    _current_stream_tls.value = stream
                except Exception:
                    pass

        torch_cuda.set_stream = _set_stream_ptpu

        # --- torch.cuda.current_stream() -> torch.ptpu.current_stream() ---
        def _current_stream_ptpu(device=None):
            return device_fn.current_stream(device)

        torch_cuda.current_stream = _current_stream_ptpu

        # --- torch.cuda.Event -> torch.ptpu.Event ---
        # DeepSeek-V4 MLA wrappers and multi_stream_utils construct
        # ``torch.cuda.Event()``; without this alias PTPU raises.
        try:
            torch_cuda.Event = device_fn.Event
        except Exception as e:
            logger.warning("Failed to patch torch.cuda.Event for PTPU: %s", e)

        torch_cuda._ptpu_stream_patched = True
        logger.info("Patched torch.cuda stream/Event APIs for PTPU")
    except Exception as e:
        logger.warning("Failed to patch torch.cuda stream APIs for PTPU: %s", e)


def patch_mla_gather_cache_ops():
    """Register torch fallbacks for MLA prefix/chunked-context KV gather.

    PTPU builds lack ``torch.ops._C_cache_ops.gather_and_maybe_dequant_cache``
    (and often ``cp_gather_cache``). Without these, Moonlight/TeleChat MLA
    crashes on prefix-cache hits after the first request.

    This fills a missing platform op; it does not override FlagGems MLA kernels.
    """
    if getattr(patch_mla_gather_cache_ops, "_done", False):
        return

    from vllm import _custom_ops as ops

    from .impl.mla import torch_gather_and_maybe_dequant_cache

    def _gather(
        src_cache,
        dst,
        block_table,
        cu_seq_lens,
        token_to_seq,
        num_tokens,
        kv_cache_dtype,
        scale,
        seq_starts=None,
    ):
        return torch_gather_and_maybe_dequant_cache(
            src_cache,
            dst,
            block_table,
            cu_seq_lens,
            token_to_seq,
            num_tokens,
            kv_cache_dtype,
            scale,
            seq_starts,
        )

    ops.gather_and_maybe_dequant_cache = _gather  # type: ignore[attr-defined]

    # Best-effort: also define on torch.ops namespace if the library exists.
    try:
        import torch

        if not hasattr(torch.ops, "_C_cache_ops"):
            torch.library.define(
                "_C_cache_ops::gather_and_maybe_dequant_cache",
                "(Tensor src_cache, Tensor(a!) dst, Tensor block_table, "
                "Tensor cu_seq_lens, Tensor token_to_seq, int num_tokens, "
                "str kv_cache_dtype, Tensor scale, Tensor? seq_starts=None) -> ()",
            )
        lib = torch.library.Library("_C_cache_ops", "IMPL")
        lib.impl("gather_and_maybe_dequant_cache", _gather, "PrivateUse1")
        lib.impl("gather_and_maybe_dequant_cache", _gather, "CPU")
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not register torch.ops gather fallback: %s", exc)

    patch_mla_gather_cache_ops._done = True  # type: ignore[attr-defined]
    logger.info(
        "Patched gather_and_maybe_dequant_cache with Sunrise torch fallback"
    )


def patch_ptpu_cudagraph():
    from .patches.patch_cudagraph import patch_ptpu_cudagraph as _patch

    _patch()


def patch_ptpu_penalties_bincount():
    from .patches.patch_penalties import patch_ptpu_penalties_bincount as _patch

    _patch()


def patch_ptpu_trunc_normal_init() -> None:
    """Run ``torch.nn.init.trunc_normal_`` on CPU when the tensor is on PTPU.

    MiniCPM-o ``Resampler4_5`` calls ``trunc_normal_(self.query, ...)`` during
    construction. That path uses ``aten::erfinv.out``, which PTPU does not
    implement (``NotImplementedError: ... 'ptpu' backend``). Weight init only
    happens at load time, so a CPU round-trip is cheap and correct.

    Also rebinds ``trunc_normal_`` on already-imported vLLM MiniCPM modules
    (they use ``from torch.nn.init import trunc_normal_``) and provides a
    ``Tensor.erfinv_`` PTPU fallback for any other callers.
    """
    if getattr(patch_ptpu_trunc_normal_init, "_done", False):
        return

    import sys

    import torch.nn.init as init

    _orig_trunc = init.trunc_normal_
    _orig_erfinv_ = torch.Tensor.erfinv_

    def _is_ptpu_tensor(tensor: torch.Tensor) -> bool:
        try:
            return bool(getattr(tensor, "is_ptpu", False)) or (
                tensor.device.type == "ptpu"
            )
        except Exception:
            return False

    def _erfinv_(self: torch.Tensor):
        if not _is_ptpu_tensor(self):
            return _orig_erfinv_(self)
        cpu = self.detach().to("cpu")
        _orig_erfinv_(cpu)
        with torch.no_grad():
            self.copy_(cpu.to(device=self.device, dtype=self.dtype))
        return self

    def _trunc_normal_(
        tensor: torch.Tensor,
        mean: float = 0.0,
        std: float = 1.0,
        a: float = -2.0,
        b: float = 2.0,
        generator=None,
    ):
        if not _is_ptpu_tensor(tensor):
            return _orig_trunc(
                tensor, mean=mean, std=std, a=a, b=b, generator=generator
            )

        cpu = tensor.detach().to("cpu")
        _orig_trunc(cpu, mean=mean, std=std, a=a, b=b, generator=generator)
        with torch.no_grad():
            tensor.copy_(cpu.to(device=tensor.device, dtype=tensor.dtype))
        return tensor

    torch.Tensor.erfinv_ = _erfinv_  # type: ignore[method-assign, assignment]
    init.trunc_normal_ = _trunc_normal_  # type: ignore[assignment]

    # Modules that already bound the original symbol at import time.
    for mod_name in (
        "vllm.model_executor.models.minicpmv",
        "vllm.model_executor.models.minicpmo",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None and hasattr(mod, "trunc_normal_"):
            mod.trunc_normal_ = _trunc_normal_

    patch_ptpu_trunc_normal_init._done = True  # type: ignore[attr-defined]
    logger.info(
        "Patched trunc_normal_/Tensor.erfinv_ for PTPU "
        "(CPU fallback; MiniCPM resampler init)"
    )


def patch_flagcx_comm_lifecycle():
    from .patches.patch_flagcx_comm import patch_flagcx_comm_lifecycle as _patch

    _patch()


def patch_flagcx_stream_adapter():
    from .patches.patch_flagcx_stream_adapter import (
        patch_flagcx_stream_adapter as _patch,
    )

    _patch()


def patch_flagcx_collective_hot_path():
    """Faster FlagCX collective wrappers for PTPU (dtype/op cache, in-place AR)."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        from vllm_fl.distributed.device_communicators.flagcx import (
            PyFlagcxCommunicator,
        )
        from vllm_fl.distributed.communicator import CommunicatorFL

        try:
            from vllm_fl.distributed.device_communicators.flagcx import (
                buffer_type,
                flagcxDataTypeEnum,
                flagcxRedOpTypeEnum,
            )
        except ImportError:
            logger.warning(
                "FlagCX symbols not importable; skipping collective hot-path "
                "patch (single-card path will not be affected)."
            )
            return

        if buffer_type is None or flagcxDataTypeEnum is None or flagcxRedOpTypeEnum is None:
            logger.warning(
                "FlagCX library not loaded (likely single-card path); "
                "skipping collective hot-path patch."
            )
            return

        from torch.distributed import ReduceOp

        # vllm.utils.current_stream is stale on PTPU; use torch.ptpu TLS.
        _live_current_stream = current_platform.torch_device_fn.current_stream

        if getattr(PyFlagcxCommunicator, "_sunrise_collective_hot_path_patched", False):
            return

        _dtype_enum_cache: dict = {}
        _op_enum_cache: dict = {}

        def _dtype_enum(dtype):
            cached = _dtype_enum_cache.get(dtype)
            if cached is None:
                cached = flagcxDataTypeEnum.from_torch(dtype)
                _dtype_enum_cache[dtype] = cached
            return cached

        def _op_enum(op):
            cached = _op_enum_cache.get(op)
            if cached is None:
                cached = flagcxRedOpTypeEnum.from_torch(op)
                _op_enum_cache[op] = cached
            return cached

        def _all_reduce_hot(
            self,
            in_tensor,
            out_tensor=None,
            op=ReduceOp.SUM,
            stream=None,
        ):
            if self.disabled:
                return None
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            if out_tensor is None:
                out_tensor = in_tensor

            cached_fn = getattr(self, "_sunrise_ar_fn", None)
            if cached_fn is None:
                cached_fn = self.flagcx._funcs["flagcxAllReduce"]
                self._sunrise_ar_fn = cached_fn

            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )

            device_ctx = getattr(self, "_device_ctx", None)
            if device_ctx is None:
                device_ctx = current_platform.torch_device_fn.device(self.device)
                self._device_ctx = device_ctx

            with device_ctx:
                rc = cached_fn(
                    buffer_type(in_tensor.data_ptr()),
                    buffer_type(out_tensor.data_ptr()),
                    in_tensor.numel(),
                    _dtype_enum(in_tensor.dtype),
                    _op_enum(op),
                    self.comm,
                    flagcx_stream,
                )
            if rc != 0:
                raise RuntimeError(
                    f"FLAGCX error: {self.flagcx.flagcxGetErrorString(rc)}"
                )
            return out_tensor

        PyFlagcxCommunicator.all_reduce = _all_reduce_hot

        def _get_default_stream_wrapper(self, _live=_live_current_stream):
            return self.flagcx.adaptor_stream_copy(_live())

        def _check_rc(self, rc):
            if rc != 0:
                raise RuntimeError(
                    f"FLAGCX error: {self.flagcx.flagcxGetErrorString(rc)}"
                )

        PyFlagcxCommunicator._get_default_stream_wrapper = _get_default_stream_wrapper
        PyFlagcxCommunicator._check_rc = _check_rc

        def _all_gather_hot(self, output_tensor, input_tensor, stream=None):
            if self.disabled:
                return
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            fn = getattr(self, "_sunrise_ag_fn", None)
            if fn is None:
                fn = self.flagcx._funcs["flagcxAllGather"]
                self._sunrise_ag_fn = fn
            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )
            self._check_rc(
                fn(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    input_tensor.numel(),
                    _dtype_enum(input_tensor.dtype),
                    self.comm,
                    flagcx_stream,
                )
            )

        def _reduce_scatter_hot(
            self, output_tensor, input_tensor, op=ReduceOp.SUM, stream=None
        ):
            if self.disabled:
                return
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            fn = getattr(self, "_sunrise_rs_fn", None)
            if fn is None:
                fn = self.flagcx._funcs["flagcxReduceScatter"]
                self._sunrise_rs_fn = fn
            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )
            self._check_rc(
                fn(
                    buffer_type(input_tensor.data_ptr()),
                    buffer_type(output_tensor.data_ptr()),
                    output_tensor.numel(),
                    _dtype_enum(input_tensor.dtype),
                    _op_enum(op),
                    self.comm,
                    flagcx_stream,
                )
            )

        def _broadcast_hot(self, tensor, src, stream=None):
            if self.disabled:
                return
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            fn = getattr(self, "_sunrise_bc_fn", None)
            if fn is None:
                fn = self.flagcx._funcs["flagcxBroadcast"]
                self._sunrise_bc_fn = fn
            # Sender provides sendbuff = tensor; receivers pass NULL sendbuff.
            tensor_ptr = tensor.data_ptr()
            if src == self.rank:
                sendbuff = buffer_type(tensor_ptr)
                recvbuff = buffer_type(tensor_ptr)
            else:
                sendbuff = buffer_type()
                recvbuff = buffer_type(tensor_ptr)
            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )
            self._check_rc(
                fn(
                    sendbuff,
                    recvbuff,
                    tensor.numel(),
                    _dtype_enum(tensor.dtype),
                    src,
                    self.comm,
                    flagcx_stream,
                )
            )

        def _send_hot(self, tensor, dst, stream=None):
            if self.disabled:
                return
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            fn = getattr(self, "_sunrise_send_fn", None)
            if fn is None:
                fn = self.flagcx._funcs["flagcxSend"]
                self._sunrise_send_fn = fn
            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )
            self._check_rc(
                fn(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    _dtype_enum(tensor.dtype),
                    dst,
                    self.comm,
                    flagcx_stream,
                )
            )

        def _recv_hot(self, tensor, src, stream=None):
            if self.disabled:
                return
            if self.comm is None:
                init_fn = getattr(self, "_ensure_initialized", None)
                if init_fn is not None:
                    init_fn()
            fn = getattr(self, "_sunrise_recv_fn", None)
            if fn is None:
                fn = self.flagcx._funcs["flagcxRecv"]
                self._sunrise_recv_fn = fn
            flagcx_stream = (
                self._get_default_stream_wrapper()
                if stream is None
                else self.flagcx.adaptor_stream_copy(stream)
            )
            self._check_rc(
                fn(
                    buffer_type(tensor.data_ptr()),
                    tensor.numel(),
                    _dtype_enum(tensor.dtype),
                    src,
                    self.comm,
                    flagcx_stream,
                )
            )

        PyFlagcxCommunicator.all_gather = _all_gather_hot
        PyFlagcxCommunicator.reduce_scatter = _reduce_scatter_hot
        PyFlagcxCommunicator.broadcast = _broadcast_hot
        PyFlagcxCommunicator.send = _send_hot
        PyFlagcxCommunicator.recv = _recv_hot

        PyFlagcxCommunicator._sunrise_collective_hot_path_patched = True

        if not getattr(CommunicatorFL, "_sunrise_all_reduce_inplace_patched", False):

            def _ar_inplace(self, input_):
                bound = self.__dict__.get("_sunrise_pfc_ar_bound")
                if bound is None:
                    pfc = self.pyflagcx_comm
                    if pfc is None:
                        out = input_.clone()
                        torch.distributed.all_reduce(
                            out, group=self.device_group
                        )
                        return out
                    bound = pfc.all_reduce
                    self._sunrise_pfc_ar_bound = bound
                return bound(input_, out_tensor=input_)

            CommunicatorFL.all_reduce = _ar_inplace
            CommunicatorFL._sunrise_all_reduce_inplace_patched = True

        logger.info("Patched FlagCX collective hot path for Sunrise/PTPU")
    except Exception as e:
        logger.warning(
            "Failed to patch FlagCX collective hot path for Sunrise: %s", e
        )


def patch_vllm_group_coordinator_dispatch():
    """Skip dead custom-op branches in GroupCoordinator AR/AG/RS on PTPU."""
    try:
        from vllm.distributed.parallel_state import GroupCoordinator
        from vllm.platforms import current_platform
    except Exception as e:
        logger.warning(
            "Failed to import GroupCoordinator (skipping dispatch patch): %s", e
        )
        return

    if getattr(GroupCoordinator, "_sunrise_phase_b3_patched", False):
        return

    # PTPU does not use torch custom-op collectives.
    try:
        uses_custom_op = current_platform.use_custom_op_collectives()
    except Exception:
        uses_custom_op = False
    if uses_custom_op:
        logger.info(
            "Skipping GroupCoordinator dispatch patch: "
            "use_custom_op_collectives() is True"
        )
        return

    def _fast_all_reduce(self, input_):
        if self.world_size == 1:
            return input_
        dc = self.device_communicator
        if dc is None:
            raise ValueError("No device communicator found")
        return dc.all_reduce(input_)

    def _fast_all_gather(self, input_, dim=-1):
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        dc = self.device_communicator
        if dc is None:
            raise ValueError("No device communicator found")
        return dc.all_gather(input_, dim)

    def _fast_reduce_scatter(self, input_, dim=-1):
        if self.world_size == 1:
            return input_
        assert -input_.dim() <= dim < input_.dim(), (
            f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
        )
        dc = self.device_communicator
        if dc is None:
            raise ValueError("No device communicator found")
        return dc.reduce_scatter(input_, dim)

    GroupCoordinator.all_reduce = _fast_all_reduce
    GroupCoordinator.all_gather = _fast_all_gather
    GroupCoordinator.reduce_scatter = _fast_reduce_scatter
    GroupCoordinator._sunrise_phase_b3_patched = True

    logger.info("Patched GroupCoordinator AR/AG/RS dispatch for Sunrise/PTPU")


def patch_op_manager_fast_path():
    """Cache resolved op fn on ``OpManager`` and call it directly when IO dump is off.

    Invalidate on both ``OpManager.policy_epoch`` and ``get_policy_epoch()`` so
    ``policy_context()`` / ``set_global_policy()`` (official CachedOp semantics)
    cannot leave a stale sunrise fast-path entry.
    """
    if os.environ.get("VLLM_FL_SUNRISE_OPMGR_FAST", "1") != "1":
        logger.info(
            "OpManager fast path disabled by VLLM_FL_SUNRISE_OPMGR_FAST=0"
        )
        return

    try:
        from vllm_fl.dispatch.manager import OpManager
        from vllm_fl.dispatch.io_dumper import is_dump_enabled
        from vllm_fl.dispatch.policy import get_policy_epoch
    except Exception as e:
        logger.warning(
            "Failed to import OpManager/is_dump_enabled (skipping): %s", e
        )
        return

    if getattr(OpManager, "_sunrise_fast_path_patched", False):
        return

    _orig_call = OpManager.call

    def _fast_call(self, op_name, *args, **kwargs):
        cache = getattr(self, "_sunrise_fast_cache", None)
        policy_epoch = get_policy_epoch()
        if cache is not None:
            entry = cache.get(op_name)
            if entry is not None:
                fn, mgr_epoch, pol_epoch = entry
                if (
                    mgr_epoch == self._state.policy_epoch
                    and pol_epoch == policy_epoch
                    and not is_dump_enabled()
                ):
                    return fn(*args, **kwargs)

        result = _orig_call(self, op_name, *args, **kwargs)

        if cache is None:
            cache = {}
            self._sunrise_fast_cache = cache
        try:
            fn = self.resolve(op_name)
            cache[op_name] = (fn, self._state.policy_epoch, policy_epoch)
        except Exception:
            pass
        return result

    OpManager.call = _fast_call
    OpManager._sunrise_fast_path_patched = True

    logger.info("Patched OpManager.call hot path for Sunrise/PTPU")


def patch_oot_layer_fast_path():
    """Resolve op fn once in OOT ``forward_oot`` when upstream lacks CachedOp.

    Official vllm-plugin-FL now routes ``RMSNormFL`` / ``SiluAndMulFL`` /
    ``GeluAndMulFL`` through ``CachedOp`` (policy-epoch aware + fallback).
    Replacing those ``forward_oot`` methods would drop that behavior, so this
    patch is a no-op when CachedOp is already in use. Keep the legacy
    resolve_op cache only for older trees that still call ``call_op`` directly.
    """
    if os.environ.get("VLLM_FL_SUNRISE_OOT_FAST", "1") != "1":
        logger.info(
            "OOT layer fast path disabled by VLLM_FL_SUNRISE_OOT_FAST=0"
        )
        return

    try:
        from vllm_fl.dispatch import CachedOp, resolve_op
        from vllm_fl.ops.layernorm import RMSNormFL
        from vllm_fl.ops.activation import SiluAndMulFL, GeluAndMulFL
        import vllm_fl.ops.layernorm as _layernorm_mod
        import vllm_fl.ops.activation as _activation_mod
    except Exception as e:
        logger.warning(
            "Failed to import OOT layer modules (skipping): %s", e
        )
        return

    # Upstream already ships CachedOp hot path — do not overwrite it.
    if any(
        isinstance(getattr(mod, name, None), CachedOp)
        for mod, name in (
            (_layernorm_mod, "_rms_norm"),
            (_activation_mod, "_silu_and_mul"),
            (_activation_mod, "_gelu_and_mul"),
        )
    ):
        logger.info(
            "Skipping sunrise OOT forward_oot patch; official CachedOp is active"
        )
        return

    def _make_passthrough(op_name):
        cache = [None]

        def _fast_forward_oot(self, *args, **kwargs):
            fn = cache[0]
            if fn is None:
                cache[0] = fn = resolve_op(op_name)
            return fn(self, *args, **kwargs)

        _fast_forward_oot.__name__ = f"_fast_forward_oot_{op_name}"
        _fast_forward_oot.__qualname__ = f"_fast_forward_oot[{op_name}]"
        return _fast_forward_oot

    patched = []

    if not getattr(RMSNormFL, "_sunrise_oot_fast_patched", False):
        RMSNormFL.forward_oot = _make_passthrough("rms_norm")
        RMSNormFL._sunrise_oot_fast_patched = True
        patched.append("RMSNormFL")

    if not getattr(SiluAndMulFL, "_sunrise_oot_fast_patched", False):
        SiluAndMulFL.forward_oot = _make_passthrough("silu_and_mul")
        SiluAndMulFL._sunrise_oot_fast_patched = True
        patched.append("SiluAndMulFL")

    if not getattr(GeluAndMulFL, "_sunrise_oot_fast_patched", False):
        GeluAndMulFL.forward_oot = _make_passthrough("gelu_and_mul")
        GeluAndMulFL._sunrise_oot_fast_patched = True
        patched.append("GeluAndMulFL")

    if patched:
        logger.info(
            "Patched OOT layer forward_oot for Sunrise/PTPU (%s)",
            ", ".join(patched),
        )


def patch_ptpu_topk_topp_sampler():
    """Use PTPU fused ``top_k_top_p_sampling_from_probs`` in ``TopKTopPSampler.forward_native``."""
    if os.environ.get("VLLM_FL_SUNRISE_PTPU_SAMPLER", "1") != "1":
        logger.info(
            "PTPU sampler patch disabled by VLLM_FL_SUNRISE_PTPU_SAMPLER=0"
        )
        return

    try:
        from vllm.v1.sample.ops.topk_topp_sampler import (
            TopKTopPSampler,
            random_sample,
        )
        import torch_ptpu.sgl_kernel as _ptpu_sgl
        _ptpu_topk_topp_fn = _ptpu_sgl.top_k_top_p_sampling_from_probs
    except Exception as e:
        logger.warning(
            "Failed to import deps for PTPU sampler patch (skipping): %s",
            e,
        )
        return

    if getattr(TopKTopPSampler, "_sunrise_ptpu_sampler_patched", False):
        return

    _orig_forward_native = TopKTopPSampler.forward_native

    def _ptpu_forward_native(self, logits, generators, k, p):
        if generators:
            return _orig_forward_native(self, logits, generators, k, p)

        if self.logprobs_mode in ("processed_logits", "processed_logprobs"):
            return _orig_forward_native(self, logits, generators, k, p)

        if k is None and p is None:
            probs = logits.softmax(dim=-1, dtype=torch.float32)
            return random_sample(probs, generators), None

        probs = logits.softmax(dim=-1, dtype=torch.float32)
        bs = probs.shape[0]
        uniform_samples = torch.rand(
            (32, bs), dtype=torch.float32, device=probs.device
        )

        k_arg = k if k is not None else probs.shape[-1]
        p_arg = p if p is not None else 1.0

        if isinstance(k_arg, torch.Tensor) and k_arg.dtype != torch.int32:
            k_arg = k_arg.to(torch.int32)
        if isinstance(p_arg, torch.Tensor) and p_arg.dtype != torch.float32:
            p_arg = p_arg.to(torch.float32)

        token_ids, _success = _ptpu_topk_topp_fn(
            probs,
            uniform_samples,
            k_arg,
            p_arg,
            filter_apply_order="joint",
            deterministic=True,
            check_nan=False,
        )
        return token_ids, None

    TopKTopPSampler.forward_native = _ptpu_forward_native
    TopKTopPSampler._sunrise_ptpu_sampler_patched = True

    logger.info(
        "Patched TopKTopPSampler.forward_native for PTPU fused top-k/top-p sampling"
    )


def patch_distributed_runtime():
    """Keep FlagCX path while mapping torch ProcessGroup backend to pccl."""
    try:
        from vllm.platforms import current_platform
        from vllm.distributed.device_communicators.base_device_communicator import (
            DeviceCommunicatorBase,
        )
        from vllm_fl.distributed.communicator import CommunicatorFL
        from vllm_fl.worker import worker as worker_mod

        platform_cls = (
            current_platform
            if isinstance(current_platform, type)
            else current_platform.__class__
        )

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        platform_cls.dist_backend = "flagcx"
        current_platform.dist_backend = "flagcx"

        if not getattr(CommunicatorFL, "_sunrise_all_gather_patched", False):
            def _all_gather(self, input_: torch.Tensor, dim: int = -1):
                world_size = self.world_size
                if world_size == 1:
                    return input_

                assert -input_.dim() <= dim < input_.dim(), (
                    f"Invalid dim ({dim}) for input tensor with shape {input_.size()}"
                )
                if dim < 0:
                    dim += input_.dim()

                pyflagcx_comm = getattr(self, "pyflagcx_comm", None)
                if pyflagcx_comm is None or pyflagcx_comm.disabled:
                    return DeviceCommunicatorBase.all_gather(self, input_, dim)

                output_tensor = self.all_gatherv(input_, dim=0, sizes=None)
                if dim == 0:
                    return output_tensor

                input_size = input_.size()
                output_tensor = output_tensor.reshape((world_size,) + input_size)
                output_tensor = output_tensor.movedim(0, dim)
                output_tensor = output_tensor.reshape(
                    input_size[:dim]
                    + (world_size * input_size[dim],)
                    + input_size[dim + 1 :]
                )
                return output_tensor

            CommunicatorFL.all_gather = _all_gather
            CommunicatorFL._sunrise_all_gather_patched = True

        init_dist = worker_mod.init_worker_distributed_environment
        if not getattr(init_dist, "_sunrise_backend_patched", False):
            def _init_worker_distributed_environment(
                vllm_config,
                rank,
                distributed_init_method=None,
                local_rank=-1,
                backend="nccl",
            ):
                backend_for_pg = backend
                if backend in ("flagcx", "nccl"):
                    backend_for_pg = "pccl"
                return init_dist(
                    vllm_config,
                    rank,
                    distributed_init_method=distributed_init_method,
                    local_rank=local_rank,
                    backend=backend_for_pg,
                )

            _init_worker_distributed_environment._sunrise_backend_patched = True
            worker_mod.init_worker_distributed_environment = (
                _init_worker_distributed_environment
            )

        logger.info(
            "Configured Sunrise/PTPU to use FlagCX communicator with pccl PGs"
        )
    except Exception as e:
        logger.warning("Failed to configure Sunrise distributed runtime: %s", e)


def patch_op_cls():
    """Register Sunrise replacements for upstream custom ops."""
    try:
        from vllm.model_executor.custom_op import PluggableLayer

        from .impl.vocab_parallel_embedding import SunriseVocabParallelEmbedding

        PluggableLayer.register_oot(
            _decorated_layer_cls=SunriseVocabParallelEmbedding,
            name="VocabParallelEmbedding",
        )
        logger.info("Patched VocabParallelEmbedding for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch VocabParallelEmbedding for Sunrise: %s", e)

    from .impl.gemma_rms_norm import register_gemma_rms_norm_oot

    register_gemma_rms_norm_oot()

    from .impl.rms_norm_gated import register_rms_norm_gated_oot

    register_rms_norm_gated_oot()


def patch_accelerator_empty_cache():
    """Redirect ``torch.accelerator.empty_cache()`` to ``torch.ptpu.empty_cache()``."""
    try:
        import torch.accelerator as _accel

        if getattr(_accel, "_sunrise_empty_cache_patched", False):
            return
        _accel.empty_cache = torch.ptpu.empty_cache
        _accel._sunrise_empty_cache_patched = True
        logger.info("Patched torch.accelerator.empty_cache for Sunrise/PTPU")
    except Exception as e:
        logger.warning("Failed to patch torch.accelerator.empty_cache: %s", e)


def patch_fused_topk_bias_router():
    """Native top-k router on PTPU (replaces CUDA-only ``vllm_topk_{sigmoid,softmax}``)."""

    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return

        import vllm.envs as envs
        from vllm.model_executor.layers.fused_moe.router import (
            fused_topk_bias_router as router_mod,
        )

        if getattr(router_mod, "_sunrise_topk_patched", False):
            return

        def _native_topk_with_bias(
            topk_weights: torch.Tensor,
            topk_indices: torch.Tensor,
            token_expert_indices: torch.Tensor,
            gating_output: torch.Tensor,
            renormalize: bool,
            e_score_correction_bias: torch.Tensor | None,
            scoring_func: str,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            if scoring_func == "softmax":
                scores = gating_output.softmax(dim=-1)
            elif scoring_func == "sigmoid":
                scores = gating_output.sigmoid()
            else:
                raise ValueError(
                    f"Unsupported scoring function: {scoring_func}"
                )

            if e_score_correction_bias is not None:
                scores_for_choice = scores + e_score_correction_bias.to(
                    scores.dtype
                ).unsqueeze(0)
            else:
                scores_for_choice = scores

            topk = topk_weights.shape[-1]
            use_sorted = getattr(envs, "VLLM_BATCH_INVARIANT", False)
            chosen_indices = torch.topk(
                scores_for_choice, k=topk, dim=-1, sorted=use_sorted
            ).indices
            chosen_weights = scores.gather(dim=-1, index=chosen_indices)
            if renormalize:
                chosen_weights = chosen_weights / chosen_weights.sum(
                    dim=-1, keepdim=True
                )

            topk_weights.copy_(chosen_weights.to(topk_weights.dtype))
            topk_indices.copy_(chosen_indices.to(topk_indices.dtype))
            return topk_weights, topk_indices

        def _patched_vllm_topk_sigmoid(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize=False, e_score_correction_bias=None,
        ):
            return _native_topk_with_bias(
                topk_weights, topk_indices, token_expert_indices,
                gating_output, renormalize, e_score_correction_bias,
                scoring_func="sigmoid",
            )

        def _patched_vllm_topk_softmax(
            topk_weights, topk_indices, token_expert_indices,
            gating_output, renormalize=False, e_score_correction_bias=None,
        ):
            return _native_topk_with_bias(
                topk_weights, topk_indices, token_expert_indices,
                gating_output, renormalize, e_score_correction_bias,
                scoring_func="softmax",
            )

        router_mod.vllm_topk_sigmoid = _patched_vllm_topk_sigmoid
        router_mod.vllm_topk_softmax = _patched_vllm_topk_softmax
        router_mod._sunrise_topk_patched = True

        logger.info(
            "Patched fused_topk_bias_router to use native topk on PTPU"
        )
    except Exception as e:
        logger.warning(
            "Failed to patch fused_topk_bias_router for Sunrise: %s", e
        )


def patch_memory_profiling_for_plain_allocator():
    """Correct KV-cache sizing when PTPU ``torch.memory_reserved()`` is always zero."""
    try:
        from vllm.platforms import current_platform

        if getattr(current_platform, "device_type", None) != "ptpu":
            return
    except Exception as e:
        logger.warning("Skip PTPU memory profiling patch: %s", e)
        return

    try:
        import threading
        import time
        from contextlib import contextmanager

        from vllm.platforms import current_platform as _cp
        from vllm_fl.worker import worker as worker_mod

        original = worker_mod.memory_profiling_fl
        if getattr(original, "_sunrise_mem_profile_patched", False):
            return

        gib = float(2**30)
        _logged = {"once": False}

        # PTPU plain allocator lacks peak stats; sample free memory during profile_run.
        try:
            _act_factor = float(os.environ.get("VLLM_FL_ACT_HEADROOM_FACTOR", "1.15"))
        except ValueError:
            _act_factor = 1.15
        try:
            _act_min = float(os.environ.get("VLLM_FL_ACT_HEADROOM_MIN_GIB", "1.0")) * gib
        except ValueError:
            _act_min = gib
        try:
            _sample_s = float(os.environ.get("VLLM_FL_ACT_SAMPLE_MS", "2")) / 1000.0
        except ValueError:
            _sample_s = 0.002

        def _mem_free():
            try:
                free, _total = _cp.torch_device_fn.mem_get_info()
                return int(free)
            except Exception:
                return None

        @contextmanager
        def memory_profiling_fl_ptpu(baseline_snapshot, weights_memory):
            # Track the minimum free memory seen while the profiling forward
            # runs -> the maximum device usage, i.e. the true activation peak.
            state = {"min_free": None}
            stop = threading.Event()

            def _sample():
                while not stop.is_set():
                    free = _mem_free()
                    if free is not None and (
                        state["min_free"] is None or free < state["min_free"]
                    ):
                        state["min_free"] = free
                    time.sleep(_sample_s)

            with original(
                baseline_snapshot, weights_memory=weights_memory
            ) as result:
                sampler = threading.Thread(target=_sample, daemon=True)
                sampler.start()
                try:
                    yield result
                finally:
                    # Drain async work so a lagging peak is still observed,
                    # then take one last sample before stopping the thread.
                    try:
                        _cp.torch_device_fn.synchronize()
                    except Exception:
                        pass
                    free = _mem_free()
                    if free is not None and (
                        state["min_free"] is None or free < state["min_free"]
                    ):
                        state["min_free"] = free
                    stop.set()
                    sampler.join(timeout=1.0)

            if (
                result.after_profile.torch_memory == 0
                and result.weights_memory > 0
            ):
                actual_used = (
                    result.after_profile.cuda_memory
                    - result.before_create.cuda_memory
                )
                corrected_non_torch = max(0, actual_used - result.weights_memory)

                # Measured transient activation peak = drop in free memory
                # during the forward, relative to the pre-forward baseline.
                measured_peak = 0
                if (
                    state["min_free"] is not None
                    and result.before_profile.free_memory > 0
                ):
                    measured_peak = max(
                        0, result.before_profile.free_memory - state["min_free"]
                    )
                # Fall back to any torch-reported peak if sampling failed.
                peak_act = max(result.torch_peak_increase, measured_peak)
                # Reserve with a safety margin: runtime steps hit shapes /
                # autotune buffers the single profiling pass may not.
                peak_reserve = max(int(peak_act * _act_factor), int(_act_min))

                result.torch_peak_increase = peak_reserve
                result.non_torch_increase = corrected_non_torch
                result.non_kv_cache_memory = (
                    corrected_non_torch + peak_reserve + result.weights_memory
                )

                if not _logged["once"]:
                    _logged["once"] = True
                    logger.info(
                        "PTPU plain-allocator memory accounting fix applied: "
                        "weights=%.2f GiB measured_peak_act=%.2f GiB "
                        "reserved_peak_act=%.2f GiB actual_used=%.2f GiB "
                        "-> non_torch=%.2f GiB non_kv_cache=%.2f GiB",
                        result.weights_memory / gib,
                        measured_peak / gib,
                        peak_reserve / gib,
                        actual_used / gib,
                        corrected_non_torch / gib,
                        result.non_kv_cache_memory / gib,
                    )

        memory_profiling_fl_ptpu._sunrise_mem_profile_patched = True
        worker_mod.memory_profiling_fl = memory_profiling_fl_ptpu
        logger.info(
            "Patched memory_profiling_fl for PTPU plain allocator "
            "(prevents weights double-counting in KV cache sizing)"
        )
    except Exception as e:
        logger.warning(
            "Failed to patch memory_profiling_fl for Sunrise/PTPU: %s", e
        )


def patch_hy_v3_shared_mlp_weights() -> None:
    """Remap Hy-MT2 shared-expert checkpoint names when vLLM hy_v3 lacks ``shared_mlp`` prefix.

    Hy-MT2 checkpoints store shared MLP weights under ``*.mlp.shared_mlp.*``.
    Upstream vLLM ``HYV3MoEFused`` originally registered parameters as
    ``*.mlp.gate_up_proj`` (missing the ``shared_mlp`` segment). Newer vLLM
    builds may already use ``prefix=f"{prefix}.shared_mlp"``; in that case
    this patch detects the correct param layout and skips remapping.
    """
    try:
        from vllm.model_executor.models import hy_v3 as hy_v3_mod
    except ImportError:
        return

    if getattr(hy_v3_mod.HYV3Model, "_sunrise_shared_mlp_weight_patch", False):
        return

    _orig_load_weights = hy_v3_mod.HYV3Model.load_weights

    def _remap_shared_mlp_name(name: str) -> str:
        if ".shared_mlp." not in name:
            return name
        return (
            name.replace(".shared_mlp.gate_proj", ".gate_proj")
            .replace(".shared_mlp.up_proj", ".up_proj")
            .replace(".shared_mlp.down_proj", ".down_proj")
        )

    def _needs_shared_mlp_remap(self) -> bool:
        for param_name, _ in self.named_parameters():
            if ".mlp.shared_mlp.gate_up_proj" in param_name:
                return False
        for param_name, _ in self.named_parameters():
            if ".mlp.gate_up_proj" in param_name and ".shared_mlp." not in param_name:
                return True
        return False

    def _load_weights(self, weights):
        if not _needs_shared_mlp_remap(self):
            return _orig_load_weights(self, weights)

        def _iter_remapped():
            for name, tensor in weights:
                yield _remap_shared_mlp_name(name), tensor

        logger.info(
            "HYV3Model.load_weights: remapping shared_mlp checkpoint names "
            "(upstream hy_v3 missing .shared_mlp param prefix)"
        )
        return _orig_load_weights(self, _iter_remapped())

    hy_v3_mod.HYV3Model.load_weights = _load_weights
    hy_v3_mod.HYV3Model._sunrise_shared_mlp_weight_patch = True
    logger.info(
        "Patched HYV3Model.load_weights for Hy-MT2 shared_mlp compatibility"
    )

