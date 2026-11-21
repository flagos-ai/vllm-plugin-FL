# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Opt-in PR773 provider for vLLM 0.32's existing allreduce/RMSNorm pass.

Set VLLM_FL_FUSED_ALLREDUCE_RMS_PROVIDER=peer (single host) or mnnvl
(explicit multicast). All TP workers must receive the same plugin and launcher
environment. Call apply_allreduce_rms_provider_patch before model compilation.
The native pass must still be enabled, and FlashInfer must initialize its native
workspace: it supplies the original graph patterns and the fallback provider.

Workspace creation and the only CPU collectives happen before pattern tracing,
outside CUDA graph capture. No exception retries a collective after launch.
This does not enable TP16/32, FlagCX, or force a quantization backend.
"""

from __future__ import annotations

import importlib
import logging
import os
import socket
from collections import Counter
from functools import wraps
from threading import RLock
from types import SimpleNamespace

logger = logging.getLogger("vllm.fl.allreduce_rms_provider")
_ENV = "VLLM_FL_FUSED_ALLREDUCE_RMS_PROVIDER"
_OFF = ("", "0", "off", "native")
_LOCK = RLock()
_WORKSPACES = {}
_DISABLED_KEYS = set()
_HOST_DISPATCHES = Counter()
_NATIVE_CALL = None


def _requested_backend() -> str | None:
    value = os.environ.get(_ENV, "").strip().lower()
    if value in _OFF:
        return None
    if value not in ("peer", "mnnvl"):
        raise ValueError(f"{_ENV} must be peer, mnnvl, or off; got {value!r}")
    return value


def allreduce_rms_provider_requested() -> bool:
    """Gate native RMSNorm IR emission in the FL OOT compile branch.

    A declined workspace remains safe: the original vLLM pass consumes the IR.
    FlagCX never reaches either fused communicator through this integration.
    """
    return "FLAGCX_PATH" not in os.environ and _requested_backend() is not None


def _agree_preflight(reports: list[dict]) -> tuple[bool, str]:
    """Pure metadata policy; the decision is identical on every TP rank."""
    failures = [f"rank {i}: {r['error']}" for i, r in enumerate(reports) if r["error"]]
    if failures:
        return False, "; ".join(failures)
    signatures = {r["signature"] for r in reports}
    if len(signatures) != 1:
        return False, "TP ranks disagree on provider, dimensions, dtype, or topology"
    if len({r.get("cached", False) for r in reports}) != 1:
        return False, "TP ranks disagree on workspace cache state"
    signature = reports[0]["signature"]
    if signature[0] == "peer":
        if len({r["host"] for r in reports}) != 1:
            return False, "peer backend requires every TP rank on the same host"
        if len({r["visible_devices"] for r in reports}) != 1:
            return False, "peer backend requires a consistent CUDA_VISIBLE_DEVICES map"
        if len({r["device"] for r in reports}) != len(reports):
            return False, "peer backend requires distinct CUDA devices on each TP rank"
    return True, "ready"


def _workspace_key(group, device: int, world_size, max_tokens, hidden, dtype):
    return (id(group), device, world_size, max_tokens, hidden, dtype)


def _prepare_for_pass(pass_obj):
    """Collectively preflight then eagerly allocate, before fake-mode tracing."""
    import torch
    import torch.distributed as dist

    group = pass_obj.group  # TP cpu_group, never the global WORLD group.
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("PR773 workspace preparation must precede CUDA graph capture")
    provider = None
    backend = None
    error = None
    device = torch.cuda.current_device()
    key = _workspace_key(
        group, device, pass_obj.tp_size, pass_obj.max_token_num,
        pass_obj.workspace_hidden_dim, pass_obj.model_dtype,
    )
    try:
        backend = _requested_backend()
        if backend is None:
            raise ValueError("provider not requested")
        if "FLAGCX_PATH" in os.environ or "flagcx" in str(dist.get_backend(group)).lower():
            raise ValueError("FlagCX communicator must not be bypassed")
        if pass_obj.tp_size not in (2, 4, 8):
            raise ValueError("PR773 supports TP2/4/8 only")
        if dist.get_world_size(group) != pass_obj.tp_size:
            raise ValueError("native fusion TP size differs from supplied TP process group")
        if pass_obj.model_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("PR773 requires FP16 or BF16")
        if min(pass_obj.max_token_num, pass_obj.workspace_hidden_dim) <= 0:
            raise ValueError("invalid workspace dimensions")
        if torch.cuda.get_device_capability(device)[0] < 9:
            raise ValueError("PR773 requires NVIDIA SM90 or newer")
        provider = importlib.import_module("vllm_fl.patches._allreduce_rms_pr773")
        if not provider._HAS_PDL:
            raise ValueError("native fusion requires Triton CUDA PDL support")
        if backend == "peer" and provider.symm_mem is None:
            raise ValueError("PyTorch symmetric memory is unavailable")
        if backend == "mnnvl" and (
            not provider._HAS_MNNVL_RUNTIME
            or not provider.is_multicast_supported(device)
        ):
            raise ValueError("explicit MNNVL requires multicast on every TP rank")
        if key in _WORKSPACES and _WORKSPACES[key][1].backend != backend:
            raise ValueError("provider cannot change after TP workspace initialization")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    report = {
        "error": error,
        "signature": (
            backend, pass_obj.tp_size, pass_obj.max_token_num,
            pass_obj.workspace_hidden_dim, str(pass_obj.model_dtype),
        ),
        "host": socket.gethostname(),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": device,
        "cached": key in _WORKSPACES,
    }
    reports = [None] * dist.get_world_size(group)
    dist.all_gather_object(reports, report, group=group)
    ready, reason = _agree_preflight(reports)
    if ready and backend == "peer":
        # The protocol dereferences every peer pointer. Check the actual TP
        # device set, rather than unrelated visible GPUs on this host.
        try:
            peer_ok = all(
                r["device"] == device
                or torch.cuda.can_device_access_peer(device, r["device"])
                for r in reports
            )
        except Exception:
            peer_ok = False
        peer_reports = [None] * len(reports)
        dist.all_gather_object(peer_reports, peer_ok, group=group)
        ready = all(peer_reports)
        reason = "peer access unavailable on at least one TP rank" if not ready else reason
    if not ready:
        _DISABLED_KEYS.add(key)
        logger.warning("PR773 provider declined on all TP ranks: %s; using native fusion", reason)
        return False
    _DISABLED_KEYS.discard(key)
    # Collective allocation failures are fatal, not a local fallback. Retrying
    # with another provider after peers enter rendezvous can hang the TP group.
    if key not in _WORKSPACES:
        workspace = provider.create_fused_allreduce_rms_norm_workspace(
            world_size=pass_obj.tp_size, rank=dist.get_rank(group),
            max_token_num=pass_obj.max_token_num,
            hidden_dim=pass_obj.workspace_hidden_dim,
            dtype=pass_obj.model_dtype, group=group, backend=backend,
        )
        _WORKSPACES[key] = (provider, workspace)
    logger.info(
        "PR773 provider ready backend=%s TP=%s rank=%s hidden=%s max_tokens=%s dtype=%s",
        backend, pass_obj.tp_size, dist.get_rank(group),
        pass_obj.workspace_hidden_dim, pass_obj.max_token_num, pass_obj.model_dtype,
    )
    return True


def _dispatch_fused_call(kwargs):
    if "FLAGCX_PATH" in os.environ:
        # A runtime backend change is a launcher error: native FlashInfer also
        # bypasses FlagCX, so it is not an admissible fallback here.
        raise RuntimeError("PR773 fused graph cannot run after selecting FlagCX")
    import torch
    from vllm.distributed import get_tp_group

    x = kwargs["allreduce_in"]
    key = _workspace_key(
        get_tp_group().cpu_group, x.device.index,
        kwargs["world_size"], kwargs["max_token_num"], x.shape[-1], x.dtype,
    )
    state = None if key in _DISABLED_KEYS else _WORKSPACES.get(key)
    # A narrower speculative draft can reuse native vLLM's larger workspace.
    # PR773 requires exact hidden width. This metadata-only decision is uniform
    # for a valid TP graph; it never depends on tensor values or kernel errors.
    if state is None:
        _HOST_DISPATCHES["native"] += 1
        return _NATIVE_CALL(**kwargs)
    provider, workspace = state
    pattern = kwargs["pattern_code"]
    pattern_map = {
        1: provider.AllReduceFusionPattern.kARResidualRMSNorm,
        2: provider.AllReduceFusionPattern.kARResidualRMSNormFP8Quant,
        3: provider.AllReduceFusionPattern.kARResidualRMSNormFP4Quant,
    }
    # Native enum values are checked at installation; reject future schemas.
    if pattern not in pattern_map:
        raise ValueError(f"unsupported PR773 fusion pattern {pattern}")
    kwargs = dict(kwargs, pattern_code=pattern_map[pattern])
    counter = f"{workspace.backend}:pattern={pattern}"
    _HOST_DISPATCHES[counter] += 1
    if _HOST_DISPATCHES[counter] == 1:
        logger.info(
            "PR773 first host dispatch backend=%s TP=%s shape=%s pattern=%s capture=%s",
            workspace.backend, kwargs["world_size"], tuple(x.shape), pattern,
            torch.cuda.is_current_stream_capturing(),
        )
    # In-place destinations, quant scale views, fp32_acc and PDL arguments are
    # forwarded exactly. Errors propagate; never retry an entered collective.
    return provider.fused_allreduce_rms_norm(**kwargs, workspace=workspace)


def allreduce_rms_provider_stats() -> dict:
    """Host-dispatch evidence only; CUDA graph replays do not execute Python."""
    return {
        "workspaces": len(_WORKSPACES),
        "host_dispatches": dict(_HOST_DISPATCHES),
        "cuda_graph_replays_counted": False,
    }


def initialize_allreduce_rms_provider(config) -> bool:
    """Eager worker initialization, including when AOT skips pass construction.

    Call after NativeGPUWorker.init_device has created CUDA and TP groups and
    before load_model/compile/profile/CUDA graphs. The ordinary native pass
    constructor also performs initialization for callers without this worker
    integration, but that alone cannot cover a saved AOT graph cache hit.
    """
    if not allreduce_rms_provider_requested():
        return False
    import torch
    from vllm.config import CompilationMode
    from vllm.distributed import get_tp_group

    compilation = config.compilation_config
    if (
        not compilation.pass_config.fuse_allreduce_rms
        or compilation.mode == CompilationMode.NONE
        or config.model_config is None
    ):
        return False
    if not apply_allreduce_rms_provider_patch():
        return False
    tp = get_tp_group()
    if tp.world_size not in (2, 4, 8):
        return False
    max_size = compilation.pass_config.flashinfer_max_size(tp.world_size)
    if max_size is None:
        return False
    fusion = importlib.import_module(
        "vllm.compilation.passes.fusion.allreduce_rms_fusion"
    )
    hidden = fusion._fused_ar_workspace_hidden_dim(config)
    dtype = config.model_config.dtype
    element_size = torch.empty((), dtype=dtype).element_size()
    max_tokens = min(
        max_size // (hidden * element_size),
        config.scheduler_config.max_num_batched_tokens,
    )
    params = SimpleNamespace(
        group=tp.cpu_group, tp_size=tp.world_size,
        max_token_num=max_tokens, workspace_hidden_dim=hidden, model_dtype=dtype,
    )
    ready = _prepare_for_pass(params)
    workspace_kwargs = dict(
        world_size=tp.world_size, rank=tp.rank_in_group,
        max_token_num=max_tokens, hidden_dim=hidden, dtype=dtype,
        group=tp.cpu_group,
    )
    # Allocate native fallbacks too: a narrower speculative draft can require
    # them even when an AOT cache hit bypasses every fusion-pass constructor.
    # Peer allocation above must precede native MNNVL VMM allocation.
    if fusion.get_fi_ar_workspace(**workspace_kwargs) is None:
        raise RuntimeError("PR773 startup could not initialize the native fallback workspace")
    quant_ready = fusion.get_fi_ar_quant_workspace(**workspace_kwargs) is not None
    logger.info(
        "PR773 worker startup initialized TP=%s hidden=%s max_tokens=%s "
        "provider=%s native_quant=%s before AOT cache loading",
        tp.world_size, hidden, max_tokens, ready, quant_ready,
    )
    return ready


def apply_allreduce_rms_provider_patch() -> bool:
    """Install the controlled pattern hook; no GPU allocation occurs here."""
    if not allreduce_rms_provider_requested():
        return False
    global torch, _NATIVE_CALL, logger
    import torch
    from vllm.logger import init_logger
    from vllm.utils.torch_utils import direct_register_custom_op

    # vLLM configures handlers/levels on the "vllm" logger tree only. An
    # unrelated "vllm_fl" name would still inherit root's WARNING level.
    logger = init_logger("vllm.fl.allreduce_rms_provider")
    fusion = importlib.import_module(
        "vllm.compilation.passes.fusion.allreduce_rms_fusion"
    )
    if fusion.flashinfer_comm is None:
        logger.warning("PR773 provider needs vLLM's native FlashInfer fusion pass")
        return False
    with _LOCK:
        original_register = fusion.AllReduceFusionPass.register_patterns
        if getattr(original_register, "_fl_pr773", False):
            return True
        enum = fusion.flashinfer_comm.AllReduceFusionPattern
        if tuple(int(getattr(enum, name)) for name in (
            "kARResidualRMSNorm", "kARResidualRMSNormFP8Quant", "kARResidualRMSNormFP4Quant"
        )) != (1, 2, 3):
            raise RuntimeError("installed FlashInfer fusion enum differs from PR773")
        _NATIVE_CALL = fusion.call_trtllm_fused_allreduce_norm

        def call_fl_pr773_fused_allreduce_norm(
            allreduce_in: torch.Tensor,
            residual: torch.Tensor,
            rms_gamma: torch.Tensor,
            rms_eps: float,
            world_size: int,
            launch_with_pdl: bool,
            fp32_acc: bool,
            max_token_num: int,
            pattern_code: int,
            norm_out: torch.Tensor | None = None,
            quant_out: torch.Tensor | None = None,
            scale_out: torch.Tensor | None = None,
            scale_factor: torch.Tensor | None = None,
            weight_bias: float = 0.0,
        ) -> None:
            _dispatch_fused_call(locals())

        direct_register_custom_op(
            op_name="fl_pr773_fused_allreduce_norm",
            op_func=call_fl_pr773_fused_allreduce_norm,
            mutates_args=["allreduce_in", "residual", "norm_out", "quant_out", "scale_out"],
        )
        replacement_op = torch.ops.vllm.fl_pr773_fused_allreduce_norm.default

        original_init = fusion.AllReduceFusionPass.__init__
        original_apply = fusion.AllReduceFusionPass.__call__

        @wraps(original_init)
        def initialize_pass(self, *args, **kwargs):
            # PyTorch symmetric peer memory must precede MNNVL VMM allocation.
            # Intercept only this pass module's local allocator reference while
            # its constructor runs. The FlashInfer module API is unchanged.
            with _LOCK:
                native_get_workspace = fusion.get_fi_ar_workspace

                def get_workspace(**workspace_kwargs):
                    params = SimpleNamespace(
                        group=workspace_kwargs["group"],
                        tp_size=workspace_kwargs["world_size"],
                        max_token_num=workspace_kwargs["max_token_num"],
                        workspace_hidden_dim=workspace_kwargs["hidden_dim"],
                        model_dtype=workspace_kwargs["dtype"],
                    )
                    self._fl_pr773_ready = _prepare_for_pass(params)
                    return native_get_workspace(**workspace_kwargs)

                fusion.get_fi_ar_workspace = get_workspace
                try:
                    return original_init(self, *args, **kwargs)
                finally:
                    fusion.get_fi_ar_workspace = native_get_workspace

        @wraps(original_register)
        def register_patterns(self):
            ready = getattr(self, "_fl_pr773_ready", False)
            # Keep both initial registration and deferred replacement tracing
            # scoped. Torch 2.13 traces replace_fn at match time in check_fn,
            # rather than freezing its operator globals at registration time.
            with _LOCK:
                old_op = fusion.flashinfer_trtllm_fused_allreduce_norm
                if ready:
                    fusion.flashinfer_trtllm_fused_allreduce_norm = replacement_op
                try:
                    return original_register(self)
                finally:
                    fusion.flashinfer_trtllm_fused_allreduce_norm = old_op

        @wraps(original_apply)
        def apply_patterns(self, graph):
            with _LOCK:
                old_op = fusion.flashinfer_trtllm_fused_allreduce_norm
                if getattr(self, "_fl_pr773_ready", False):
                    fusion.flashinfer_trtllm_fused_allreduce_norm = replacement_op
                try:
                    return original_apply(self, graph)
                finally:
                    fusion.flashinfer_trtllm_fused_allreduce_norm = old_op

        register_patterns._fl_pr773 = True
        fusion.AllReduceFusionPass.__init__ = initialize_pass
        fusion.AllReduceFusionPass.__call__ = apply_patterns
        fusion.AllReduceFusionPass.register_patterns = register_patterns
    logger.info("Installed opt-in PR773 provider hook (%s)", _requested_backend())
    return True
