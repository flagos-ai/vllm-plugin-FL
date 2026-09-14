# SPDX-License-Identifier: Apache-2.0
"""Per-layer activation probe for GLM-5-Next bring-up on DCU.

Opt-in via ``VLLM_FL_LAYER_PROBE=1``. Wraps ``Glm5NextDecoderLayer.forward``
and logs per-layer statistics of the 4-tuple ``(hidden_states, residual,
post, comb)`` so the first diverging layer can be identified without a
reference implementation.

Only rank 0 logs, only for the first ``VLLM_FL_LAYER_PROBE_STEPS`` (default 2)
forward passes, so this stays usable on a live server.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Sentinel file: survives vLLM's env filtering into worker processes.
_SENTINEL = "/tmp/vllm_fl_layer_probe"

# Output file. The probe previously logged via init_logger, but
# vllm_fl.patches.* records never reached the server log (vllm_fl.dispatch.*
# ones did), so a log-only probe is unfalsifiable: silence could mean "not
# installed" or "installed but not logging". Writing to a file, append-mode,
# one line per record, removes that ambiguity -- and per-rank paths keep the
# 16 TP workers from interleaving.
_OUT_DIR = "/tmp/vllm_fl_probe_out"


def _out_path() -> str:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = os.getpid()
    return f"{_OUT_DIR}/rank{rank}.log"


def _emit(line: str) -> None:
    """Record a probe line to both the logger and the output file."""
    logger.info("%s", line)
    try:
        os.makedirs(_OUT_DIR, exist_ok=True)
        with open(_out_path(), "a") as fh:
            fh.write(line + "\n")
    except Exception:
        pass

_PROBE_STATE = {"step": 0, "armed": False}


def arm_probe() -> None:
    """Enable probing. Called after startup so the profile run is skipped.

    The profile run uses dummy inputs and runs before KV cache allocation, so
    its activations say nothing about a real prefill -- and instrumenting it
    only risks perturbing memory profiling.
    """
    _PROBE_STATE["armed"] = True
    _emit("PROBE-ARMED")


def _stat(name: str, t) -> str:
    if t is None:
        return f"{name}=None"
    if not isinstance(t, torch.Tensor):
        return f"{name}=<{type(t).__name__}>"
    f = t.detach().float()
    # NOTE: deliberately avoid boolean-mask indexing (f[torch.isfinite(f)]).
    # That lowers to nonzero(), which forces a device->host sync and, on this
    # stack, dispatches into a FlagGems path that fails with
    # "Triton Error [HIP]: invalid argument" inside the profile run.
    # nan_to_num + reductions keep everything on-device and shape-static.
    n_nan = int(torch.isnan(f).sum())
    n_inf = int(torch.isinf(f).sum())
    clean = torch.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    n_total = f.numel()
    n_finite = n_total - n_nan - n_inf
    if n_finite <= 0:
        return f"{name}[{tuple(t.shape)}] ALL-NONFINITE nan={n_nan} inf={n_inf}"
    # Sums are over the zero-filled tensor, so non-finite slots contribute 0.
    ssum = float(clean.pow(2).sum())
    rms = (ssum / n_finite) ** 0.5
    amax = float(clean.abs().max())
    mean = float(clean.sum()) / n_finite
    return (
        f"{name}[{tuple(t.shape)}] rms={rms:.4e} absmax={amax:.4e} "
        f"mean={mean:+.3e} nan={n_nan} inf={n_inf}"
    )


def install_layer_probe() -> None:
    # NOTE: vLLM spawns EngineCore/Worker processes with a *filtered* environment
    # (it warns "Unknown vLLM environment variable detected" for anything not on
    # its allowlist). VLLM_FL_LAYER_PROBE therefore reaches the API server but
    # NOT the workers -- and the workers are where the model actually runs.
    # So fall back to a sentinel file, which every process can see. serve_node.sh
    # creates it when VLLM_FL_LAYER_PROBE=1.
    enabled = os.environ.get("VLLM_FL_LAYER_PROBE", "0") in ("1", "true", "True")
    if not enabled:
        enabled = os.path.exists(_SENTINEL)
    if not enabled:
        return

    max_steps = int(os.environ.get("VLLM_FL_LAYER_PROBE_STEPS", "0") or 0)
    if max_steps <= 0:
        try:
            with open(_SENTINEL) as fh:
                max_steps = int((fh.read().strip() or "2"))
        except Exception:
            max_steps = 2

    try:
        from vllm.models.glm5next.nvidia import model as _m
    except Exception as e:
        logger.warning("FL: layer probe unavailable: %s", e)
        return

    layer_cls = getattr(_m, "Glm5NextDecoderLayer", None)
    if layer_cls is None:
        for cand in dir(_m):
            if cand.endswith("DecoderLayer"):
                layer_cls = getattr(_m, cand)
                break
    if layer_cls is None:
        logger.warning("FL: layer probe: no DecoderLayer class found")
        return

    if getattr(layer_cls, "_fl_probe_patched", False):
        return

    orig_forward = layer_cls.forward

    def probed_forward(self, positions, hidden_states, residual, post, comb):
        idx = getattr(self, "layer_idx", None)
        if idx is None:
            idx = getattr(self, "_fl_probe_idx", "?")

        step = _PROBE_STATE["step"]
        active = _PROBE_STATE["armed"] and step < max_steps
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            active = active and get_tensor_model_parallel_rank() == 0
        except Exception:
            pass

        if active:
            _emit(
                f"PROBE step={step} layer={idx} IN  "
                f'{_stat("h", hidden_states)} | {_stat("res", residual)} | '
                f'{_stat("post", post)} | {_stat("comb", comb)}'
            )

        out = orig_forward(self, positions, hidden_states, residual, post, comb)

        if active:
            h2, r2, p2, c2 = out
            _emit(
                f"PROBE step={step} layer={idx} OUT "
                f'{_stat("h", h2)} | {_stat("res", r2)} | '
                f'{_stat("post", p2)} | {_stat("comb", c2)}'
            )
        return out

    layer_cls.forward = probed_forward
    layer_cls._fl_probe_patched = True

    # Tag each MLP with its owning layer index so MLPPROBE lines are
    # attributable (Glm5NextMLP itself carries no layer_idx).
    _orig_layer_init = layer_cls.__init__

    def _init_tag(self, *a, **kw):
        _orig_layer_init(self, *a, **kw)
        mlp = getattr(self, "mlp", None)
        if mlp is not None:
            try:
                mlp._fl_layer_idx = getattr(self, "layer_idx", "?")
            except Exception:
                pass

    layer_cls.__init__ = _init_tag

    # Also probe embedding output and the final norm, so a broken prefill can be
    # attributed to the stack rather than to what feeds it.
    text_model_cls = getattr(_m, "Glm5NextModel", None)
    if text_model_cls is not None and not getattr(
        text_model_cls, "_fl_probe_patched", False
    ):
        orig_model_forward = text_model_cls.forward

        def probed_model_forward(self, *args, **kwargs):
            step = _PROBE_STATE["step"]
            armed = _PROBE_STATE["armed"]
            rank0 = True
            try:
                from vllm.distributed import get_tensor_model_parallel_rank

                rank0 = get_tensor_model_parallel_rank() == 0
            except Exception:
                pass
            out = orig_model_forward(self, *args, **kwargs)
            if armed and step < max_steps and rank0:
                _emit(f'PROBE step={step} FINAL {_stat("out", out)}')
            if armed:
                _PROBE_STATE["step"] = step + 1
            return out

        text_model_cls.forward = probed_model_forward
        text_model_cls._fl_probe_patched = True

    _emit(f"PROBE-INSTALLED steps={max_steps} class={layer_cls.__name__}")

    # Arm only after warm-up, so the probe measures a real request rather than
    # the dummy profile run (which also perturbs memory profiling).
    try:
        from vllm_fl.worker.worker import WorkerFL

        if not getattr(WorkerFL, "_fl_probe_arm_patched", False):
            _orig_warmup = WorkerFL.compile_or_warm_up_model

            def _warmup_then_arm(self, *a, **kw):
                r = _orig_warmup(self, *a, **kw)
                arm_probe()
                return r

            WorkerFL.compile_or_warm_up_model = _warmup_then_arm
            WorkerFL._fl_probe_arm_patched = True
    except Exception as e:
        logger.warning("FL: probe auto-arm unavailable (%s); arming now", e)
        arm_probe()

    install_mlp_probe()

def install_mlp_probe() -> None:
    """Probe the dense-MLP internals (layers 0-2, first_k_dense_replace=3).

    Layer 0's output is 29x smaller than a CPU fp32 reference while the
    attention-side residual matches within 16%, so the divergence is inside
    Glm5NextMLP. This records gate_up (pre-activation), the activation output,
    and the post-down_proj result so the failing stage is unambiguous.

    gate/up halves are logged separately: a MergedColumnParallelLinear shard
    mix-up would leave each half individually plausible but pair them wrongly,
    which only shows up when they are compared against each other.
    """
    try:
        from vllm.models.glm5next.nvidia import model as _m
    except Exception as e:
        _emit(f"MLPPROBE-UNAVAILABLE {e}")
        return
    cls = getattr(_m, "Glm5NextMLP", None)
    if cls is None or getattr(cls, "_fl_mlp_probe_patched", False):
        return

    orig = cls.forward

    def probed(self, x):
        idx = getattr(self, "_fl_layer_idx", "?")
        step = _PROBE_STATE["step"]
        active = _PROBE_STATE["armed"] and step < 2
        try:
            from vllm.distributed import get_tensor_model_parallel_rank

            active = active and get_tensor_model_parallel_rank() == 0
        except Exception:
            pass
        if not active:
            return orig(self, x)

        gate_up, _ = self.gate_up_proj(x)
        d = gate_up.shape[-1] // 2
        act = self.act_fn(gate_up)
        out, _ = self.down_proj(act)
        _emit(
            f"MLPPROBE step={step} mlp={idx} "
            f'{_stat("in", x)} | {_stat("gate_up", gate_up)} | '
            f'{_stat("gate_half", gate_up[..., :d])} | '
            f'{_stat("up_half", gate_up[..., d:])} | '
            f'{_stat("act", act)} | {_stat("out", out)} '
            f"| swiglu_limit={getattr(self, 'swiglu_limit', None)} "
            f"| act_fn={type(self.act_fn).__name__} "
            f"| fwd={getattr(getattr(self.act_fn, '_forward_method', None), '__name__', '?')}"
        )
        return out

    cls.forward = probed
    cls._fl_mlp_probe_patched = True
    _emit("MLPPROBE-INSTALLED")
