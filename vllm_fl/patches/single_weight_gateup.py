"""Qualify and install single-resident INT8 Llama gate/up weights."""

from __future__ import annotations

import gc
import hashlib
import os
from pathlib import Path

import torch

from vllm.logger import init_logger

from vllm_fl.ops import prefill_gateup as gateup_op

logger = init_logger("vllm.vllm_fl.single_weight_gateup")

SWITCH = "VLLM_FL_SINGLE_WEIGHT_GATEUP"
QWEIGHT_BUFFER = "fl_single_weight_gateup_qweight"
SCALE_BUFFER = "fl_single_weight_gateup_scale"
SMALL_M_VALUES = (1, 8, 32, 64)

_installed = False
_original_forward = None
_converted_layers = 0


def enabled() -> bool:
    return os.environ.get(SWITCH, "0").strip().lower() in ("1", "true", "yes")


def _switch_was_profile_defaulted() -> bool:
    from vllm_fl import device_profiles

    return device_profiles.was_defaulted(SWITCH)


def _unsupported_reason(config, device) -> str | None:
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        return "requires the CUDA execution path"
    if config.parallel_config.tensor_parallel_size != 1:
        return "requires tensor_parallel_size=1"
    if config.lora_config is not None:
        return "LoRA is enabled"
    model_config = config.model_config
    if getattr(model_config, "dtype", None) != torch.bfloat16:
        return "model dtype is not BF16"
    if (
        getattr(model_config, "quantization", None) is not None
        or getattr(model_config, "quantization_config", None) is not None
        or getattr(model_config, "is_quantized", False)
        or getattr(config, "quant_config", None) is not None
    ):
        return "model/global quantization is configured"
    from vllm import envs

    if envs.VLLM_BATCH_INVARIANT:
        return "batch-invariant mode is enabled"
    return None


def validate_early(config, device) -> None:
    """Reject incompatible explicit settings before any global patch is installed."""
    if not enabled():
        return
    reason = _unsupported_reason(config, device)
    if reason and not _switch_was_profile_defaulted():
        raise ValueError(f"{SWITCH}=1 is unsupported: {reason}")


def _module_types():
    from vllm.model_executor.layers.activation import SiluAndMul
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.llama import LlamaMLP

    return LlamaMLP, SiluAndMul, UnquantizedLinearMethod


def _source_hash(layers: list[int]) -> str:
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__),
        root / "ops" / "prefill_gateup.py",
        root / "patches" / "prefill_gateup.py",
        root / "device_profiles.py",
        root / "optimizations" / "runtime.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(f"{SWITCH}={os.environ.get(SWITCH, '0')};".encode())
    digest.update(repr(sorted(layers)).encode())
    return digest.hexdigest()


def _remove_buffer(module, name: str) -> None:
    buffers = getattr(module, "_buffers", None)
    if buffers is not None:
        buffers.pop(name, None)
    nonpersistent = getattr(module, "_non_persistent_buffers_set", None)
    if nonpersistent is not None:
        nonpersistent.discard(name)


def _cleanup_new_buffers(gate) -> None:
    _remove_buffer(gate, QWEIGHT_BUFFER)
    _remove_buffer(gate, SCALE_BUFFER)


def _layer_index(gate, fallback: int) -> int:
    prefix = getattr(gate, "prefix", "") or ""
    pieces = prefix.split(".")
    try:
        position = pieces.index("layers")
        return int(pieces[position + 1])
    except (ValueError, IndexError, TypeError):
        return fallback


def _targets(model, device):
    LlamaMLP, SiluAndMul, UnquantizedLinearMethod = _module_types()
    result = []
    for fallback_index, module in enumerate(model.modules()):
        if not isinstance(module, LlamaMLP):
            continue
        gate = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        weight = getattr(gate, "weight", None)
        if (
            gate is None
            or down is None
            or type(getattr(gate, "quant_method", None)) is not UnquantizedLinearMethod
            or tuple(getattr(weight, "shape", ())) != (12288, 2048)
            or weight.dtype != torch.bfloat16
            or not weight.is_cuda
            or weight.device != torch.device(device)
            or not weight.is_contiguous()
            or getattr(gate, "bias", None) is not None
            or getattr(gate, "tp_size", 1) != 1
            or not isinstance(getattr(module, "act_fn", None), SiluAndMul)
        ):
            continue
        down_weight = getattr(down, "weight", None)
        if (
            type(getattr(down, "quant_method", None)) is not UnquantizedLinearMethod
            or tuple(getattr(down_weight, "shape", ())) != (2048, 6144)
            or down_weight.dtype != torch.bfloat16
            or not down_weight.is_cuda
            or down_weight.device != torch.device(device)
            or not down_weight.is_contiguous()
            or getattr(down, "bias", None) is not None
            or getattr(down, "tp_size", 1) != 1
        ):
            continue
        result.append(
            {
                "module": module,
                "gate": gate,
                "weight": weight,
                "scale": getattr(gate, "fl_prefill_gateup_weight_scale", None),
                "layer": _layer_index(gate, fallback_index),
            }
        )
    result.sort(key=lambda item: item["layer"])
    return result


def _round_away_from_zero(value: torch.Tensor) -> torch.Tensor:
    return torch.where(value >= 0, torch.floor(value + 0.5), torch.ceil(value - 0.5))


def _silu_mul_from_projection(projection: torch.Tensor) -> torch.Tensor:
    gate, up = projection.to(torch.bfloat16).float().chunk(2, dim=-1)
    return (gate / (1.0 + torch.exp(-gate)) * up).to(torch.bfloat16)


def _cpu_semantic_oracle(
    weight: torch.Tensor, qweight: torch.Tensor, scale: torch.Tensor, layer: int
):
    """Compare sampled INT8 gate/up math with an independent FP32 CPU oracle."""
    n, k = weight.shape
    h = n // 2
    cols = torch.tensor(
        [round(i * (h - 1) / 7) for i in range(8)],
        dtype=torch.int64,
        device=weight.device,
    )
    rows = torch.cat((cols, cols + h))
    w_cpu = weight.index_select(0, rows).contiguous().float().cpu()
    q_cpu = qweight.index_select(0, rows).contiguous().cpu()
    s_cpu = scale.index_select(0, rows).contiguous().cpu()
    exact_scale = w_cpu.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    torch.testing.assert_close(s_cpu, exact_scale, rtol=1e-6, atol=0.0)
    expected_q = torch.round(w_cpu / s_cpu[:, None]).clamp(-127, 127).to(torch.int8)
    torch.testing.assert_close(q_cpu, expected_q, rtol=0, atol=0)

    generator = torch.Generator(device="cpu").manual_seed(20261111 + layer)
    x_cpu = torch.randn((17, k), generator=generator).to(torch.bfloat16)
    x_fp = x_cpu.float()
    sx = x_fp.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    qx = _round_away_from_zero(x_fp / sx[:, None]).clamp(-127, 127).to(torch.int64)

    projection = (qx @ q_cpu.to(torch.int64).T).float() * sx[:, None] * s_cpu[None, :]
    expected = _silu_mul_from_projection(projection)
    if not torch.isfinite(expected).all():
        raise ValueError("CPU INT8 gate/up reference is non-finite")
    return cols, x_cpu, expected


def _candidate_mlp(module, x, qweight, scale):
    rows = x.reshape(-1, x.shape[-1])
    gateup = torch.ops.vllm_fl.prefill_gateup_int8_prequantized_rowmajor(
        rows, qweight, scale
    )
    return module.down_proj(gateup.reshape(*x.shape[:-1], gateup.shape[-1]))[0]


GRAPH_CALLS = 12
GRAPH_ROUNDS = 6


def _capture_checked_graph(call, device):
    eager = [call(index) for index in range(3)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.device(device):
        with torch.cuda.graph(graph):
            outputs = [call(index % 3) for index in range(GRAPH_CALLS)]
        graph.replay()
        graph.replay()
        torch.cuda.synchronize(device)
    for index, output in enumerate(outputs):
        if (
            not torch.isfinite(output).all()
            or not torch.isfinite(eager[index % 3]).all()
        ):
            raise RuntimeError("gate/up startup graph produced non-finite output")
        torch.testing.assert_close(output, eager[index % 3], rtol=2e-2, atol=5e-4)
    return graph, outputs


def _check_graphs(representatives, m, device, *, require_performance):
    from vllm_fl.patches import prefill_gateup

    inputs = []
    for target, _, _ in representatives:
        generator = torch.Generator(device=device).manual_seed(
            20261111 + 101 * target["layer"] + m
        )
        x = torch.randn(
            (m, target["weight"].shape[1]),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        inputs.append((x, x.clone()))

    def baseline(index):
        target, _, _ = representatives[index]
        return _original_forward(target["module"], inputs[index][0])

    def candidate(index):
        target, qweight, scale = representatives[index]
        return _candidate_mlp(target["module"], inputs[index][1], qweight, scale)

    graphs = {
        "baseline": _capture_checked_graph(baseline, device),
        "candidate": _capture_checked_graph(candidate, device),
    }
    if not require_performance:
        return True

    event_us, wall_ms = prefill_gateup._median_graph_us_and_wall(
        graphs,
        device,
        rounds=GRAPH_ROUNDS,
    )
    logger.info(
        "Single-weight gate/up startup M=%d: Graph%d event_us=%s wall_ms=%s",
        m,
        GRAPH_CALLS,
        event_us,
        wall_ms,
    )
    return (
        event_us["candidate"] <= event_us["baseline"]
        and wall_ms["candidate"] <= wall_ms["baseline"]
    )


def _qualify(targets, device, *, require_performance):
    representatives = []
    for index in (0, len(targets) // 2, len(targets) - 1):
        target = targets[index]
        qweight, scale, (cols, x_cpu, expected) = _layer_candidate(target)
        actual = torch.ops.vllm_fl.prefill_gateup_int8_prequantized_rowmajor(
            x_cpu.to(qweight.device), qweight, scale
        ).index_select(1, cols)
        torch.testing.assert_close(
            actual.float(), expected.to(device).float(), rtol=2e-2, atol=5e-4
        )
        representatives.append((target, qweight, scale))

    for m in SMALL_M_VALUES:
        if not _check_graphs(
            representatives, m, device, require_performance=require_performance
        ):
            return False
        gc.collect()
        torch.cuda.empty_cache()
    return True


def _attach_and_remove_weight(
    target, qweight: torch.Tensor, scale: torch.Tensor
) -> tuple[int, int, int]:
    gate = target["gate"]
    weight = target["weight"]
    if gate._parameters.get("weight") is not weight:
        raise RuntimeError("gate/up parameter changed during startup selection")
    had_scale = target["scale"] is not None
    try:
        gate.register_buffer(QWEIGHT_BUFFER, qweight, persistent=False)
        # Roll back even if the second registration fails.
        gate.register_buffer(SCALE_BUFFER, scale, persistent=False)
        if (
            getattr(gate, QWEIGHT_BUFFER) is not qweight
            or getattr(gate, SCALE_BUFFER) is not scale
            or gate._parameters.get("weight") is not weight
        ):
            raise RuntimeError(
                "single-weight gate/up buffers failed their commit check"
            )
    except Exception:
        _cleanup_new_buffers(gate)
        raise

    q_bytes = qweight.numel() * qweight.element_size()
    weight_bytes = weight.numel() * weight.element_size()
    scale_bytes = scale.numel() * scale.element_size()

    # Removing the old aliases is safe before the commit point because the BF16
    # parameter and its original prefill scale are still available for rollback.
    _remove_buffer(gate, "fl_prefill_gateup_weight_scale")

    # This pop is the conversion commit point. No exception is caught after it:
    # a failed post-commit startup must stop service startup, not claim fallback.
    gate._parameters.pop("weight")
    target["weight"] = None
    delta = q_bytes + (0 if had_scale else scale_bytes) - weight_bytes
    del weight
    return q_bytes, scale_bytes, delta


def _install_forward(original):
    def forward(self, x):
        gate = self.gate_up_proj
        qweight = getattr(gate, QWEIGHT_BUFFER, None)
        if qweight is None:
            return original(self, x)
        scale = getattr(gate, SCALE_BUFFER)
        if (
            x.dtype != torch.bfloat16
            or not x.is_cuda
            or x.device != qweight.device
            or x.shape[-1] != qweight.shape[1]
        ):
            raise RuntimeError(
                "selected single-weight gate/up requires CUDA BF16 input with matching hidden size"
            )
        if x.ndim < 2:
            raise RuntimeError(
                "selected single-weight gate/up requires a token dimension"
            )
        return _candidate_mlp(self, x, qweight, scale)

    return forward


def install(config, device) -> bool:
    """Wrap LlamaMLP after prefill's hook so unchanged layers keep that path."""
    global _installed, _original_forward, _converted_layers
    if not enabled():
        return False
    validate_early(config, device)
    reason = _unsupported_reason(config, device)
    if reason:
        if _switch_was_profile_defaulted():
            logger.info(
                "Single-weight gate/up profile skipped; preserving original path: %s",
                reason,
            )
            return False
        raise ValueError(f"{SWITCH}=1 is unsupported: {reason}")
    if _installed:
        return True
    LlamaMLP, _, _ = _module_types()

    _original_forward = LlamaMLP.forward
    LlamaMLP.forward = _install_forward(_original_forward)
    LlamaMLP._fl_single_weight_gateup = True
    _installed = True
    _converted_layers = 0
    logger.info("Installed single-weight INT8 gate/up wrapper")
    return True


def _layer_candidate(target):
    weight = target["weight"]
    scale = target["scale"]
    new_scale = scale if scale is not None else gateup_op.weight_scales(weight)
    if (
        new_scale.dtype != torch.float32
        or tuple(new_scale.shape) != (weight.shape[0],)
        or not new_scale.is_contiguous()
        or new_scale.device != weight.device
        or not torch.isfinite(new_scale).all().item()
    ):
        raise ValueError(
            "single-weight scale violates contiguous FP32 row-scale contract"
        )
    expected_scale = gateup_op.weight_scales(weight)
    if not torch.allclose(new_scale, expected_scale, rtol=1e-6, atol=0.0):
        raise ValueError("prefill row scale differs from the single-weight QW contract")
    qweight = gateup_op.quantize_weight_rowmajor(weight, new_scale)
    if (
        qweight.dtype != torch.int8
        or tuple(qweight.shape) != tuple(weight.shape)
        or not qweight.is_contiguous()
        or qweight.device != weight.device
    ):
        raise ValueError("single-weight QW violates contiguous row-major INT8 contract")
    oracle = _cpu_semantic_oracle(weight, qweight, new_scale, target["layer"])
    return qweight, new_scale, oracle


def finalize(model, device, config, *, model_memory_usage: int = 0) -> dict:
    """Qualify the INT8 path and replace eligible BF16 weights transactionally."""
    global _converted_layers
    stats = dict(layers=0, delta_bytes=0, qweight_bytes=0, scale_bytes=0, bf16_bytes=0)
    if not _installed:
        return stats

    targets = _targets(model, device)
    if not targets:
        logger.warning("Single-weight gate/up disabled: no eligible Llama MLP layers")
        return stats
    try:
        if not _qualify(
            targets,
            torch.device(device),
            require_performance=_switch_was_profile_defaulted(),
        ):
            logger.info(
                "Single-weight gate/up disabled: startup Graph cost gate failed"
            )
            return stats
    except Exception:
        logger.warning(
            "Single-weight gate/up disabled after startup validation", exc_info=True
        )
        return stats

    layers = []
    for target in targets:
        gate = target["gate"]
        weight = target["weight"]
        qweight = None
        try:
            qweight, scale, _ = _layer_candidate(target)
            weight_bytes = weight.numel() * weight.element_size()
            added_scale_bytes = (
                scale.numel() * scale.element_size() if target["scale"] is None else 0
            )
            delta = qweight.numel() + added_scale_bytes - weight_bytes
            if (
                model_memory_usage > 0
                and model_memory_usage + stats["delta_bytes"] + delta < 0
            ):
                raise ValueError("conversion would make model_memory_usage negative")
            q_bytes, scale_bytes, delta = _attach_and_remove_weight(
                target, qweight, scale
            )
        except Exception:
            if "weight" not in gate._parameters:
                raise
            if qweight is not None and getattr(gate, QWEIGHT_BUFFER, None) is qweight:
                _cleanup_new_buffers(gate)
            logger.warning(
                "Single-weight gate/up left layer %s unchanged",
                target["layer"],
                exc_info=True,
            )
            continue
        stats["layers"] += 1
        stats["qweight_bytes"] += q_bytes
        stats["scale_bytes"] += scale_bytes
        stats["bf16_bytes"] += weight_bytes
        stats["delta_bytes"] += delta
        layers.append(target["layer"])
        _converted_layers = len(layers)
        del weight, qweight

    gc.collect()
    torch.cuda.empty_cache()
    config.additional_config["fl_single_weight_gateup_layers"] = len(layers)
    config.additional_config["fl_single_weight_gateup_hash"] = (
        _source_hash(layers) if layers else "disabled-no-layers"
    )
    if layers:
        logger.info(
            "Single-weight gate/up selected: layers=%d small_m=int8 large_m=int8",
            len(layers),
        )
    else:
        logger.warning(
            "Single-weight gate/up left all layers unchanged after validation"
        )
    return stats


def has_converted_layers() -> bool:
    return _converted_layers > 0
