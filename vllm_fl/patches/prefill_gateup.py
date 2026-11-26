"""Opt-in on-demand INT8 gate/up for large BF16 prefill batches."""

import gc
import hashlib
import inspect
import os
import statistics
import time
from pathlib import Path

import torch

from vllm.logger import init_logger

import vllm_fl.envs as fl_envs
from vllm_fl.ops import prefill_gateup as gateup_op

logger = init_logger("vllm.vllm_fl.prefill_gateup")
SW_BUFFER = "fl_prefill_gateup_weight_scale"
SWITCH = "VLLM_FL_PREFILL_GATEUP_INT8"
MIN_PREFILL_TOKENS = 2048
MAX_NRMSE = 0.03
MIN_SPEEDUP = 0.10

_installed = False
_gate_result: bool | None = None
_scale_layers = 0
_scale_bytes = 0
_minimum_tokens = MIN_PREFILL_TOKENS
_orig_apply = None
_weight_layout = "packed"


def _unsupported_reason(config, device) -> str | None:
    if torch.device(device).type != "cuda" or torch.version.hip is not None:
        return "requires the CUDA execution path"
    if config.parallel_config.tensor_parallel_size != 1:
        return "requires tensor_parallel_size=1"
    if config.lora_config is not None:
        return "LoRA is enabled"
    if getattr(config.model_config, "dtype", None) != torch.bfloat16:
        return "model dtype is not BF16"
    from vllm import envs

    if envs.VLLM_BATCH_INVARIANT:
        return "batch-invariant mode is enabled"
    return None


def _token_threshold(config) -> int:
    max_num_seqs = getattr(getattr(config, "scheduler_config", None), "max_num_seqs", 0)
    return max(MIN_PREFILL_TOKENS, int(max_num_seqs or 0) + 1)


def _eligible_weight(layer, device, linear_method_type) -> bool:
    weight = getattr(layer, "weight", None)
    return bool(
        (getattr(layer, "prefix", "") or "").endswith("gate_up_proj")
        and type(getattr(layer, "quant_method", None)) is linear_method_type
        and tuple(getattr(weight, "shape", ())) == (12288, 2048)
        and getattr(layer, "bias", None) is None
        and getattr(layer, "tp_size", 1) == 1
        and weight.dtype == torch.bfloat16
        and weight.is_cuda
        and weight.device == torch.device(device)
        and weight.is_contiguous()
    )


def _can_dispatch(layer, x, device, linear_method_type) -> bool:
    return bool(
        _eligible_weight(layer, device, linear_method_type)
        and x.ndim == 2
        and x.is_cuda
        and x.dtype == torch.bfloat16
        and x.stride(1) == 1
        and x.device == layer.weight.device
    )


def _source_hash() -> str:
    root = Path(__file__).parents[1]
    paths = (
        Path(__file__),
        root / "ops" / "prefill_gateup.py",
        root / "device_profiles.py",
        root / "optimizations" / "runtime.py",
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(f"{SWITCH}={os.environ.get(SWITCH, 'auto')}".encode())
    digest.update(f"minimum_tokens={_minimum_tokens}".encode())
    digest.update(f"weight_layout={_weight_layout}".encode())
    return digest.hexdigest()


def _baseline_gateup(orig_apply, method, layer, x, activation):
    projection = orig_apply(method, layer, x)
    return activation(projection)


def _round_away_from_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, torch.floor(x + 0.5), torch.ceil(x - 0.5))


def _cpu_numerical_selfcheck(
    weight: torch.Tensor, scale: torch.Tensor, layout: str = "packed"
) -> None:
    """Check the small-M quantize/dot/epilogue path against independent CPU math."""
    if layout not in ("packed", "rowmajor"):
        raise ValueError(f"unsupported gate/up weight layout: {layout}")
    n, k = weight.shape
    h = n // 2
    cols = torch.tensor(
        [round(i * (h - 1) / 7) for i in range(8)],
        dtype=torch.int64,
        device=weight.device,
    )
    rows = torch.cat((cols, cols + h))
    sample_weight = weight.index_select(0, rows).contiguous()
    sample_scale = scale.index_select(0, rows).contiguous()
    w_cpu = sample_weight.float().cpu()
    sw_cpu = sample_scale.cpu()
    cpu_scale = w_cpu.abs().amax(dim=1).clamp_min(1e-8) / 127.0
    torch.testing.assert_close(sw_cpu, cpu_scale, rtol=1e-6, atol=0.0)
    qw_cpu = torch.round(w_cpu / sw_cpu[:, None]).clamp(-127, 127).to(torch.int8)

    max_error = 0.0
    for m in (17, 32):
        generator = torch.Generator(device="cpu").manual_seed(20261108 + m)
        x_cpu = torch.randn((m, k), generator=generator).to(torch.bfloat16)
        x_gpu = x_cpu.to(device=weight.device)
        q_gpu, sx_gpu = gateup_op.quantize_activation_padded(x_gpu)
        x_fp = x_cpu.float()
        sx_cpu = x_fp.abs().amax(dim=1).clamp_min(1e-8) / 127.0
        q_cpu = (
            _round_away_from_zero(x_fp / sx_cpu[:, None])
            .clamp(-127, 127)
            .to(torch.int8)
        )
        torch.testing.assert_close(q_gpu[:m].cpu(), q_cpu, rtol=0, atol=0)
        torch.testing.assert_close(sx_gpu[:m].cpu(), sx_cpu, rtol=1e-6, atol=0.0)
        assert torch.count_nonzero(q_gpu[m:]).item() == 0
        assert torch.count_nonzero(sx_gpu[m:]).item() == 0

        if layout == "rowmajor":
            qw_gpu = gateup_op.quantize_weight_rowmajor(sample_weight, sample_scale)
            torch.testing.assert_close(qw_gpu.cpu(), qw_cpu, rtol=0, atol=0)
            actual = gateup_op.prefill_gateup_int8_rowmajor(
                x_gpu, sample_weight, sample_scale
            ).cpu()
        else:
            qw_gpu = gateup_op.pack_weight_transpose(sample_weight, sample_scale)
            torch.testing.assert_close(
                qw_gpu.cpu(), qw_cpu.t().contiguous(), rtol=0, atol=0
            )
            actual = gateup_op.prefill_gateup_int8(
                x_gpu, sample_weight, sample_scale
            ).cpu()

        accum = q_cpu.to(torch.int32) @ qw_cpu.to(torch.int32).t()
        projection = (
            (accum.float() * sx_cpu[:, None] * sw_cpu[None, :])
            .to(torch.bfloat16)
            .float()
        )
        h_sample = rows.numel() // 2
        expected_int8 = (
            torch.nn.functional.silu(projection[:, :h_sample])
            * projection[:, h_sample:]
        ).to(torch.bfloat16)
        torch.testing.assert_close(actual, expected_int8, rtol=0, atol=0.015625)

        fp_projection = (x_fp @ w_cpu.t()).to(torch.bfloat16).float()
        expected_fp = (
            torch.nn.functional.silu(fp_projection[:, :h_sample])
            * fp_projection[:, h_sample:]
        ).to(torch.bfloat16)
        error = _nrmse(actual, expected_fp)
        if not torch.isfinite(torch.tensor(error)) or error > MAX_NRMSE:
            raise ValueError(f"CPU reference NRMSE {error:.6f} exceeds {MAX_NRMSE:.3f}")
        max_error = max(max_error, error)
    logger.info(
        "Prefill gate/up INT8 CPU selfcheck PASS: layout=%s M=17,32 "
        "sampled_gateup=8+8 max_nrmse=%.5f",
        layout,
        max_error,
    )


def _nrmse(actual: torch.Tensor, reference: torch.Tensor) -> float:
    diff = actual.float() - reference.float()
    ref = reference.float()
    return (
        diff.square().mean().sqrt() / ref.square().mean().sqrt().clamp_min(1e-12)
    ).item()


def _capture(fn, device):
    fn()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    torch.cuda.synchronize(device)
    return graph, output


def _median_graph_us(graphs, device) -> dict[str, float]:
    samples = {name: [] for name in graphs}
    for _ in range(3):
        for name in ("baseline", "candidate", "candidate", "baseline"):
            graph, _ = graphs[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000.0)
    return {name: statistics.median(values) for name, values in samples.items()}


def _median_graph_us_and_wall(graphs, device, rounds=3):
    """Measure each graph with events and host wall around replay plus sync."""
    event_samples = {name: [] for name in graphs}
    wall_samples = {name: [] for name in graphs}
    order = ("baseline", "candidate", "candidate", "baseline")
    for _ in range(rounds):
        for name in order:
            graph, _ = graphs[name]
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            host_start = time.perf_counter_ns()
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            wall_samples[name].append((time.perf_counter_ns() - host_start) / 1e6)
            event_samples[name].append(start.elapsed_time(end) * 1000.0)
    return (
        {name: statistics.median(values) for name, values in event_samples.items()},
        {name: statistics.median(values) for name, values in wall_samples.items()},
    )


def _rowmajor_targets(model, device):
    """Collect three real gate/up and down weight pairs without copying them."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    targets = []
    for module in model.modules():
        gate = getattr(module, "gate_up_proj", None)
        down = getattr(module, "down_proj", None)
        scale = getattr(gate, SW_BUFFER, None)
        if (
            gate is None
            or down is None
            or scale is None
            or not _eligible_weight(gate, device, UnquantizedLinearMethod)
        ):
            continue
        down_weight = getattr(down, "weight", None)
        if (
            tuple(getattr(down_weight, "shape", ())) != (2048, 6144)
            or down_weight.dtype != torch.bfloat16
            or not down_weight.is_cuda
            or down_weight.device != torch.device(device)
            or not down_weight.is_contiguous()
            or type(getattr(down, "quant_method", None)) is not UnquantizedLinearMethod
            or getattr(down, "tp_size", 1) != 1
            or getattr(down, "bias", None) is not None
        ):
            raise ValueError(
                "row-major startup requires contiguous BF16 6144x2048 down weights"
            )
        activation_module = getattr(module, "act_fn", None)
        activation = _native_activation(activation_module)
        targets.append((gate, scale, down_weight, activation))
        if len(targets) == 3:
            break
    if len(targets) != 3:
        raise RuntimeError(
            f"row-major startup found {len(targets)} eligible MLP layers; requires 3"
        )
    return targets


def _native_activation(activation_module):
    from vllm.model_executor.layers.activation import SiluAndMul

    if not isinstance(activation_module, SiluAndMul):
        raise TypeError("target MLP activation is not vLLM SiluAndMul")
    activation = getattr(activation_module, "_forward_method", None)
    if not callable(activation):
        raise TypeError("target SiluAndMul has no callable _forward_method")
    forward_oot = getattr(activation_module, "forward_oot", None)
    if callable(forward_oot) and activation == forward_oot:
        raise RuntimeError("target _forward_method is the OOT dispatcher, not native")
    return activation


def _registered_mm_callable(flag_gems):
    """Read the active worker registrar without changing its configuration."""
    registrar = getattr(flag_gems, "current_work_registrar", None)
    if registrar is None:
        raise RuntimeError("FlagGems worker registrar is not active")
    get_all_keys = getattr(registrar, "get_all_keys", None)
    if not callable(get_all_keys):
        raise RuntimeError("FlagGems worker registrar cannot report registered keys")
    config = getattr(registrar, "config", None)
    if config is None:
        raise RuntimeError("FlagGems worker registrar has no active config")
    entries = [entry for entry in config if len(entry) >= 2 and entry[0] == "mm"]
    if "mm" not in get_all_keys() or len(entries) != 1 or not callable(entries[0][1]):
        raise RuntimeError(
            "FlagGems worker registrar has no unique registered aten::mm"
        )
    registered_mm = entries[0][1]
    exported_mm = getattr(flag_gems, "mm", None)
    if not callable(exported_mm):
        raise RuntimeError("FlagGems exported mm callable is unavailable")
    identity_match = registered_mm is exported_mm
    try:
        unwrapped_match = inspect.unwrap(registered_mm) is inspect.unwrap(exported_mm)
    except ValueError:
        unwrapped_match = False
    if not (identity_match or unwrapped_match):
        raise RuntimeError(
            "active registered aten::mm is not the exported FlagGems mm callable"
        )
    return registrar, registered_mm


def _run_rowmajor_startup_gate(model, device) -> bool:
    """Measure row-major INT8 against the active FlagGems mm MLP route."""
    import flag_gems

    device = torch.device(device)
    if _minimum_tokens > 2048:
        raise RuntimeError(
            f"M=2048 anchor is below configured INT8 threshold {_minimum_tokens}"
        )
    targets = _rowmajor_targets(model, device)
    _registered_mm_callable(flag_gems)

    generator = torch.Generator(device=device)
    inputs = []
    for index in range(len(targets)):
        generator.manual_seed(20261109 + index)
        baseline_x = torch.randn(
            (2048, 2048), device=device, dtype=torch.bfloat16, generator=generator
        )
        # Distinct buffers keep the two graph inputs independently resident while
        # preserving identical values for the A/B comparison.
        candidate_x = baseline_x.clone()
        inputs.append((baseline_x, candidate_x))

    def baseline_all():
        outputs = []
        for (gate, _scale, down_weight, activation), (x, _candidate_x) in zip(
            targets, inputs
        ):
            projection = torch.ops.aten.mm.default(x, gate.weight.t())
            hidden = activation(projection)
            output = torch.ops.aten.mm.default(hidden, down_weight.t())
            outputs.append((hidden, output))
        return outputs

    def candidate_all():
        outputs = []
        for (gate, scale, down_weight, _activation), (_baseline_x, x) in zip(
            targets, inputs
        ):
            hidden = torch.ops.vllm_fl.prefill_gateup_int8_rowmajor(
                x, gate.weight, scale, _minimum_tokens
            )
            output = torch.ops.aten.mm.default(hidden, down_weight.t())
            outputs.append((hidden, output))
        return outputs

    passed = True
    max_gate_error = 0.0
    max_mlp_error = 0.0
    with torch.inference_mode():
        expected = baseline_all()
        actual = candidate_all()
        torch.cuda.synchronize(device)

        for index, (
            (base_gate, base_mlp),
            (candidate_gate, candidate_mlp),
        ) in enumerate(zip(expected, actual)):
            finite = bool(
                torch.isfinite(base_gate).all()
                and torch.isfinite(base_mlp).all()
                and torch.isfinite(candidate_gate).all()
                and torch.isfinite(candidate_mlp).all()
            )
            gate_error = _nrmse(candidate_gate, base_gate) if finite else float("inf")
            mlp_error = _nrmse(candidate_mlp, base_mlp) if finite else float("inf")
            passed &= finite and gate_error <= MAX_NRMSE and mlp_error <= MAX_NRMSE
            max_gate_error = max(max_gate_error, gate_error)
            max_mlp_error = max(max_mlp_error, mlp_error)
            logger.info(
                "Prefill gate/up INT8 row-major correctness layer=%d M=2048 "
                "gateup_nrmse=%.5f mlp_nrmse=%.5f finite=%s",
                index,
                gate_error,
                mlp_error,
                finite,
            )

        del expected, actual
        graphs = {
            "baseline": _capture(baseline_all, device),
            "candidate": _capture(candidate_all, device),
        }
        for graph, _output in graphs.values():
            graph.replay()
            graph.replay()
        torch.cuda.synchronize(device)
        event_us = _median_graph_us(graphs, device)
        speedup = 1.0 - event_us["candidate"] / event_us["baseline"]
        passed &= speedup >= MIN_SPEEDUP
        logger.info(
            "Prefill gate/up INT8 row-major full-MLP startup gate M=2048 layers=%d: "
            "baseline_us=%.3f candidate_us=%.3f speedup=%.1f%% "
            "max_gateup_nrmse=%.5f max_mlp_nrmse=%.5f",
            len(targets),
            event_us["baseline"],
            event_us["candidate"],
            100.0 * speedup,
            max_gate_error,
            max_mlp_error,
        )
        del graphs
    return passed


def _startup_gate_rowmajor(model, device) -> bool:
    """Run the row-major gate and release all scratch before KV allocation."""
    try:
        return _run_rowmajor_startup_gate(model, device)
    finally:
        # The helper owns all temporary inputs, outputs, and CUDA graphs. Its
        # frame is gone before collection, so cached scratch does not compete
        # with the model's later KV-cache sizing.
        gc.collect()
        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()


def _startup_gate(orig_apply, method, layer, scale, activation) -> bool:
    import flag_gems

    device = layer.weight.device
    registrar = getattr(flag_gems, "current_work_registrar", None)
    registered_keys = registrar.get_all_keys() if registrar is not None else []
    logger.info(
        "Prefill gate/up INT8 BF16 comparator: UnquantizedLinearMethod.apply -> "
        "torch.nn.functional.linear; native_activation_callable=%s FlagGems=%s backend=%s "
        "aten_linear_registered=%s",
        type(activation).__name__,
        getattr(flag_gems, "__version__", "unknown"),
        getattr(getattr(flag_gems, "runtime", None), "device", None),
        "linear" in registered_keys,
    )
    if "linear" not in registered_keys:
        raise RuntimeError(
            "FlagGems aten::linear is not registered for the BF16 baseline"
        )
    weight = layer.weight
    passed = True
    max_error = 0.0
    with torch.inference_mode():
        m = 2048
        generator = torch.Generator(device=device).manual_seed(20261108 + m)
        x = torch.randn(
            (m, weight.shape[1]),
            device=device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        baseline = lambda: _baseline_gateup(orig_apply, method, layer, x, activation)
        candidate = lambda: torch.ops.vllm_fl.prefill_gateup_int8(
            x, weight, scale, _minimum_tokens
        )

        expected = baseline()
        actual = candidate()
        torch.cuda.synchronize(device)
        finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
        error = _nrmse(actual, expected) if finite else float("inf")
        graphs = {
            "baseline": _capture(baseline, device),
            "candidate": _capture(candidate, device),
        }
        timing = _median_graph_us(graphs, device)
        speedup = 1.0 - timing["candidate"] / timing["baseline"]
        passed &= finite and error <= MAX_NRMSE and speedup >= MIN_SPEEDUP
        max_error = max(max_error, error)
        logger.info(
            "Prefill gate/up INT8 startup gate M=%d: baseline_us=%.3f "
            "candidate_us=%.3f speedup=%.1f%% nrmse=%.5f finite=%s",
            m,
            timing["baseline"],
            timing["candidate"],
            100.0 * speedup,
            error,
            finite,
        )
        baseline = candidate = graphs = expected = actual = x = None
        torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()

    if passed:
        logger.info(
            "Prefill gate/up INT8 startup gate PASS: qualified prefill anchor M=2048 >=10%% faster; max_nrmse=%.5f",
            max_error,
        )
    return passed


def install(config, device) -> bool:
    """Install the narrow layer hook when the device profile or user enables it."""
    global _installed, _gate_result, _minimum_tokens, _orig_apply
    global _scale_layers, _scale_bytes
    if not fl_envs.switch(
        SWITCH, auto=False, reason="enabled only by a verified device profile"
    ):
        return False
    reason = _unsupported_reason(config, device)
    if reason:
        logger.warning("Prefill gate/up INT8 DISABLED: %s", reason)
        return False

    from vllm_fl.device_profiles import prefill_gateup_weight_layout

    global _weight_layout
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _weight_layout = prefill_gateup_weight_layout(device_index)
    if _weight_layout not in ("packed", "rowmajor"):
        logger.warning(
            "Prefill gate/up INT8 DISABLED: unsupported weight layout %r",
            _weight_layout,
        )
        return False

    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.models.llama import LlamaMLP

    if getattr(LlamaMLP, "_fl_prefill_gateup_int8", False):
        return True

    orig_post = UnquantizedLinearMethod.process_weights_after_loading
    orig_apply = UnquantizedLinearMethod.apply
    orig_forward = LlamaMLP.forward
    _minimum_tokens = _token_threshold(config)
    _gate_result = None
    _scale_layers = 0
    _scale_bytes = 0
    _orig_apply = orig_apply
    gateup_op.bind_small_m_activation(None)

    def post(self, layer):
        global _gate_result, _scale_layers, _scale_bytes
        orig_post(self, layer)
        if not _eligible_weight(layer, device, UnquantizedLinearMethod):
            return
        if _gate_result is False:
            return
        try:
            scale = gateup_op.weight_scales(layer.weight)
            if not bool(torch.isfinite(scale).all()):
                raise ValueError("non-finite gate/up weight scale")
            existing_scale = getattr(layer, SW_BUFFER, None)
            if existing_scale is not None:
                existing_scale.copy_(scale)
                return
            layer.register_buffer(SW_BUFFER, scale, persistent=False)
            _scale_layers += 1
            _scale_bytes += scale.numel() * scale.element_size()
        except Exception as exc:
            _gate_result = False
            logger.warning(
                "Prefill gate/up INT8 DISABLED: startup preparation failed (%s)",
                f"{type(exc).__name__}: {exc}",
                exc_info=True,
            )

    def forward(self, x):
        proj = self.gate_up_proj
        if _gate_result is True and _can_dispatch(
            proj, x, device, UnquantizedLinearMethod
        ):
            scale = getattr(proj, SW_BUFFER, None)
            if scale is not None:
                if _weight_layout == "rowmajor":
                    z = torch.ops.vllm_fl.prefill_gateup_int8_rowmajor(
                        x, proj.weight, scale, _minimum_tokens
                    )
                else:
                    z = torch.ops.vllm_fl.prefill_gateup_int8(
                        x, proj.weight, scale, _minimum_tokens
                    )
                return self.down_proj(z)[0]
        return orig_forward(self, x)

    UnquantizedLinearMethod.process_weights_after_loading = post
    LlamaMLP.forward = forward
    LlamaMLP._fl_prefill_gateup_int8 = True
    config.additional_config["fl_prefill_gateup_int8_layout"] = _weight_layout
    config.additional_config["fl_prefill_gateup_int8_hash"] = _source_hash()
    _installed = True
    logger.info(
        "Installed opt-in prefill gate/up INT8 hook, hash=%s, min_tokens=%d",
        config.additional_config["fl_prefill_gateup_int8_hash"],
        _minimum_tokens,
    )
    return True


def _prepare_small_m_activation(model):
    """Bind and warm one target MLP's existing compiled native activation."""
    target = None
    for module in model.modules():
        projection = getattr(module, "gate_up_proj", None)
        if getattr(projection, SW_BUFFER, None) is not None:
            target = module
            break
    if target is None:
        raise RuntimeError("no MLP with prefill gate/up scales was found")

    activation_module = getattr(target, "act_fn", None)
    activation = _native_activation(activation_module)

    hidden = target.gate_up_proj.weight.shape[0] // 2
    device = target.gate_up_proj.weight.device
    gateup_op.bind_small_m_activation(activation)
    try:
        with torch.inference_mode():
            for m in (64, 1):
                sample = torch.zeros(
                    (m, 2 * hidden), device=device, dtype=torch.bfloat16
                )
                torch._dynamo.mark_static(sample, 1)
                output = activation(sample)
                if (
                    tuple(output.shape) != (m, hidden)
                    or output.dtype != torch.bfloat16
                    or output.device != device
                    or not bool(torch.isfinite(output).all())
                ):
                    raise ValueError(f"invalid native activation output at M={m}")
                del sample, output
        torch.cuda.synchronize(device)
    except Exception:
        gateup_op.bind_small_m_activation(None)
        raise

    logger.info(
        "Prefill gate/up BF16 fallback activation ready: %s",
        type(activation_module).__name__,
    )
    return target, activation


def finalize(model) -> None:
    """Bind the native activation, run startup gates, and report readiness."""
    global _gate_result
    if not _installed:
        return
    if _gate_result is False:
        gateup_op.bind_small_m_activation(None)
        logger.warning("Prefill gate/up INT8 DISABLED: no passing startup gate")
        return
    try:
        target, activation = _prepare_small_m_activation(model)
        if _orig_apply is None:
            raise RuntimeError("original UnquantizedLinearMethod.apply is unavailable")
        method = target.gate_up_proj.quant_method
        _cpu_numerical_selfcheck(
            target.gate_up_proj.weight,
            getattr(target.gate_up_proj, SW_BUFFER),
            layout=_weight_layout,
        )
        if _weight_layout == "rowmajor":
            _gate_result = _startup_gate_rowmajor(
                model, target.gate_up_proj.weight.device
            )
        else:
            _gate_result = _startup_gate(
                _orig_apply,
                method,
                target.gate_up_proj,
                getattr(target.gate_up_proj, SW_BUFFER),
                activation,
            )
        if not _gate_result:
            gateup_op.bind_small_m_activation(None)
            logger.warning(
                "Prefill gate/up INT8 DISABLED: qualified prefill anchor M=2048 did not meet speed/numerical gate"
            )
            return
        gateup_op.enable_runtime_dispatch_logging()
        logger.info(
            "Prefill gate/up INT8 ready: layers=%d scale_bytes=%d min_tokens=%d mode=on_demand",
            _scale_layers,
            _scale_bytes,
            _minimum_tokens,
        )
    except Exception as exc:
        _gate_result = False
        gateup_op.bind_small_m_activation(None)
        logger.warning(
            "Prefill gate/up INT8 DISABLED: startup preparation failed (%s)",
            f"{type(exc).__name__}: {exc}",
            exc_info=True,
        )
