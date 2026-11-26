"""Opt-in FP64 weighted-boundary top-p selection.

The Triton implementation is imported only after the feature switch and CUDA
device profile have been checked, keeping unsupported-device startup light.
"""

import logging
import math
import statistics
import time

import torch

from vllm_fl import device_profiles, envs

logger = logging.getLogger("vllm.vllm_fl.topp_select")
_SWITCH = "VLLM_FL_TOPP_SELECT"
_MIN_VOCAB = 257
_MAX_VOCAB = 262144
_MAX_BATCH = 64
_TV_LIMIT = 1e-6
_PERF_RATIO_LIMIT = 0.9

_INSTALLED = False
_WRAPPER = None


def _ordered_keys(values: torch.Tensor) -> torch.Tensor:
    bits = values.contiguous().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    negative = (bits & 0x80000000) != 0
    return torch.where(negative, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)


def _cpu_reference(logits: torch.Tensor, p: torch.Tensor):
    """Independent FP64 stable-key reference; returns output, keep, and probs."""
    x = logits.detach().to(device="cpu", dtype=torch.float32).contiguous()
    ps = p.detach().to(device="cpu", dtype=torch.float32).contiguous()
    batch, vocab = x.shape
    output = x.clone()
    keep = torch.ones((batch, vocab), dtype=torch.bool)
    probabilities = torch.zeros((batch, vocab), dtype=torch.float64)
    cdf_before = torch.full((batch, vocab), float("nan"), dtype=torch.float64)
    cdf_after = torch.full_like(cdf_before, float("nan"))
    keys = _ordered_keys(x)

    for row in range(batch):
        values = x[row]
        p_value = float(ps[row])
        invalid = bool(
            torch.isnan(values).any()
            or torch.isposinf(values).any()
            or torch.isneginf(values).all()
            or math.isnan(p_value)
        )
        if invalid or p_value > 1.0:
            continue

        finite = torch.isfinite(values)
        max_value = values[finite].to(torch.float64).max()
        weights = torch.zeros(vocab, dtype=torch.float64)
        weights[finite] = torch.exp(values[finite].to(torch.float64) - max_value)
        z = weights.sum(dtype=torch.float64)
        if z > 0:
            probabilities[row] = weights / z

        order = torch.argsort(keys[row], stable=True)
        ordered_probs = probabilities[row].index_select(0, order)
        after = torch.cumsum(ordered_probs, dim=0)
        before = after - ordered_probs
        cdf_before[row, order] = before
        cdf_after[row, order] = after
        if p_value <= 0.0:
            maximum_key = keys[row, finite].max()
            candidates = torch.where(keys[row] == maximum_key)[0]
            keep[row] = False
            keep[row, candidates[-1]] = True
        else:
            alpha = 1.0 - p_value
            remove = after <= alpha
            remove[-1] = False
            keep[row, order[remove]] = False
        output[row, ~keep[row]] = -float("inf")
    return output, keep, probabilities, cdf_before, cdf_after


def _keep_from_output(source: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
    # An original -inf and a masked -inf are indistinguishable; keep it in the
    # semantic mask because its probability is zero in the variation check.
    source_bits = source.contiguous().view(torch.int32)
    output_bits = output.contiguous().view(torch.int32)
    return (source_bits == output_bits) | torch.isneginf(source)


def _budget_check(
    source, actual, expected, probabilities, cdf_before, cdf_after, p_values
):
    source = source.detach().to(device="cpu").contiguous()
    actual = actual.detach().to(device="cpu").contiguous()
    p_values = p_values.detach().to(device="cpu", dtype=torch.float32)
    source_bits = source.view(torch.int32)
    actual_bits = actual.view(torch.int32)
    expected_bits = expected.contiguous().view(torch.int32)
    expected_keep = _keep_from_output(source, expected)
    neg_inf_bits = (
        torch.tensor(-float("inf"), dtype=torch.float32).view(torch.int32).item()
    )
    allowed = (actual_bits == source_bits) | (actual_bits == neg_inf_bits)
    if not bool(allowed.all()):
        return False, float("inf"), "output changed a retained logit"

    actual_keep = _keep_from_output(source, actual)
    if not bool(actual_keep.any(dim=-1).all()):
        return False, float("inf"), "a row retained no token"
    max_tv = 0.0
    for row in range(source.shape[0]):
        p_value = float(p_values[row])
        row_actual = actual_bits[row]
        row_expected = expected_bits[row]
        logits = source[row]
        invalid = bool(
            torch.isnan(logits).any()
            or torch.isposinf(logits).any()
            or torch.isneginf(logits).all()
            or math.isnan(p_value)
        )
        if invalid or p_value <= 0.0 or p_value > 1.0:
            if not torch.equal(row_actual, row_expected):
                return False, float("inf"), "special-value or p-boundary row differed"
            continue

        expected_row_keep = expected_keep[row].to(device="cpu", dtype=torch.bool)
        actual_row_keep = actual_keep[row].to(device="cpu", dtype=torch.bool)
        probs = probabilities[row].to(device="cpu", dtype=torch.float64)
        left = probs * expected_row_keep
        right = probs * actual_row_keep
        left_sum = left.sum()
        right_sum = right.sum()
        if left_sum > 0 and right_sum == 0:
            return False, float("inf"), "candidate kept zero reference probability mass"
        tv = 0.0
        if left_sum > 0 and right_sum > 0:
            tv = 0.5 * ((left / left_sum) - (right / right_sum)).abs().sum().item()
        max_tv = max(max_tv, tv)
        if tv > _TV_LIMIT:
            return False, max_tv, "row total variation exceeded budget"

        differing = torch.where(expected_row_keep != actual_row_keep)[0]
        if differing.numel():
            before = cdf_before[row].index_select(0, differing)
            after = cdf_after[row].index_select(0, differing)
            alpha = 1.0 - p_value
            distance = torch.minimum((before - alpha).abs(), (after - alpha).abs())
            if not bool(torch.isfinite(distance).all()) or bool(
                (distance > _TV_LIMIT).any()
            ):
                return (
                    False,
                    max_tv,
                    "mask difference was outside the FP64 CDF boundary budget",
                )
    return True, max_tv, "ok"


def _bits_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
    return torch.equal(
        left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)
    )


def _validate_small_cases(kernel_apply, device: torch.device) -> bool:
    generator = torch.Generator(device="cpu").manual_seed(8731109)
    cases = []
    cases.append(
        (
            torch.randn((5, 257), generator=generator, dtype=torch.float32),
            torch.tensor([0.0, 0.5, 0.95, 1.0, 1.2], dtype=torch.float32),
        )
    )
    tie = torch.tensor([-4.0, -2.0, -1.0, -0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.0])
    index = torch.arange(257)
    cases.append(
        (
            torch.stack([tie[(index + row * 3) % tie.numel()] for row in range(5)]),
            torch.tensor([0.0, 0.5, 0.95, 1.0, 1.2], dtype=torch.float32),
        )
    )
    extreme = torch.full((6, 1025), -1000.0, dtype=torch.float32)
    extreme[0, 0] = 1000.0
    extreme[0, 1:4] = torch.tensor([999.0, 999.0, 998.0])
    extreme[1, :] = -float("inf")
    extreme[2, 0] = float("nan")
    extreme[2, 1] = 2.0
    extreme[3, 5] = float("inf")
    extreme[3, 6] = 1.0
    extreme[4, :] = torch.arange(1025, dtype=torch.float32) % 7
    extreme[4, 0] = -float("inf")
    extreme[4, 1] = -0.0
    extreme[4, 2] = 0.0
    extreme[5, :] = torch.linspace(-8.0, 8.0, 1025)
    cases.append(
        (
            extreme,
            torch.tensor(
                [0.95, 0.95, 0.5, 0.95, 1.0, float("nan")], dtype=torch.float32
            ),
        )
    )

    for source_cpu, p_cpu in cases:
        expected, _, _, _, _ = _cpu_reference(source_cpu, p_cpu)
        source = source_cpu.to(device)
        p = p_cpu.to(device)
        eager = kernel_apply(source, p)
        torch.cuda.synchronize(device)
        equal = _bits_equal(eager.cpu(), expected)
        if not equal:
            logger.warning("top-p small CPU reference output was not bit exact")
            return False
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = kernel_apply(source, p)
        graph.replay()
        torch.cuda.synchronize(device)
        if not _bits_equal(eager, graph_output):
            logger.warning("top-p eager/graph output mismatch in small case")
            return False
    return True


def _make_inputs(batch: int, vocab: int, device: torch.device):
    inputs = []
    ps = []
    for rotation in range(3):
        generator = torch.Generator(device="cpu").manual_seed(
            611020 + batch * 7 + rotation
        )
        logits = torch.randn(
            (batch, vocab), generator=generator, dtype=torch.bfloat16
        ).float()
        inputs.append(logits.to(device))
        ps.append(torch.full((batch,), 0.95, dtype=torch.float32, device=device))
    return inputs, ps


def _validate_real_inputs(
    kernel_apply, original, batch: int, inputs, ps, device
) -> bool:
    for rotation, (source, p) in enumerate(zip(inputs, ps)):
        candidate = kernel_apply(source, p)
        torch.cuda.synchronize(device)
        expected, _, probs, cdf_before, cdf_after = _cpu_reference(source, p)
        budget_ok, tv, reason = _budget_check(
            source, candidate, expected, probs, cdf_before, cdf_after, p
        )
        if not budget_ok:
            logger.warning(
                "top-p FP64 reference gate failed: batch=%d rotation=%d tv=%.9g reason=%s",
                batch,
                rotation,
                tv,
                reason,
            )
            return False
        if not _bits_equal(candidate.cpu(), expected):
            logger.info(
                "top-p FP64 boundary mask differs within budget: batch=%d rotation=%d max_tv=%.9g",
                batch,
                rotation,
                tv,
            )
        baseline_input = source.clone()
        baseline = original(baseline_input, None, p)
        torch.cuda.synchronize(device)
        if not torch.equal(
            candidate.cpu().view(torch.int32), baseline.cpu().view(torch.int32)
        ):
            logger.info(
                "top-p candidate differs from upstream FP32 mask: batch=%d rotation=%d",
                batch,
                rotation,
            )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = kernel_apply(source, p)
        graph.replay()
        torch.cuda.synchronize(device)
        if not _bits_equal(candidate, graph_output):
            logger.warning(
                "top-p eager/graph output mismatch: batch=%d rotation=%d",
                batch,
                rotation,
            )
            return False
        actual_keep = _keep_from_output(source, candidate)
        if not bool(actual_keep.any(dim=-1).all()):
            logger.warning(
                "top-p candidate removed every token: batch=%d rotation=%d",
                batch,
                rotation,
            )
            return False
    return True


def _measure_group(fn, inputs, ps, device):
    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    wall_start = time.perf_counter()
    outputs = []
    for source, p in zip(inputs, ps):
        work = source.clone()
        outputs.append(fn(work, None, p))
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    event_ms = start.elapsed_time(end)
    del outputs
    return wall_ms, event_ms


def _benchmark_batch(original, candidate, inputs, ps, device):
    # Warm both complete entries on all three independent rotations.
    _measure_group(original, inputs, ps, device)
    _measure_group(candidate, inputs, ps, device)
    samples = {"original": [], "candidate": []}
    for _ in range(2):
        for arm in ("candidate", "original", "original", "candidate"):
            fn = candidate if arm == "candidate" else original
            samples[arm].append(_measure_group(fn, inputs, ps, device))
    summaries = {}
    for arm in ("original", "candidate"):
        summaries[arm] = {
            "wall_ms": statistics.median(value[0] for value in samples[arm]),
            "event_ms": statistics.median(value[1] for value in samples[arm]),
        }
    return summaries


def _startup_gate(device, vocab: int, original, kernel_apply):
    def candidate_entry(logits, k, p):
        if k is not None:
            return original(logits, k, p)
        return kernel_apply(logits, p)

    if not _validate_small_cases(kernel_apply, device):
        return None
    results = {}
    for batch in (8, 32, 64):
        inputs, ps = _make_inputs(batch, vocab, device)
        if not _validate_real_inputs(kernel_apply, original, batch, inputs, ps, device):
            return None
        results[batch] = _benchmark_batch(original, candidate_entry, inputs, ps, device)
        old = results[batch]["original"]
        new = results[batch]["candidate"]
        wall_ratio = new["wall_ms"] / old["wall_ms"]
        event_ratio = new["event_ms"] / old["event_ms"]
        logger.info(
            "top-p startup full-entry batch=%d vocab=%d 3rot-total-wall=%.3f/%.3fms 3rot-total-event=%.3f/%.3fms",
            batch,
            vocab,
            old["wall_ms"],
            new["wall_ms"],
            old["event_ms"],
            new["event_ms"],
        )
        results[batch]["wall_ratio"] = wall_ratio
        results[batch]["event_ratio"] = event_ratio

    passing = {
        batch: values["wall_ratio"] <= _PERF_RATIO_LIMIT
        and values["event_ratio"] <= _PERF_RATIO_LIMIT
        for batch, values in results.items()
    }
    if not passing[64]:
        return None
    if passing[8] and passing[32]:
        min_batch = 8
    elif passing[32]:
        min_batch = 32
    else:
        min_batch = 64
    logger.info("top-p startup gate passed; dispatch batch range >=%d", min_batch)
    return min_batch


def _eligible(logits, k, p, min_batch: int, supported_vocab: int) -> bool:
    return (
        k is None
        and p is not None
        and isinstance(logits, torch.Tensor)
        and isinstance(p, torch.Tensor)
        and logits.ndim == 2
        and logits.dtype == torch.float32
        and logits.is_cuda
        and logits.device.type == "cuda"
        and logits.is_contiguous()
        and logits.shape[0] >= min_batch
        and logits.shape[0] <= _MAX_BATCH
        and logits.shape[1] == supported_vocab
        and _MIN_VOCAB <= logits.shape[1] <= _MAX_VOCAB
        and p.ndim == 1
        and p.shape[0] == logits.shape[0]
        and p.dtype == torch.float32
        and p.device == logits.device
        and p.is_contiguous()
    )


def _make_wrapper(original, kernel_apply, min_batch: int, supported_vocab: int):
    logged = False

    def apply(logits, k, p):
        nonlocal logged
        if not _eligible(logits, k, p, min_batch, supported_vocab):
            return original(logits, k, p)
        if not logged:
            logger.info(
                "FP64 top-p selection active: batch=%d vocab=%d min_batch=%d",
                logits.shape[0],
                logits.shape[1],
                min_batch,
            )
            logged = True
        return kernel_apply(logits, p)

    return apply


def install(device: torch.device, vocab: int) -> bool:
    """Install only after numerical, graph/eager, and complete-entry gates pass."""
    global _INSTALLED, _WRAPPER
    try:
        if device.type != "cuda" or torch.version.hip is not None:
            return False
        profile = device_profiles.lookup(device.index) or {}
        if not envs.switch(_SWITCH, auto=profile.get("topp_select") is True):
            return False
        if not _MIN_VOCAB <= vocab <= _MAX_VOCAB:
            logger.info("FP64 top-p vocab unsupported (%d); using upstream", vocab)
            return False
        from vllm.v1.sample.ops import topk_topp_sampler

        if _INSTALLED:
            return topk_topp_sampler.apply_top_k_top_p is _WRAPPER
        original = topk_topp_sampler.apply_top_k_top_p
        from vllm_fl.ops._topp_select_triton import apply as kernel_apply

        min_batch = _startup_gate(device, vocab, original, kernel_apply)
        if min_batch is None:
            logger.warning("FP64 top-p startup gate failed; using upstream")
            return False
        wrapper = _make_wrapper(original, kernel_apply, min_batch, vocab)
        topk_topp_sampler.apply_top_k_top_p = wrapper
        _WRAPPER = wrapper
        _INSTALLED = True
        logger.info("FP64 top-p selection enabled")
        return True
    except Exception:
        logger.warning(
            "FP64 top-p selection unavailable; using upstream", exc_info=True
        )
        return False
