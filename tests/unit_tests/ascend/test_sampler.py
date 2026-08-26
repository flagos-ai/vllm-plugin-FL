from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


def _sampler_module():
    from vllm_fl.dispatch.backends.vendor.ascend.impl import sampler

    return sampler


def test_ascend_sampler_preserves_logprobs_mode():
    sampler = _sampler_module()

    instance = sampler.AscendSampler(logprobs_mode="processed_logits")

    assert instance.logprobs_mode == "processed_logits"
    assert isinstance(
        instance.topk_topp_sampler,
        sampler.AscendTopKTopPSampler,
    )
    assert instance.topk_topp_sampler.logprobs_mode == "processed_logits"
    assert instance.topk_topp_sampler.enable_global_stream_random_sample


def test_ascend_sampler_propagates_global_stream_opt_in():
    sampler = _sampler_module()

    instance = sampler.AscendSampler(
        enable_global_stream_random_sample=False
    )

    assert not instance.topk_topp_sampler.enable_global_stream_random_sample


def test_npu_platform_binds_forward_to_ascend_override(monkeypatch):
    from vllm.v1.sample.ops import topk_topp_sampler as upstream_sampler

    sampler = _sampler_module()
    fake_npu_platform = SimpleNamespace(
        is_cuda=lambda: False,
        is_cpu=lambda: False,
    )
    monkeypatch.setattr(
        upstream_sampler,
        "current_platform",
        fake_npu_platform,
    )

    instance = sampler.AscendTopKTopPSampler()

    assert (
        instance.forward.__func__
        is sampler.AscendTopKTopPSampler.forward_native
    )


@pytest.mark.parametrize(
    ("dispatch_result", "expected"),
    [(True, True), (False, False)],
)
def test_capability_check_requires_privateuse1_kernel(
    monkeypatch,
    dispatch_result,
    expected,
):
    sampler = _sampler_module()
    dispatch_check = MagicMock(return_value=dispatch_result)
    monkeypatch.setattr(
        sampler.torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        dispatch_check,
    )

    result = sampler._has_ascend_top_k_top_p_kernel()

    assert result is expected
    dispatch_check.assert_called_once_with(
        "_C_ascend::npu_apply_top_k_top_p",
        "PrivateUse1",
    )


def test_capability_check_handles_missing_schema(monkeypatch):
    sampler = _sampler_module()
    dispatch_check = MagicMock(side_effect=RuntimeError("schema missing"))
    monkeypatch.setattr(
        sampler.torch._C,
        "_dispatch_has_kernel_for_dispatch_key",
        dispatch_check,
    )

    assert not sampler._has_ascend_top_k_top_p_kernel()


def test_op_resolution_rejects_schema_without_privateuse1_kernel(
    monkeypatch,
):
    sampler = _sampler_module()
    monkeypatch.setattr(
        sampler,
        "_has_ascend_top_k_top_p_kernel",
        lambda: False,
    )

    assert sampler._get_ascend_top_k_top_p_op() is None


def test_apply_top_k_top_p_prefers_ascend_op(monkeypatch):
    sampler = _sampler_module()
    logits = torch.randn(4, 16)
    k = torch.full((4,), 8, dtype=torch.int32)
    p = torch.full((4,), 0.9)
    filtered = torch.empty_like(logits)
    ascend_op = MagicMock(return_value=filtered)
    fallback = MagicMock(side_effect=AssertionError("fallback was used"))
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        sampler,
        "_get_ascend_top_k_top_p_op",
        lambda: ascend_op,
    )
    monkeypatch.setattr(sampler, "apply_top_k_top_p_pytorch", fallback)

    result = sampler.apply_top_k_top_p(logits, k, p)

    assert result is filtered
    ascend_op.assert_called_once_with(logits, k=k, p=p)
    fallback.assert_not_called()


def test_missing_ascend_op_uses_fixed_pytorch_fallback(monkeypatch):
    sampler = _sampler_module()
    logits = torch.randn(32, 64)
    p = torch.full((32,), 0.95)
    filtered = torch.empty_like(logits)
    fallback = MagicMock(return_value=filtered)
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(
        sampler,
        "_get_ascend_top_k_top_p_op",
        lambda: None,
    )
    monkeypatch.setattr(sampler, "apply_top_k_top_p_pytorch", fallback)

    result = sampler.apply_top_k_top_p(logits, None, p)

    assert result is filtered
    fallback.assert_called_once_with(logits, None, p)


def test_batch_invariant_mode_uses_fixed_pytorch_fallback(monkeypatch):
    sampler = _sampler_module()
    logits = torch.randn(8, 32)
    k = torch.full((8,), 16, dtype=torch.int32)
    filtered = torch.empty_like(logits)
    ascend_op = MagicMock(side_effect=AssertionError("Ascend op was used"))
    fallback = MagicMock(return_value=filtered)
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(
        sampler,
        "_get_ascend_top_k_top_p_op",
        lambda: ascend_op,
    )
    monkeypatch.setattr(sampler, "apply_top_k_top_p_pytorch", fallback)

    result = sampler.apply_top_k_top_p(logits, k, None)

    assert result is filtered
    fallback.assert_called_once_with(logits, k, None)
    ascend_op.assert_not_called()


def test_no_filter_is_identity_and_does_not_resolve_op(monkeypatch):
    sampler = _sampler_module()
    logits = torch.randn(2, 8)
    resolve_op = MagicMock(side_effect=AssertionError("op was resolved"))
    fallback = MagicMock(side_effect=AssertionError("fallback was used"))
    monkeypatch.setattr(sampler, "_get_ascend_top_k_top_p_op", resolve_op)
    monkeypatch.setattr(sampler, "apply_top_k_top_p_pytorch", fallback)

    result = sampler.apply_top_k_top_p(logits, None, None)

    assert result is logits
    resolve_op.assert_not_called()
    fallback.assert_not_called()


@pytest.mark.parametrize(
    "logprobs_mode",
    [
        "processed_logits",
        "processed_logprobs",
        "raw_logits",
        "raw_logprobs",
    ],
)
def test_forward_native_preserves_processed_logprobs_contract(
    monkeypatch,
    logprobs_mode,
):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler(
        logprobs_mode=logprobs_mode,
        enable_global_stream_random_sample=False,
    )
    logits = torch.randn(3, 7)
    filtered = torch.randn(3, 7)
    sampled = torch.tensor([1, 2, 3])
    apply_filter = MagicMock(return_value=filtered)
    random_sample = MagicMock(return_value=sampled)
    monkeypatch.setattr(sampler, "apply_top_k_top_p", apply_filter)
    monkeypatch.setattr(sampler, "random_sample", random_sample)

    result, returned_values = instance.forward_native(
        logits,
        {},
        None,
        torch.full((3,), 0.9),
    )

    assert result is sampled
    if logprobs_mode == "processed_logits":
        assert returned_values is filtered
    elif logprobs_mode == "processed_logprobs":
        torch.testing.assert_close(
            returned_values,
            filtered.log_softmax(dim=-1, dtype=torch.float32),
        )
    else:
        assert returned_values is None
    random_sample.assert_called_once()


def test_async_resources_are_lazy():
    sampler = _sampler_module()

    instance = sampler.AscendTopKTopPSampler()

    assert instance._async_exponential_stream is None
    assert instance._async_exponential_event is None
    assert not instance.has_pending_async_exponential


@pytest.mark.parametrize("generator_indices", [[], [1], [0, 1, 2]])
def test_fill_exponential_handles_none_partial_and_all_generators(
    generator_indices,
):
    sampler = _sampler_module()
    q = torch.empty((3, 11), dtype=torch.float32)
    generators = {
        index: torch.Generator().manual_seed(100 + index)
        for index in generator_indices
    }
    expected_seeded_rows = {}
    for index in generator_indices:
        expected = torch.empty(11, dtype=torch.float32)
        expected.exponential_(
            generator=torch.Generator().manual_seed(100 + index)
        )
        expected_seeded_rows[index] = expected

    sampler._fill_exponential_(q, generators)

    assert torch.isfinite(q).all()
    assert (q > 0).all()
    for index, expected in expected_seeded_rows.items():
        torch.testing.assert_close(q[index], expected)


def test_fill_exponential_rejects_invalid_generator_index():
    sampler = _sampler_module()
    q = torch.empty((2, 4), dtype=torch.float32)

    with pytest.raises(IndexError, match="invalid=\\[2\\]"):
        sampler._fill_exponential_(
            q,
            {2: torch.Generator().manual_seed(1)},
        )


def test_global_stream_random_sample_is_lazy_and_preserves_order(monkeypatch):
    sampler = _sampler_module()
    events = []
    scopes = []
    stream = object()
    current_stream = MagicMock()
    current_stream.wait_stream.side_effect = lambda value: events.append(
        ("wait_stream", value)
    )
    stream_factory = MagicMock(return_value=stream)

    class StreamContext:
        def __enter__(self):
            events.append(("enter_stream", stream))

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("exit_stream", stream))

    fake_npu = SimpleNamespace(
        Stream=stream_factory,
        stream=MagicMock(return_value=StreamContext()),
        current_stream=MagicMock(return_value=current_stream),
    )
    q = MagicMock()
    probs = MagicMock()
    divided = MagicMock()
    argmaxed = MagicMock()
    sampled = object()
    probs.div_.return_value = divided
    divided.argmax.return_value = argmaxed
    argmaxed.view.return_value = sampled
    generators = {1: object()}
    empty_like = MagicMock(
        side_effect=lambda value: events.append(("empty_like", value)) or q
    )
    fill = MagicMock(
        side_effect=lambda value, gens: events.append(("fill", value, gens))
    )
    monkeypatch.setattr(sampler, "_GLOBAL_RANDOM_SAMPLE_STREAM", None)
    monkeypatch.setattr(sampler.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(sampler.torch, "empty_like", empty_like)
    monkeypatch.setattr(sampler, "_fill_exponential_", fill)
    monkeypatch.setattr(
        sampler,
        "record_function_or_nullcontext",
        lambda name: scopes.append(name) or nullcontext(),
    )

    result = sampler.global_stream_random_sample(probs, generators)
    assert sampler._global_random_sample_stream() is stream

    assert result is sampled
    stream_factory.assert_called_once_with()
    fake_npu.stream.assert_called_once_with(stream)
    fake_npu.current_stream.assert_called_once_with()
    fill.assert_called_once_with(q, generators)
    current_stream.wait_stream.assert_called_once_with(stream)
    assert events == [
        ("enter_stream", stream),
        ("empty_like", probs),
        ("fill", q, generators),
        ("exit_stream", stream),
        ("wait_stream", stream),
    ]
    assert scopes == [
        "sampler: exponential_submit",
        "sampler: wait_random_stream",
        "sampler: div_argmax",
    ]
    probs.div_.assert_called_once_with(q)


def test_prepare_async_exponential_is_one_shot_and_records_event(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler()
    async_stream = MagicMock()
    current_stream = MagicMock()
    event = MagicMock()
    fake_npu = SimpleNamespace(
        Stream=MagicMock(return_value=async_stream),
        Event=MagicMock(return_value=event),
        current_stream=MagicMock(return_value=current_stream),
        stream=MagicMock(return_value=nullcontext()),
    )
    monkeypatch.setattr(sampler.torch, "npu", fake_npu, raising=False)
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", False)

    instance.prepare_async_exponential(
        batch_size=3,
        vocab_size=5,
        generators={},
        device=torch.device("cpu"),
    )

    assert instance.has_pending_async_exponential
    assert instance._pending_q.shape == (3, 5)
    assert instance._pending_q.dtype == torch.float32
    async_stream.wait_stream.assert_called_once_with(current_stream)
    event.record.assert_called_once_with()
    with pytest.raises(RuntimeError, match="before consuming pending q"):
        instance.prepare_async_exponential(
            batch_size=3,
            vocab_size=5,
            generators={},
            device=torch.device("cpu"),
        )


def test_batch_invariant_rejects_direct_async_prepare(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler()
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", True)

    with pytest.raises(RuntimeError, match="VLLM_BATCH_INVARIANT=1"):
        instance.prepare_async_exponential(
            batch_size=1,
            vocab_size=4,
            generators={},
            device=torch.device("cpu"),
        )

    assert instance._async_exponential_stream is None


def test_discard_pending_q_synchronizes_clears_and_allows_recovery(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler(
        enable_global_stream_random_sample=False
    )
    event = MagicMock()
    instance._pending_q = torch.ones((2, 3), dtype=torch.float32)
    instance._pending_q_event = event

    assert instance.discard_async_exponential()
    event.synchronize.assert_called_once_with()
    assert not instance.has_pending_async_exponential
    assert instance._pending_q_event is None
    assert not instance.discard_async_exponential()

    sampled = torch.tensor([1, 2])
    fallback = MagicMock(return_value=sampled)
    monkeypatch.setattr(sampler, "random_sample", fallback)
    result, _ = instance.forward_native(torch.randn(2, 3), {}, None, None)
    assert result is sampled
    fallback.assert_called_once()


def test_discard_clears_state_even_when_event_synchronize_fails():
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler()
    event = MagicMock()
    event.synchronize.side_effect = RuntimeError("event failed")
    instance._pending_q = torch.ones((1, 2), dtype=torch.float32)
    instance._pending_q_event = event

    with pytest.raises(RuntimeError, match="event failed"):
        instance.discard_async_exponential()

    assert not instance.has_pending_async_exponential
    assert instance._pending_q_event is None


def test_forward_consumes_pending_q_once_without_random_fallback(monkeypatch):
    sampler = _sampler_module()
    scopes = []
    instance = sampler.AscendTopKTopPSampler(
        enable_global_stream_random_sample=True
    )
    logits = torch.tensor([[2.0, 1.0], [0.5, 1.5]])
    q = torch.tensor([[100.0, 0.1], [0.1, 100.0]])
    event = MagicMock()
    record_q = MagicMock()
    fallback = MagicMock(side_effect=AssertionError("fallback was used"))
    global_random = MagicMock(
        side_effect=AssertionError("global random fallback was used")
    )
    monkeypatch.setattr(sampler, "apply_top_k_top_p", lambda x, k, p: x)
    monkeypatch.setattr(sampler, "random_sample", fallback)
    monkeypatch.setattr(sampler, "global_stream_random_sample", global_random)
    monkeypatch.setattr(instance, "_record_q_stream", record_q)
    monkeypatch.setattr(
        sampler,
        "record_function_or_nullcontext",
        lambda name: scopes.append(name) or nullcontext(),
    )
    instance._pending_q = q
    instance._pending_q_event = event

    sampled, _ = instance.forward_native(logits, {}, None, None)

    torch.testing.assert_close(sampled, torch.tensor([1, 0]))
    event.synchronize.assert_called_once_with()
    record_q.assert_called_once_with(q)
    assert not instance.has_pending_async_exponential
    assert instance._pending_q_event is None
    fallback.assert_not_called()
    global_random.assert_not_called()
    assert scopes == [
        "sampler: softmax",
        "sampler: wait_random_stream",
        "sampler: div_argmax",
    ]


@pytest.mark.parametrize(
    ("q", "error_fragment"),
    [
        (torch.ones((1, 3), dtype=torch.float32), "shape q=(1, 3)"),
        (
            torch.ones((2, 3), dtype=torch.float32, device="meta"),
            "device q=meta probs=cpu",
        ),
        (torch.ones((2, 3), dtype=torch.float64), "dtype q=torch.float64"),
    ],
)
def test_async_q_mismatch_raises_and_clears_without_fallback(
    monkeypatch,
    q,
    error_fragment,
):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler()
    event = MagicMock()
    fallback = MagicMock(side_effect=AssertionError("fallback was used"))
    monkeypatch.setattr(sampler, "random_sample", fallback)
    instance._pending_q = q
    instance._pending_q_event = event

    with pytest.raises(RuntimeError) as exc_info:
        instance.forward_native(torch.randn(2, 3), {}, None, None)

    assert error_fragment in str(exc_info.value)
    event.synchronize.assert_called_once_with()
    assert not instance.has_pending_async_exponential
    assert instance._pending_q_event is None
    fallback.assert_not_called()


def test_no_pending_q_keeps_existing_random_sample(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler(
        enable_global_stream_random_sample=False
    )
    sampled = torch.tensor([2, 1])
    fallback = MagicMock(return_value=sampled)
    monkeypatch.setattr(sampler, "random_sample", fallback)

    result, _ = instance.forward_native(torch.randn(2, 3), {}, None, None)

    assert result is sampled
    fallback.assert_called_once()


def test_global_stream_opt_in_uses_target_random_path(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler(
        enable_global_stream_random_sample=True
    )
    sampled = torch.tensor([0, 2])
    global_random = MagicMock(return_value=sampled)
    fallback = MagicMock(side_effect=AssertionError("fallback was used"))
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", False)
    monkeypatch.setattr(sampler, "global_stream_random_sample", global_random)
    monkeypatch.setattr(sampler, "random_sample", fallback)

    result, _ = instance.forward_native(torch.randn(2, 3), {}, None, None)

    assert result is sampled
    global_random.assert_called_once()
    fallback.assert_not_called()


def test_batch_invariant_forces_upstream_random_path(monkeypatch):
    sampler = _sampler_module()
    instance = sampler.AscendTopKTopPSampler(
        enable_global_stream_random_sample=True
    )
    sampled = torch.tensor([2, 0])
    global_random = MagicMock(
        side_effect=AssertionError("global stream path was used")
    )
    fallback = MagicMock(return_value=sampled)
    monkeypatch.setattr(sampler.envs, "VLLM_BATCH_INVARIANT", True)
    monkeypatch.setattr(sampler, "global_stream_random_sample", global_random)
    monkeypatch.setattr(sampler, "random_sample", fallback)

    result, _ = instance.forward_native(torch.randn(2, 3), {}, None, None)

    assert result is sampled
    fallback.assert_called_once()
    global_random.assert_not_called()
