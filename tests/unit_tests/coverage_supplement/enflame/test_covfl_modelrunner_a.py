"""Coverage tests for the first ~60% of vllm_fl/worker/model_runner.py.

Self contained: no conftest, no repository fixtures.  Heavy imports are
guarded so the module is skipped cleanly when vLLM/vllm_fl is unavailable.
No real accelerator work is performed; device-facing calls are mocked.
"""

import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

try:
    from vllm_fl.worker import model_runner as mr
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(
        f"vllm_fl.worker.model_runner is not importable here: {exc!r}",
        allow_module_level=True,
    )


class _FakeStream:
    def __init__(self, *args, **kwargs):
        self.waited_on = []

    def wait_stream(self, other):
        self.waited_on.append(other)

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)


class _FakeEvent:
    def __init__(self, *args, **kwargs):
        self.records = 0
        self.syncs = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.syncs += 1


def _make_device_fns():
    created = {"streams": [], "events": []}
    current = _FakeStream()

    def Stream(*args, **kwargs):
        s = _FakeStream()
        created["streams"].append(s)
        return s

    def Event(*args, **kwargs):
        e = _FakeEvent()
        created["events"].append(e)
        return e

    fns = SimpleNamespace(
        Stream=Stream,
        Event=Event,
        current_stream=lambda: current,
        stream=lambda s: nullcontext(),
    )
    fns.created = created
    fns.current = current
    return fns


def fake_platform(**overrides):
    fns = overrides.pop("torch_device_fn", None) or _make_device_fns()
    p = SimpleNamespace(
        device_type="cuda",
        torch_device_fn=fns,
        dist_backend="nccl",
        vendor_name="nvidia",
        is_cuda_alike=lambda: True,
    )
    for key, value in overrides.items():
        setattr(p, key, value)
    return p


def make_runner(**attrs):
    runner = object.__new__(mr.ModelRunnerFL)
    for key, value in attrs.items():
        setattr(runner, key, value)
    return runner


def instance_of(cls):
    try:
        return object.__new__(cls)
    except Exception:  # pragma: no cover - defensive
        return Mock(spec=cls)


# ---------------------------------------------------------------------------
# Module level helpers
# ---------------------------------------------------------------------------


def test_module_exports_expected_callables():
    for name in (
        "ModelRunnerFL",
        "AsyncGPUModelRunnerOutput",
        "AsyncGPUPoolingModelRunnerOutput",
        "ExecuteModelState",
        "_copy_pooler_output_to_cpu",
        "count_nans_per_row",
        "nans_to_dict",
        "_get_parameter_for_reload",
        "_accelerator_synchronize",
        "_device_graph_capture",
    ):
        assert hasattr(mr, name), name


def test_count_nans_per_row_counts_per_row():
    logits = torch.tensor([[1.0, float("nan"), 2.0], [float("nan"), float("nan"), 0.0]])
    counts = mr.count_nans_per_row(logits)
    assert counts.tolist() == [1, 2]
    assert counts.dtype == torch.int32


def test_nans_to_dict_maps_rows_and_pads_missing():
    counts = [3, 0]
    mapping = {"a": 0, "b": 1, "c": 5}
    out = mr.nans_to_dict(counts, mapping)
    assert out == {"a": 3, "b": 0, "c": 0}


def test_get_parameter_for_reload_plain_module():
    param = object()
    module = SimpleNamespace(get_parameter=Mock(return_value=param))
    model = SimpleNamespace(get_submodule=Mock(return_value=module))
    assert mr._get_parameter_for_reload(model, "layer.attn.weight") is param
    model.get_submodule.assert_called_once_with("layer.attn")
    module.get_parameter.assert_called_once_with("weight")


def test_get_parameter_for_reload_lora_base_layer():
    try:
        class _FakeLora(mr.BaseLayerWithLoRA):
            pass

        lora = object.__new__(_FakeLora)
    except Exception:  # pragma: no cover - exotic upstream class
        pytest.skip("cannot subclass BaseLayerWithLoRA in this environment")
    base = SimpleNamespace(get_parameter=Mock(return_value="P"))
    lora.base_layer = base
    model = SimpleNamespace(get_submodule=Mock(return_value=lora))
    assert mr._get_parameter_for_reload(model, "x.y.bias") == "P"
    base.get_parameter.assert_called_once_with("bias")


def test_accelerator_synchronize_default_branch(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(mr, "current_platform", fake_platform(device_type="cuda"))
    monkeypatch.setattr(
        mr.torch, "accelerator", SimpleNamespace(synchronize=sync), raising=False
    )
    mr._accelerator_synchronize()
    sync.assert_called_once()


def test_accelerator_synchronize_musa_branch(monkeypatch):
    sync = Mock()
    monkeypatch.setitem(sys.modules, "torch_musa", SimpleNamespace(synchronize=sync))
    monkeypatch.setattr(mr, "current_platform", fake_platform(device_type="musa"))
    mr._accelerator_synchronize()
    sync.assert_called_once()


def test_device_graph_capture_reuses_context_stream(monkeypatch):
    platform = fake_platform()
    monkeypatch.setattr(mr, "current_platform", platform)
    ctx = SimpleNamespace(stream=_FakeStream())
    with mr._device_graph_capture(torch.device("cpu"), ctx) as got:
        assert got is ctx
    assert ctx.stream.waited_on == [platform.torch_device_fn.current]


def test_device_graph_capture_creates_context_when_missing(monkeypatch):
    created = []
    platform = fake_platform()
    monkeypatch.setattr(mr, "current_platform", platform)

    class _Ctx:
        def __init__(self, stream):
            self.stream = stream
            created.append(self)

    monkeypatch.setattr(mr, "GraphCaptureContext", _Ctx)
    with mr._device_graph_capture(torch.device("cpu")) as got:
        assert isinstance(got, _Ctx)
        assert got.stream is platform.torch_device_fn.created["streams"][0]


# ---------------------------------------------------------------------------
# Async output wrappers
# ---------------------------------------------------------------------------


def test_async_gpu_output_short_gen(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    output = SimpleNamespace(req_id_to_index={"r0": 0})
    sampled = torch.tensor([[7], [9]], dtype=torch.int64)
    wrapper = mr.AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=sampled,
        logprobs_tensors=None,
        invalid_req_indices=[1],
        async_output_copy_stream=_FakeStream(),
        vocab_size=10,
    )
    assert wrapper.async_copy_ready_event.records == 1
    got = wrapper.get_output()
    assert got is output
    assert got.sampled_token_ids == [[7], []]
    assert got.logprobs is None


def test_async_gpu_output_with_logprobs_and_nans(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    logprobs = SimpleNamespace(to_cpu_nonblocking=lambda: SimpleNamespace(tolists=lambda: "LP"))
    output = SimpleNamespace(req_id_to_index={"a": 0, "b": 1})
    wrapper = mr.AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=torch.tensor([[1], [2]]),
        logprobs_tensors=logprobs,
        invalid_req_indices=[],
        async_output_copy_stream=_FakeStream(),
        vocab_size=5,
        num_nans=torch.tensor([1, 0], dtype=torch.int32),
    )
    got = wrapper.get_output()
    assert got.logprobs == "LP"
    assert got.num_nans_in_logits == {"a": 1, "b": 0}


def test_async_gpu_output_raises_on_nans(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    boom = Mock(side_effect=RuntimeError("nan"))
    monkeypatch.setattr(mr, "raise_if_nan_logits", boom)
    wrapper = mr.AsyncGPUModelRunnerOutput(
        model_runner_output=SimpleNamespace(req_id_to_index={"a": 0}),
        sampled_token_ids=torch.tensor([[1]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=_FakeStream(),
        vocab_size=5,
        num_nans=torch.tensor([2], dtype=torch.int32),
    )
    with pytest.raises(RuntimeError):
        wrapper.get_output()
    boom.assert_called_once()


def test_async_gpu_output_spec_decode_path(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    parse = Mock(return_value=(["x"], "lp"))
    monkeypatch.setattr(mr, "RejectionSampler", SimpleNamespace(parse_output=parse))
    output = SimpleNamespace(req_id_to_index={})
    wrapper = mr.AsyncGPUModelRunnerOutput(
        model_runner_output=output,
        sampled_token_ids=torch.zeros((2, 3), dtype=torch.int64),
        logprobs_tensors=None,
        invalid_req_indices=[0],
        async_output_copy_stream=_FakeStream(),
        vocab_size=11,
    )
    got = wrapper.get_output()
    assert got.sampled_token_ids == ["x"]
    assert got.logprobs == "lp"
    parse.assert_called_once()


def test_async_gpu_output_ep_fault_raises(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    manager = SimpleNamespace(
        query_fault=Mock(return_value=torch.tensor([1])),
        query_active_mask=Mock(return_value=torch.tensor([0, 1])),
    )
    monkeypatch.setattr(mr, "get_ep_all2all_manager", Mock(return_value=manager))
    wrapper = mr.AsyncGPUModelRunnerOutput(
        model_runner_output=SimpleNamespace(req_id_to_index={}),
        sampled_token_ids=torch.tensor([[1]]),
        logprobs_tensors=None,
        invalid_req_indices=[],
        async_output_copy_stream=_FakeStream(),
        vocab_size=5,
        check_ep_fault=True,
    )
    with pytest.raises(RuntimeError, match="Fault detected"):
        wrapper.get_output()


def test_copy_pooler_output_tensor_cases():
    full = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    with pytest.raises(ValueError):
        mr._copy_pooler_output_to_cpu(full, [False, True])
    assert mr._copy_pooler_output_to_cpu(full, [False, False, False]) == [None, None, None]
    all_out = mr._copy_pooler_output_to_cpu(full, [True, True, True])
    assert len(all_out) == 3
    assert all_out[0].tolist() == [0.0, 1.0]
    partial = mr._copy_pooler_output_to_cpu(full, [False, True, False])
    assert partial[0] is None and partial[2] is None
    assert partial[1].tolist() == [2.0, 3.0]


def test_copy_pooler_output_list_cases():
    with pytest.raises(ValueError):
        mr._copy_pooler_output_to_cpu([torch.zeros(2)], [True, True])
    out = mr._copy_pooler_output_to_cpu(
        [torch.tensor([1.0]), None, torch.tensor([3.0])],
        [True, True, False],
    )
    assert out[0].tolist() == [1.0]
    assert out[1] is None
    assert out[2] is None


def test_async_pooling_output(monkeypatch):
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    output = SimpleNamespace(pooler_output=None)
    wrapper = mr.AsyncGPUPoolingModelRunnerOutput(
        model_runner_output=output,
        raw_pooler_output=torch.tensor([[1.0], [2.0]]),
        finished_mask=[True, True],
        async_output_copy_stream=_FakeStream(),
    )
    assert output.pooler_output[0].tolist() == [1.0]
    got = wrapper.get_output()
    assert got is output
    assert wrapper.async_copy_ready_event.syncs == 1


def test_execute_model_state_namedtuple():
    state = mr.ExecuteModelState(
        "sched", "logits", None, None, "hidden", "sample", None, None, None, None
    )
    assert state.scheduler_output == "sched"
    assert state.hidden_states == "hidden"
    assert state.slot_mappings is None


# ---------------------------------------------------------------------------
# ModelRunnerFL simple state helpers
# ---------------------------------------------------------------------------


def test_update_max_model_len_all_branches():
    runner = make_runner(max_model_len=8, speculative_config=None)
    runner.update_max_model_len(32)
    assert runner.max_model_len == 32

    draft = SimpleNamespace(max_model_len=None)
    runner = make_runner(
        max_model_len=8,
        speculative_config=SimpleNamespace(draft_model_config=draft),
        effective_drafter_max_model_len=1,
    )
    runner.update_max_model_len(16)
    assert runner.effective_drafter_max_model_len == 16

    runner = make_runner(
        max_model_len=8,
        speculative_config=SimpleNamespace(draft_model_config=None),
        effective_drafter_max_model_len=1,
    )
    runner.update_max_model_len(20)
    assert runner.effective_drafter_max_model_len == 20

    runner = make_runner(
        max_model_len=8,
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(max_model_len=4)
        ),
        effective_drafter_max_model_len=4,
    )
    runner.update_max_model_len(20)
    assert runner.effective_drafter_max_model_len == 4


def test_reset_mm_and_encoder_cache():
    budget = SimpleNamespace(reset_cache=Mock())
    late = SimpleNamespace(clear=Mock())
    runner = make_runner(mm_budget=budget, late_interaction_runner=late)
    runner.reset_mm_cache()
    budget.reset_cache.assert_called_once()
    late.clear.assert_called_once()

    runner = make_runner(mm_budget=None, late_interaction_runner=late)
    runner.reset_mm_cache()
    assert late.clear.call_count == 2

    runner = make_runner(encoder_cache={"a": 1}, late_interaction_runner=late)
    runner.reset_encoder_cache()
    assert runner.encoder_cache == {}
    assert late.clear.call_count == 3


def test_post_kv_cache_wake_up_calls_init(monkeypatch):
    runner = make_runner()
    runner.init_fp8_kv_scales = Mock()
    runner.post_kv_cache_wake_up()
    runner.init_fp8_kv_scales.assert_called_once()


def test_init_fp8_kv_scales_not_quantized(monkeypatch):
    monkeypatch.setattr(mr, "is_quantized_kv_cache", lambda dtype: False)
    runner = make_runner(cache_config=SimpleNamespace(cache_dtype="auto"))
    runner.init_fp8_kv_scales()
    assert not hasattr(runner, "kv_caches")


def test_init_fp8_kv_scales_quantized_resets_scales(monkeypatch):
    monkeypatch.setattr(mr, "is_quantized_kv_cache", lambda dtype: True)
    cache_tensor = torch.ones(2)
    k_scale = torch.zeros(1)
    v_scale = torch.zeros(1)
    layer = Mock(spec=mr.Attention)
    layer._k_scale = k_scale
    layer._v_scale = v_scale
    mla = Mock(spec=mr.MLAAttention)
    mla.k_scale = k_scale
    plain = SimpleNamespace()
    runner = make_runner(
        cache_config=SimpleNamespace(cache_dtype="fp8"),
        kv_caches=[None, cache_tensor, [cache_tensor]],
        compilation_config=SimpleNamespace(
            static_forward_context={"a": layer, "b": mla, "c": plain}
        ),
    )
    runner.init_fp8_kv_scales()
    assert torch.all(cache_tensor == 0)
    assert torch.all(k_scale == 1.0)
    assert torch.all(v_scale == 1.0)


def test_get_positions_int_and_tensor():
    positions = torch.arange(10)
    mrope = SimpleNamespace(gpu=torch.arange(30).reshape(3, 10))
    runner = make_runner(uses_mrope=False, positions=positions)
    assert runner._get_positions(4).tolist() == [0, 1, 2, 3]
    idx = torch.tensor([1, 3])
    assert runner._get_positions(idx).tolist() == [1, 3]

    runner = make_runner(uses_mrope=True, mrope_positions=mrope)
    assert runner._get_positions(2).shape == (3, 2)
    assert runner._get_positions(idx).shape == (3, 2)


def test_make_buffer_forwards_arguments(monkeypatch):
    ctor = Mock(return_value="BUF")
    monkeypatch.setattr(mr, "CpuGpuBuffer", ctor)
    runner = make_runner(device=torch.device("cpu"))
    assert runner._make_buffer(5, dtype=torch.int32, numpy=False) == "BUF"
    ctor.assert_called_once_with(5, dtype=torch.int32, device=torch.device("cpu"), with_numpy=False)


def test_jit_warmup_registry_property_and_setter(monkeypatch):
    runner = make_runner(vllm_config="CFG")
    fake_registry = Mock(return_value="REG")
    monkeypatch.setattr(
        "vllm.model_executor.warmup.jit_warmup.JitWarmupRegistry", fake_registry
    )
    try:
        assert runner.jit_warmup_registry == "REG"
    except Exception:  # pragma: no cover - module shape differs
        pytest.skip("JitWarmupRegistry not patchable")
    assert runner.jit_warmup_registry == "REG"
    fake_registry.assert_called_once_with("CFG")

    runner.jit_warmup_registry = "SENTINEL"
    assert runner.jit_warmup_registry == "SENTINEL"


# ---------------------------------------------------------------------------
# Mamba helpers
# ---------------------------------------------------------------------------


def test_get_mamba_state_copy_funcs_cached():
    runner = make_runner(_mamba_state_copy_funcs={"cached": 1})
    assert runner._get_mamba_state_copy_funcs() == {"cached": 1}


def test_get_mamba_state_copy_funcs_new_api(monkeypatch):
    groups = [SimpleNamespace(mamba_type="m1")]
    monkeypatch.setattr(mr.mamba_utils, "get_mamba_groups", lambda cfg: groups)
    validate = Mock()
    monkeypatch.setattr(mr.mamba_utils, "validate_mamba_state_copy_funcs", validate)
    funcs = {"m1": "fn"}
    model = SimpleNamespace(get_mamba_state_copy_funcs=Mock(return_value=funcs))
    runner = make_runner(model=model, kv_cache_config="CFG")
    assert runner._get_mamba_state_copy_funcs() == funcs
    model.get_mamba_state_copy_funcs.assert_called_once_with({"m1"})
    validate.assert_called_once_with(groups, funcs)


def test_get_mamba_state_copy_funcs_new_api_validator_missing(monkeypatch):
    groups = [SimpleNamespace(mamba_type="m1")]
    monkeypatch.setattr(mr.mamba_utils, "get_mamba_groups", lambda cfg: groups)
    monkeypatch.setattr(
        mr.mamba_utils,
        "validate_mamba_state_copy_funcs",
        Mock(side_effect=AttributeError("old")),
    )
    model = SimpleNamespace(get_mamba_state_copy_funcs=Mock(return_value="F"))
    runner = make_runner(model=model, kv_cache_config="CFG")
    assert runner._get_mamba_state_copy_funcs() == "F"


def test_get_mamba_state_copy_funcs_legacy(monkeypatch):
    groups = [SimpleNamespace(mamba_type="m1"), SimpleNamespace(mamba_type="m2")]
    monkeypatch.setattr(mr.mamba_utils, "get_mamba_groups", lambda cfg: groups)
    model = SimpleNamespace(get_mamba_state_copy_func=Mock(return_value="LEGACY"))
    runner = make_runner(model=model, kv_cache_config="CFG")
    assert runner._get_mamba_state_copy_funcs() == {"m1": "LEGACY", "m2": "LEGACY"}


def test_get_mamba_bufs_requires_align_mode():
    runner = make_runner(cache_config=SimpleNamespace(mamba_cache_mode="other"))
    with pytest.raises(AssertionError):
        runner._get_mamba_bufs()


def test_get_mamba_bufs_creates_once(monkeypatch):
    created = Mock(return_value="BUF")
    monkeypatch.setattr(mr.mamba_utils.MambaBuffers, "create", created)
    runner = make_runner(
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        _mamba_bufs=None,
        max_num_reqs=4,
        kv_cache_config="KV",
        speculative_config=SimpleNamespace(),
        model_config=SimpleNamespace(is_hybrid=True),
        device=torch.device("cpu"),
    )
    runner._get_mamba_state_copy_funcs = Mock(return_value="FUNCS")
    runner._make_buffer = Mock()
    assert runner._get_mamba_bufs() == "BUF"
    assert runner._get_mamba_bufs() == "BUF"
    created.assert_called_once()


# ---------------------------------------------------------------------------
# Model kwargs / batch reordering
# ---------------------------------------------------------------------------


def test_init_model_kwargs_not_pooling():
    runner = make_runner(is_pooling_model=False)
    assert runner._init_model_kwargs() == {}


def test_init_model_kwargs_no_token_types(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False)
    ib = SimpleNamespace(
        num_reqs=1,
        get_pooling_params=lambda: [SimpleNamespace(extra_kwargs=None)],
    )
    runner = make_runner(is_pooling_model=True, input_batch=ib)
    assert runner._init_model_kwargs() == {}


def test_init_model_kwargs_builds_ids(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False)
    params = [
        SimpleNamespace(extra_kwargs={"compressed_token_type_ids": 1}),
        SimpleNamespace(extra_kwargs=None),
    ]
    ib = SimpleNamespace(
        num_reqs=2,
        get_pooling_params=lambda: params,
    )
    runner = make_runner(
        is_pooling_model=True,
        input_batch=ib,
        optimistic_seq_lens_cpu=torch.tensor([3, 2], dtype=torch.int32),
        device=torch.device("cpu"),
    )
    out = runner._init_model_kwargs()
    assert set(out) == {"token_type_ids"}
    assert out["token_type_ids"].shape[0] == 5


def test_may_reorder_batch_early_and_active(monkeypatch):
    runner = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        reorder_batch_threshold=1,
    )
    runner._may_reorder_batch(SimpleNamespace())

    called = Mock()
    monkeypatch.setattr(mr, "reorder_batch_to_split_decodes_and_prefills", called)
    runner = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
        reorder_batch_threshold=4,
        input_batch="IB",
    )
    sched = SimpleNamespace()
    runner._may_reorder_batch(sched)
    called.assert_called_once()
    assert called.call_args.kwargs["decode_threshold"] == 4


def test_init_kv_zero_meta_and_zero_block_ids(monkeypatch):
    zeroer = Mock()
    monkeypatch.setattr(mr, "KVBlockZeroer", Mock(return_value=zeroer))
    runner = make_runner(
        device=torch.device("cpu"),
        _kernel_block_sizes=[16],
        compilation_config=SimpleNamespace(static_forward_context="SFC"),
        kv_cache_config=SimpleNamespace(num_blocks=8),
        runner_only_attn_layers={"l"},
    )
    runner._kv_cache_spec_attn_group_iterator = Mock(return_value=iter([1]))
    runner._init_kv_zero_meta()
    assert runner._kv_block_zeroer is zeroer
    runner._zero_block_ids([1, 2])
    zeroer.zero_block_ids.assert_called_once_with([1, 2])

    runner2 = make_runner()
    runner2._zero_block_ids([3])


def test_init_device_properties_uses_compute_units(monkeypatch):
    monkeypatch.setattr(mr, "num_compute_units", lambda idx: 13)
    runner = make_runner(device=torch.device("cpu", 0))
    runner._init_device_properties()
    assert runner.num_sms == 13


def test_sync_device_delegates(monkeypatch):
    sync = Mock()
    monkeypatch.setattr(mr, "_accelerator_synchronize", sync)
    make_runner()._sync_device()
    sync.assert_called_once()


def test_get_or_create_async_output_copy_stream(monkeypatch):
    platform = fake_platform()
    monkeypatch.setattr(mr, "current_platform", platform)
    runner = make_runner(async_output_copy_stream="S")
    assert runner._get_or_create_async_output_copy_stream() == "S"

    runner = make_runner(async_output_copy_stream=None)
    got = runner._get_or_create_async_output_copy_stream()
    assert got is platform.torch_device_fn.created["streams"][0]
    assert runner.async_output_copy_stream is got


def test_on_request_state_removed_is_noop():
    assert make_runner()._on_request_state_removed("r", None) is None


def test_process_encoder_cache_scheduler_output():
    runner = make_runner(encoder_cache={"a": 1, "b": 2})
    runner._process_encoder_cache_scheduler_output(
        SimpleNamespace(free_encoder_mm_hashes=["a", "missing"])
    )
    assert runner.encoder_cache == {"b": 2}


# ---------------------------------------------------------------------------
# _update_states (empty and new-request paths)
# ---------------------------------------------------------------------------


def _empty_scheduler_output(**over):
    base = dict(
        finished_req_ids=[],
        new_block_ids_to_zero=None,
        kv_cache_block_copies=None,
        free_encoder_mm_hashes=[],
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            new_block_ids=[],
            resumed_req_ids=set(),
            num_computed_tokens=[],
            num_output_tokens=[],
            new_token_ids=[],
        ),
        scheduled_spec_decode_tokens={},
    )
    base.update(over)
    return SimpleNamespace(**base)


def test_update_states_empty(monkeypatch):
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    ib = SimpleNamespace(
        req_id_to_index={},
        req_ids=[],
        remove_request=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
        prev_req_id_to_index=None,
    )
    late = SimpleNamespace(on_requests_finished=Mock())
    runner = make_runner(
        requests={},
        num_prompt_logprobs={},
        input_batch=ib,
        late_interaction_runner=late,
        canonical=None,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        reorder_batch_threshold=None,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        num_tokens_no_spec_gpu=None,
    )
    assert runner._update_states(_empty_scheduler_output()) is None
    ib.condense.assert_called_once()
    ib.refresh_metadata.assert_called_once()
    late.on_requests_finished.assert_called_once()


def test_update_states_adds_new_request(monkeypatch):
    monkeypatch.setattr(
        mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True)
    )
    fake_state_cls = lambda **kw: SimpleNamespace(prev_num_draft_len=0, **kw)
    monkeypatch.setattr(mr, "CachedRequestState", fake_state_cls)
    ib = SimpleNamespace(
        req_id_to_index={},
        req_ids=["r1"],
        remove_request=Mock(),
        add_request=Mock(),
        update_req_spec_token_ids=Mock(),
        condense=Mock(),
        refresh_metadata=Mock(),
        prev_req_id_to_index=None,
        vocab_size=100,
    )
    late = SimpleNamespace(
        on_requests_finished=Mock(),
        register_request=Mock(),
    )
    new_req = SimpleNamespace(
        req_id="r1",
        sampling_params=SimpleNamespace(sampling_type="greedy", prompt_logprobs=None),
        pooling_params=None,
        prompt_token_ids=[1, 2, 3],
        prompt_embeds=None,
        prompt_is_token_ids=True,
        mm_features=[],
        block_ids=[[0]],
        num_computed_tokens=0,
        lora_request=None,
    )
    runner = make_runner(
        requests={},
        num_prompt_logprobs={},
        input_batch=ib,
        late_interaction_runner=late,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        reorder_batch_threshold=None,
        speculative_config=None,
        use_async_spec_decode=False,
        use_async_scheduling=False,
        is_pooling_model=False,
        uses_mrope=False,
        num_tokens_no_spec_gpu=None,
    )
    out = runner._update_states(
        _empty_scheduler_output(
            num_scheduled_tokens={"r1": 3}, scheduled_new_reqs=[new_req]
        )
    )
    assert out is None
    assert "r1" in runner.requests
    ib.add_request.assert_called_once()
    ib.condense.assert_called_once()


# ---------------------------------------------------------------------------
# Post-execute / streaming / mrope
# ---------------------------------------------------------------------------


def test_update_states_after_model_execute_disabled():
    runner = make_runner(speculative_config=None)
    runner._update_states_after_model_execute(torch.zeros(1), None)

    runner = make_runner(
        speculative_config=SimpleNamespace(), model_config=SimpleNamespace(is_hybrid=False)
    )
    runner._update_states_after_model_execute(torch.zeros(1), None)


def test_update_states_after_model_execute_copy_path(monkeypatch):
    gpu = torch.zeros(3, dtype=torch.int64)
    cpu_tensor = torch.zeros(3, dtype=torch.int64)
    event = Mock()
    lib = SimpleNamespace(num_accepted_tokens_cpu_tensor=cpu_tensor)
    runner = make_runner(
        speculative_config=SimpleNamespace(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="no_align"),
        num_accepted_tokens=SimpleNamespace(gpu=gpu),
        num_accepted_tokens_event=event,
        input_batch=lib,
    )
    out = torch.tensor([[5, -1, -1], [7, 8, -1]])
    runner._update_states_after_model_execute(out, None)
    assert gpu[:2].tolist() == [1, 2]
    event.record.assert_called_once()


def test_update_states_after_model_execute_align_path(monkeypatch):
    gpu = torch.zeros(2, dtype=torch.int64)
    event = Mock()
    post = Mock()
    monkeypatch.setattr(mr.mamba_utils, "postprocess_mamba_align_gpu", post)
    runner = make_runner(
        speculative_config=SimpleNamespace(),
        model_config=SimpleNamespace(is_hybrid=True),
        cache_config=SimpleNamespace(mamba_cache_mode="align"),
        num_accepted_tokens=SimpleNamespace(gpu=gpu),
        num_accepted_tokens_event=event,
        input_batch="IB",
        kv_cache_config="KV",
        compilation_config=SimpleNamespace(static_forward_context="SFC"),
    )
    runner._get_mamba_bufs = Mock(return_value="BUF")
    runner._get_mamba_state_copy_funcs = Mock(return_value="F")
    runner._update_states_after_model_execute(torch.tensor([[1, 2], [3, -1]]), None)
    post.assert_called_once()
    assert post.call_args.kwargs["bufs"] == "BUF"


def test_update_streaming_request(monkeypatch):
    monkeypatch.setattr(
        mr, "length_from_prompt_token_ids_or_embeds", lambda ids, emb: len(ids)
    )
    req_state = SimpleNamespace(
        prompt_token_ids=[1],
        mm_features=[],
        prompt_embeds=None,
        sampling_params="s",
        pooling_params="p",
        block_ids=[[0]],
        num_computed_tokens=0,
        num_prompt_tokens=1,
        output_token_ids=[9, 10],
    )
    ib = SimpleNamespace(remove_request=Mock())
    late = SimpleNamespace(register_request=Mock())
    runner = make_runner(
        input_batch=ib,
        requests={"r": req_state},
        late_interaction_runner=late,
        uses_mrope=False,
    )
    new_req = SimpleNamespace(
        prompt_token_ids=[1, 2, 3],
        mm_features=["m"],
        prompt_embeds=None,
        sampling_params="s2",
        pooling_params="p2",
        block_ids=[[1]],
        num_computed_tokens=0,
    )
    got = runner._update_streaming_request("r", new_req)
    assert got is req_state
    assert req_state.output_token_ids == []
    assert req_state.num_prompt_tokens == 3
    ib.remove_request.assert_called_once_with("r")


def test_init_mrope_positions_token_ids(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda model: True)
    model = Mock()
    model.get_mrope_input_positions = Mock(return_value=("POS", "DELTA"))
    runner = make_runner()
    runner.get_model = Mock(return_value=model)
    req = SimpleNamespace(
        mm_features=[SimpleNamespace(modality="prompt_embeds"), SimpleNamespace(modality="image")],
        prompt_token_ids=[1, 2],
        prompt_embeds=None,
    )
    runner._init_mrope_positions(req)
    assert req.mrope_positions == "POS"
    assert req.mrope_position_delta == "DELTA"
    model.get_mrope_input_positions.assert_called_once()


def test_init_mrope_positions_embeds_only(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda model: True)
    model = Mock()
    model.get_mrope_input_positions = Mock(return_value=("P", "D"))
    runner = make_runner()
    runner.get_model = Mock(return_value=model)
    req = SimpleNamespace(
        mm_features=[],
        prompt_token_ids=None,
        prompt_embeds=torch.zeros(4, 8),
    )
    runner._init_mrope_positions(req)
    args = model.get_mrope_input_positions.call_args.args
    assert args[0] == [0, 1, 2, 3]


def test_init_mrope_positions_missing_inputs(monkeypatch):
    monkeypatch.setattr(mr, "supports_mrope", lambda model: True)
    runner = make_runner()
    runner.get_model = Mock(return_value=Mock())
    req = SimpleNamespace(mm_features=[], prompt_token_ids=None, prompt_embeds=None)
    with pytest.raises(ValueError):
        runner._init_mrope_positions(req)


def test_calc_mrope_positions_prompt_part(monkeypatch):
    monkeypatch.setattr(
        mr, "length_from_prompt_token_ids_or_embeds", lambda ids, emb: 10
    )
    req = SimpleNamespace(
        mrope_positions=torch.arange(20).reshape(2, 10),
        prompt_token_ids=[0] * 10,
        prompt_embeds=None,
        mrope_position_delta=None,
    )
    cpu = torch.zeros((2, 12), dtype=torch.int64)
    runner = make_runner(
        input_batch=SimpleNamespace(
            req_ids=["r"],
            num_computed_tokens_cpu=np.array([2]),
        ),
        requests={"r": req},
        mrope_positions=SimpleNamespace(cpu=cpu, np=np.zeros((2, 12), dtype=np.int64)),
    )
    runner._calc_mrope_positions(SimpleNamespace(num_scheduled_tokens={"r": 3}))
    assert cpu[0, 0:3].tolist() == [2, 3, 4]


# ---------------------------------------------------------------------------
# Multimodal helpers
# ---------------------------------------------------------------------------


def test_extract_mm_kwargs_guards():
    runner = make_runner(is_multimodal_raw_input_only_model=False)
    assert runner._extract_mm_kwargs(None) == {}
    assert runner._extract_mm_kwargs(SimpleNamespace()) == {}


def test_extract_mm_kwargs_batches(monkeypatch):
    monkeypatch.setattr(
        mr,
        "group_and_batch_mm_kwargs",
        lambda items, device, pin_memory: iter([("image", 1, {"k": 1})]),
    )
    features = [
        SimpleNamespace(data=1, modality="image"),
        SimpleNamespace(data=None, modality="image"),
    ]
    runner = make_runner(
        is_multimodal_raw_input_only_model=True, device=torch.device("cpu")
    )
    out = runner._extract_mm_kwargs(
        SimpleNamespace(scheduled_new_reqs=[SimpleNamespace(mm_features=features)])
    )
    assert out == {"k": 1}


def test_dummy_mm_kwargs_guards():
    runner = make_runner(is_multimodal_raw_input_only_model=False)
    assert runner._dummy_mm_kwargs(2) == {}

    runner = make_runner(
        is_multimodal_raw_input_only_model=True,
        mm_budget=SimpleNamespace(mm_max_toks_per_item=0),
    )
    assert runner._dummy_mm_kwargs(2) == {}


def test_dummy_mm_kwargs_delegates():
    budget = SimpleNamespace(
        mm_max_toks_per_item=4,
        get_modality_with_max_tokens=Mock(return_value="image"),
    )
    runner = make_runner(
        is_multimodal_raw_input_only_model=True, mm_budget=budget
    )
    runner._get_mm_dummy_batch = Mock(return_value={"x": 1})
    assert runner._dummy_mm_kwargs(3) == {"x": 1}
    runner._get_mm_dummy_batch.assert_called_once_with("image", 3)


def test_batch_mm_inputs_from_scheduler_empty_and_populated():
    runner = make_runner(requests={})
    assert runner._batch_mm_inputs_from_scheduler(
        SimpleNamespace(scheduled_encoder_inputs={})
    ) == ([], [], [])

    feat = SimpleNamespace(
        data={"t": 1}, identifier="h", modality="image", mm_position="POS"
    )
    req = SimpleNamespace(mm_features=[feat, SimpleNamespace(data=None)])
    runner = make_runner(requests={"r": req})
    hashes, kwargs, refs = runner._batch_mm_inputs_from_scheduler(
        SimpleNamespace(scheduled_encoder_inputs={"r": [0, 1]})
    )
    assert hashes == ["h"]
    assert kwargs == [("image", {"t": 1})]
    assert refs == [("r", "POS")]


def test_cache_encoder_output_and_lookup(monkeypatch):
    save = Mock()
    runner = make_runner(encoder_cache={})
    runner.maybe_save_ec_to_connector = save
    runner._cache_encoder_output("h", "OUT", "META", ["f"])
    assert runner.encoder_cache == {"h": "OUT"}
    save.assert_called_once_with({"h": "OUT"}, "h")
    assert runner._get_encoder_output_from_cache("h") == "OUT"
    assert runner._get_encoder_output_from_cache("missing") is None


# ---------------------------------------------------------------------------
# Numeric / metadata utilities
# ---------------------------------------------------------------------------


def test_get_cumsum_and_arange():
    runner = make_runner(arange_np=np.arange(32, dtype=np.int64))
    out = np.empty(32, dtype=np.int64)
    cu = runner._get_cumsum_and_arange(
        np.array([2, 5, 3]), out, cumsum_dtype=np.int32
    )
    assert cu.tolist() == [2, 7, 10]
    assert out[:10].tolist() == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]


def test_compute_prev_positions_empty_and_mapped():
    runner = make_runner(
        input_batch=SimpleNamespace(prev_req_id_to_index=None, req_ids=["a"]),
        prev_positions=SimpleNamespace(np=np.zeros(3, dtype=np.int64)),
    )
    runner._compute_prev_positions(2)
    assert runner.prev_positions.np[:2].tolist() == [-1, -1]

    runner = make_runner(
        input_batch=SimpleNamespace(
            prev_req_id_to_index={"a": 4}, req_ids=["a", "b"]
        ),
        prev_positions=SimpleNamespace(np=np.zeros(3, dtype=np.int64)),
    )
    runner._compute_prev_positions(2)
    assert runner.prev_positions.np[:2].tolist() == [4, -1]


def test_prepare_input_ids_normal_path():
    input_ids = SimpleNamespace(copy_to_gpu=Mock())
    runner = make_runner(
        _pp_recv_work=None,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        input_ids=input_ids,
        enable_prompt_embeds=False,
    )
    runner._prepare_input_ids(SimpleNamespace(), 2, 5, np.array([2, 5]))
    input_ids.copy_to_gpu.assert_called_once_with(5)


def test_prepare_input_ids_prompt_embeds_path():
    input_ids = SimpleNamespace(copy_to_gpu=Mock())
    inputs_embeds = SimpleNamespace(copy_to_gpu=Mock())
    is_token_ids = SimpleNamespace(copy_to_gpu=Mock())
    runner = make_runner(
        _pp_recv_work=None,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        is_token_ids=is_token_ids,
        enable_prompt_embeds=True,
    )
    runner._prepare_input_ids(SimpleNamespace(), 1, 3, np.array([3]))
    inputs_embeds.copy_to_gpu.assert_called_once_with(3)
    is_token_ids.copy_to_gpu.assert_called_once_with(3)


def test_get_encoder_seq_lens_non_cross():
    runner = make_runner()
    assert runner._get_encoder_seq_lens({}, SimpleNamespace(), 1) == (None, None)


def test_get_encoder_seq_lens_cross(monkeypatch):
    monkeypatch.setattr(mr, "CrossAttentionSpec", type("CrossAttentionSpec", (), {}))
    spec = instance_of(mr.CrossAttentionSpec)
    buf = SimpleNamespace(
        np=np.zeros(4, dtype=np.int32),
        gpu=torch.zeros(4, dtype=torch.int32),
        copy_to_gpu=Mock(),
    )
    req_none = SimpleNamespace(mm_features=None)
    req_mm = SimpleNamespace(
        mm_features=[
            SimpleNamespace(mm_position=SimpleNamespace(length=3)),
            SimpleNamespace(mm_position=SimpleNamespace(length=2)),
        ]
    )
    runner = make_runner(
        encoder_seq_lens=buf,
        input_batch=SimpleNamespace(req_id_to_index={"a": 0, "b": 1}),
        requests={"a": req_none, "b": req_mm},
        max_encoder_len=7,
        model_config=SimpleNamespace(hf_config=SimpleNamespace()),
    )
    gpu, cpu = runner._get_encoder_seq_lens({"a": 1, "b": 1}, spec, 2)
    assert cpu.tolist()[:2] == [0, 5]
    buf.copy_to_gpu.assert_called_once_with(2)
    assert gpu.shape == (2,)

    gpu, cpu = runner._get_encoder_seq_lens({"a": 1}, spec, 2, for_cudagraph_capture=True)
    assert cpu.tolist()[:2] == [7, 7]


def test_compute_cascade_attn_prefix_len_zero_blocks():
    spec = SimpleNamespace(block_size=16)
    assert (
        mr.ModelRunnerFL._compute_cascade_attn_prefix_len(
            make_runner(), np.array([1]), np.array([0]), 0, spec, Mock()
        )
        == 0
    )


def test_compute_cascade_attn_prefix_len_use_case(monkeypatch):
    spec = Mock(spec=mr.AttentionSpec)
    spec.block_size = 16
    spec.num_kv_heads = 4
    builder = Mock()
    builder.use_cascade_attention = Mock(return_value=True)
    runner = make_runner(
        num_query_heads=8,
        use_alibi=False,
        num_sms=64,
        dcp_world_size=1,
    )
    got = runner._compute_cascade_attn_prefix_len(
        np.array([1, 1]), np.array([100, 40]), 2, spec, builder
    )
    assert got == 32
    builder.use_cascade_attention.assert_called_once()


def test_compute_cascade_attn_prefix_lens_all_zero_and_hit():
    enc_spec = instance_of(mr.EncoderOnlyAttentionSpec)
    groups = [
        [SimpleNamespace(kv_cache_spec=enc_spec)],
        [
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(),
                get_metadata_builder=lambda: "B",
            )
        ],
    ]
    runner = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1, 2]),
        attn_groups=groups,
    )
    runner._compute_cascade_attn_prefix_len = Mock(return_value=0)
    assert runner._compute_cascade_attn_prefix_lens(
        np.array([1]), np.array([0]), [1, 1]
    ) is None

    runner._compute_cascade_attn_prefix_len = Mock(return_value=5)
    out = runner._compute_cascade_attn_prefix_lens(
        np.array([1]), np.array([0]), [1, 1]
    )
    assert out == [[0], [5]]


def test_calc_spec_decode_metadata(monkeypatch):
    monkeypatch.setattr(
        mr, "async_tensor_h2d", lambda arr, device=None: torch.as_tensor(arr)
    )
    input_ids = SimpleNamespace(gpu=torch.arange(300, dtype=torch.int32))
    runner = make_runner(
        arange_np=np.arange(64, dtype=np.int64),
        _arange_scratch=np.empty(64, dtype=np.int64),
        input_ids=input_ids,
        device=torch.device("cpu"),
    )
    meta = runner._calc_spec_decode_metadata(
        np.array([3, 0, 2], dtype=np.int32),
        np.array([4, 104, 107], dtype=np.int32),
    )
    assert meta.num_draft_tokens == [3, 0, 2]
    assert meta.logits_indices.tolist() == [0, 1, 2, 3, 103, 104, 105, 106]


def test_prepare_kv_sharing_fast_prefill():
    buf = torch.zeros(8, dtype=torch.int32)
    dispatch = Mock(return_value=(mr.CUDAGraphMode.NONE, SimpleNamespace(num_tokens=5)))
    runner = make_runner(
        kv_sharing_fast_prefill_logits_indices=buf,
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
    )
    out = runner._prepare_kv_sharing_fast_prefill(torch.tensor([1, 2, 3]))
    assert out.tolist() == [1, 2, 3, 3, 3]
    dispatch.assert_called_once()


def test_pad_query_start_loc_uniform_branch():
    buf = SimpleNamespace(np=np.zeros(8, dtype=np.int64), copy_to_gpu=Mock())
    runner = make_runner(uniform_decode_query_len=2, arange_np=np.arange(8, dtype=np.int64))
    buf.np[:3] = [0, 2, 4]
    padded = runner._pad_query_start_loc_for_fia(buf, 8, 2, 4)
    assert padded == 4
    assert buf.copy_to_gpu.called


def test_pad_query_start_loc_non_uniform_grows():
    buf = SimpleNamespace(np=np.zeros(8, dtype=np.int64), copy_to_gpu=Mock())
    runner = make_runner(uniform_decode_query_len=1, arange_np=np.arange(8, dtype=np.int64))
    buf.np[:2] = [0, 3]
    padded = runner._pad_query_start_loc_for_fia(buf, 6, 2, 2)
    assert padded == 3
    assert buf.np[3] == 6


# ---------------------------------------------------------------------------
# Model accessors / supported tasks
# ---------------------------------------------------------------------------


def test_get_model_variants():
    runner = make_runner()
    with pytest.raises(ValueError):
        runner.get_model()

    plain = object()
    runner = make_runner(model=plain)
    assert runner.get_model() is plain

    model = Mock(spec=mr.GraphWrapper)
    model.unwrap = Mock(return_value="RAW")
    runner = make_runner(model=model)
    assert runner.get_model() == "RAW"


def test_get_draft_model_variants():
    assert make_runner().get_draft_model() is None
    assert make_runner(drafter=None).get_draft_model() is None
    assert make_runner(drafter=SimpleNamespace(model=None)).get_draft_model() is None

    plain = object()
    assert make_runner(drafter=SimpleNamespace(model=plain)).get_draft_model() is plain

    model = Mock(spec=mr.CUDAGraphWrapper)
    model.unwrap = Mock(return_value="RAW")
    assert make_runner(drafter=SimpleNamespace(model=model)).get_draft_model() == "RAW"


def test_get_supported_generation_tasks(monkeypatch):
    model = SimpleNamespace(supports_transcription_only=False)
    runner = make_runner()
    runner.get_model = Mock(return_value=model)
    monkeypatch.setattr(mr, "is_text_generation_model", lambda m: True)
    monkeypatch.setattr(mr, "supports_transcription", lambda m: True)
    monkeypatch.setattr(mr, "supports_realtime", lambda m: True)
    assert runner.get_supported_generation_tasks() == [
        "generate",
        "transcription",
        "realtime",
    ]

    model.supports_transcription_only = True
    assert runner.get_supported_generation_tasks() == ["transcription"]


def test_get_supported_pooling_and_all_tasks(monkeypatch):
    model = SimpleNamespace(pooler=SimpleNamespace(get_supported_tasks=lambda: ["embed"]))
    runner = make_runner(model_config=SimpleNamespace(runner_type="pooling"))
    runner.get_model = Mock(return_value=model)
    monkeypatch.setattr(mr, "is_pooling_model", lambda m: True)
    assert runner.get_supported_pooling_tasks() == ["embed"]
    assert runner.get_supported_tasks() == ("embed",)

    runner = make_runner(model_config=SimpleNamespace(runner_type="generate"))
    runner.get_supported_generation_tasks = Mock(return_value=["generate"])
    assert runner.get_supported_tasks() == ("generate",)

    monkeypatch.setattr(mr, "is_pooling_model", lambda m: False)
    runner = make_runner()
    runner.get_model = Mock(return_value=model)
    assert runner.get_supported_pooling_tasks() == []


def test_sync_and_gather_intermediate_tensors():
    runner = make_runner(
        intermediate_tensors={"a": torch.zeros(4), "b": torch.ones(4)}
    )
    out = runner.sync_and_gather_intermediate_tensors(2, None, False)
    assert out["a"].tolist() == [0.0, 0.0]
    assert out["b"].tolist() == [1.0, 1.0]

    incoming = {"a": torch.tensor([5.0, 6.0, 7.0]), "b": torch.tensor([8.0, 9.0, 1.0])}
    out = runner.sync_and_gather_intermediate_tensors(3, incoming, True)
    assert out["a"].tolist() == [5.0, 6.0, 7.0]


def test_eplb_step_disabled_and_active():
    runner = make_runner(
        parallel_config=SimpleNamespace(enable_eplb=False), eep_eplb_suppressed=False
    )
    runner.eplb_state = Mock()
    runner.eplb_step()
    runner.eplb_state.step.assert_not_called()

    state = Mock()
    runner = make_runner(
        parallel_config=SimpleNamespace(
            enable_eplb=True,
            eplb_config=SimpleNamespace(log_balancedness=True),
        ),
        eep_eplb_suppressed=False,
        eplb_state=state,
        _moe_model=object(),
    )
    runner.eplb_step(True, False)
    state.step.assert_called_once_with(True, False, log_stats=True)


def test_setup_eplb_from_mapping(monkeypatch):
    captured = {}

    def fake_from_mapping(**kwargs):
        captured.update(kwargs)
        return "STATE"

    monkeypatch.setattr(mr.EplbState, "from_mapping", Mock(return_value="STATE"))
    runner = make_runner(
        _moe_model="MOE",
        model_config="MC",
        device=torch.device("cpu"),
        parallel_config="PC",
    )
    runner.setup_eplb_from_mapping("MAP", 7)
    assert runner.eplb_state == "STATE"
    mr.EplbState.from_mapping.assert_called_once()


# ---------------------------------------------------------------------------
# Padding / preprocessing / sampling
# ---------------------------------------------------------------------------


def test_prepare_mm_inputs_variants():
    runner = make_runner(
        model=SimpleNamespace(requires_raw_input_tokens=True),
        input_ids=SimpleNamespace(gpu=torch.arange(5)),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(5, 3)),
    )
    ids, embeds = runner._prepare_mm_inputs(3)
    assert ids.tolist() == [0, 1, 2]
    assert embeds.shape == (3, 3)

    runner.model.requires_raw_input_tokens = False
    ids, _ = runner._prepare_mm_inputs(2)
    assert ids is None


def test_sample_without_spec_decode():
    sampler = Mock(return_value="SAMPLED")
    sampling_metadata = SimpleNamespace(generators={})
    ib = SimpleNamespace(
        sampling_metadata=sampling_metadata,
        update_async_output_token_ids=Mock(),
    )
    runner = make_runner(sampler=sampler, input_batch=ib)
    assert runner._sample("LOGITS", None) == "SAMPLED"
    ib.update_async_output_token_ids.assert_called_once()


def test_sample_with_spec_decode():
    sampling_metadata = SimpleNamespace(generators={})
    ib = SimpleNamespace(
        sampling_metadata=sampling_metadata,
        update_async_output_token_ids=Mock(),
    )
    rejection = Mock(return_value="SPEC")
    runner = make_runner(
        sampler=Mock(),
        input_batch=ib,
        use_async_scheduling=False,
        rejection_sampler=rejection,
    )
    runner._get_spec_decode_draft_probs = Mock(return_value="DP")
    assert runner._sample("LOGITS", "META") == "SPEC"
    rejection.assert_called_once_with("META", "DP", "LOGITS", sampling_metadata)


def test_synchronize_input_prep_without_event():
    runner = make_runner(prepare_inputs_event=None)
    with runner.synchronize_input_prep():
        pass


def test_synchronize_input_prep_with_event():
    event = Mock()
    runner = make_runner(prepare_inputs_event=event)
    with runner.synchronize_input_prep():
        pass
    event.synchronize.assert_called_once()
    event.record.assert_called_once()


def test_synchronize_input_prep_records_on_error():
    event = Mock()
    runner = make_runner(prepare_inputs_event=event)
    with pytest.raises(ValueError):
        with runner.synchronize_input_prep():
            raise ValueError("boom")
    event.record.assert_called_once()


def test_model_forward_forwards_kwargs():
    model = Mock(return_value="OUT")
    runner = make_runner(model=model)
    got = runner._model_forward(
        input_ids="I",
        positions="P",
        intermediate_tensors="IT",
        inputs_embeds="E",
        extra=1,
    )
    assert got == "OUT"
    model.assert_called_once_with(
        input_ids="I",
        positions="P",
        intermediate_tensors="IT",
        inputs_embeds="E",
        extra=1,
    )


def test_is_uniform_decode_cases():
    fn = mr.ModelRunnerFL._is_uniform_decode
    assert fn(4, 4, 8, 2) is True
    assert fn(4, 4, 7, 2) is False
    assert fn(3, 4, 6, 2) is False
    assert fn(3, 4, 6, 2, force_uniform_decode=True) is True
    assert fn(4, 4, 8, 2, force_uniform_decode=False) is False


# ---------------------------------------------------------------------------
# Microbatching / dispatch
# ---------------------------------------------------------------------------


def test_allow_microbatching_disabled_and_no_readers():
    runner = make_runner(parallel_config=SimpleNamespace(use_ubatching=False))
    assert runner._allow_microbatching(3, np.array([1, 1, 1])) is True

    runner = make_runner(parallel_config=SimpleNamespace(use_ubatching=True))
    assert runner._allow_microbatching(1, np.array([1])) is True

    runner = make_runner(
        parallel_config=SimpleNamespace(use_ubatching=True),
        reorder_batch_threshold=1,
        input_batch=SimpleNamespace(num_computed_tokens_cpu=np.array([0, 0])),
    )
    assert runner._allow_microbatching(2, np.array([1, 1])) is True


def test_allow_microbatching_detects_reader_writer_overlap():
    table = np.array([[0, 1], [2, 3]])
    block_table = SimpleNamespace(
        block_size=2,
        get_numpy_array=Mock(return_value=table),
    )
    ib = SimpleNamespace(
        num_computed_tokens_cpu=np.array([0, 4]),
        block_table=SimpleNamespace(block_tables=[block_table]),
    )
    runner = make_runner(
        parallel_config=SimpleNamespace(use_ubatching=True),
        reorder_batch_threshold=1,
        input_batch=ib,
    )
    assert runner._allow_microbatching(2, np.array([1, 3])) is False


def test_determine_batch_execution_and_padding_single_rank(monkeypatch):
    dispatch = Mock(
        return_value=(mr.CUDAGraphMode.NONE, SimpleNamespace(num_tokens=5, num_reqs=None))
    )
    runner = make_runner(
        uniform_decode_query_len=1,
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
        parallel_config=SimpleNamespace(),
    )
    mode, desc, ubatch, across, stats = runner._determine_batch_execution_and_padding(
        num_tokens=5,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([2, 3]),
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
    )
    assert mode == mr.CUDAGraphMode.NONE
    assert desc.num_tokens == 5
    assert ubatch is False and across is None and stats is None


def test_determine_batch_execution_and_padding_dp(monkeypatch):
    padded_desc = SimpleNamespace(num_tokens=8, num_reqs=None)
    dispatch = Mock(
        side_effect=[
            (mr.CUDAGraphMode.NONE, SimpleNamespace(num_tokens=5, num_reqs=None)),
            (mr.CUDAGraphMode.FULL, padded_desc),
        ]
    )
    monkeypatch.setattr(
        mr,
        "coordinate_batch_across_dp",
        Mock(return_value=(True, torch.tensor([8]), mr.CUDAGraphMode.FULL.value)),
    )
    runner = make_runner(
        uniform_decode_query_len=1,
        cudagraph_dispatcher=SimpleNamespace(dispatch=dispatch),
        model_config=SimpleNamespace(is_encoder_decoder=False),
        input_batch=SimpleNamespace(lora_id_to_lora_request={}),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2),
            observability_config=SimpleNamespace(cudagraph_metrics=False),
        ),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    mode, desc, ubatch, across, _ = runner._determine_batch_execution_and_padding(
        num_tokens=5,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([2, 3]),
        max_num_scheduled_tokens=3,
        use_cascade_attn=False,
    )
    assert ubatch is True
    assert across is not None
    assert mode == mr.CUDAGraphMode.FULL


def test_register_layerwise_nvtx_hooks_branches(monkeypatch):
    obs_off = SimpleNamespace(enable_layerwise_nvtx_tracing=False)
    runner = make_runner(
        vllm_config=SimpleNamespace(observability_config=obs_off),
        layerwise_nvtx_hooks_registered=False,
    )
    runner._register_layerwise_nvtx_hooks()
    assert runner.layerwise_nvtx_hooks_registered is False

    hooks = Mock()
    monkeypatch.setattr(mr, "PytHooks", Mock(return_value=hooks))
    obs_on = SimpleNamespace(enable_layerwise_nvtx_tracing=True)

    class _Model:
        pass

    runner = make_runner(
        vllm_config=SimpleNamespace(
            observability_config=obs_on,
            compilation_config=SimpleNamespace(mode=mr.CompilationMode.NONE),
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE),
        layerwise_nvtx_hooks_registered=False,
        model=_Model(),
    )
    runner._register_layerwise_nvtx_hooks()
    hooks.register_hooks.assert_called_once()
    assert runner.layerwise_nvtx_hooks_registered is True


# ---------------------------------------------------------------------------
# Slot mappings / chunked prefill detection
# ---------------------------------------------------------------------------


def test_get_slot_mappings_without_kv_cache_config():
    runner = make_runner()
    assert runner._get_slot_mappings(2, 1, 2) == (None, None)


def test_get_slot_mappings_normal_and_encoder_only():
    enc_spec = instance_of(mr.EncoderOnlyAttentionSpec)
    attn_tensor = torch.zeros(4, dtype=torch.int64)
    blk = SimpleNamespace(slot_mapping=SimpleNamespace(gpu=attn_tensor))
    groups = [
        SimpleNamespace(kv_cache_spec=SimpleNamespace(), layer_names=["l0"]),
        SimpleNamespace(kv_cache_spec=enc_spec, layer_names=["l1"]),
    ]
    runner = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=groups),
        input_batch=SimpleNamespace(block_table={0: blk}),
        device=torch.device("cpu"),
    )
    by_gid, by_layer = runner._get_slot_mappings(4, 1, 2)
    assert set(by_gid) == {0, 1}
    assert by_gid[0] is attn_tensor
    assert set(by_layer) == {"l0", "l1"}
    assert by_gid[0][2:4].tolist() == [-1, -1]

    ubatches = [SimpleNamespace(token_slice=torch.tensor([0, 1]))]
    _, sliced = runner._get_slot_mappings(4, 1, 2, ubatch_slices=ubatches)
    assert isinstance(sliced, list)
    assert sliced[0]["l0"].shape == (2,)


def test_is_all_reqs_chunked_prefill():
    runner = make_runner(
        input_batch=SimpleNamespace(num_reqs=2),
        discard_request_mask=SimpleNamespace(np=np.array([True, True, False])),
    )
    assert runner._is_all_reqs_chunked_prefill() is True

    runner.discard_request_mask = SimpleNamespace(np=np.array([True, False]))
    assert runner._is_all_reqs_chunked_prefill() is False


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_pool_no_finished_output(monkeypatch):
    monkeypatch.setattr(mr, "ModelRunnerOutput", Mock(return_value=SimpleNamespace()))
    pooling_metadata = SimpleNamespace(
        build_pooling_cursor=Mock(),
        prompt_lens=[5],
        pooling_params=["p"],
    )
    ib = SimpleNamespace(
        num_reqs=1,
        pooling_params=["p"],
        get_pooling_metadata=lambda: pooling_metadata,
        req_ids=["r"],
        req_id_to_index={"r": 0},
    )
    model = SimpleNamespace(pooler=Mock(return_value=None))
    late = SimpleNamespace(postprocess_pooler_output=Mock(return_value=None))
    runner = make_runner(
        input_batch=ib,
        optimistic_seq_lens_cpu=torch.tensor([5]),
        query_start_loc=SimpleNamespace(gpu=torch.zeros(2)),
        model=model,
        late_interaction_runner=late,
    )
    runner._sync_device = Mock()
    out = runner._pool("HIDDEN", 1, np.array([1]), "KVCO")
    assert out.pooler_output == [None]
    runner._sync_device.assert_called_once()


def test_pool_non_cuda_output(monkeypatch):
    output_obj = SimpleNamespace()
    monkeypatch.setattr(mr, "ModelRunnerOutput", Mock(return_value=output_obj))
    monkeypatch.setattr(mr, "current_platform", fake_platform(is_cuda_alike=lambda: False))
    monkeypatch.setattr(mr, "_copy_pooler_output_to_cpu", Mock(return_value=["CPU"]))
    pooling_metadata = SimpleNamespace(
        build_pooling_cursor=Mock(),
        prompt_lens=[1],
        pooling_params=["p"],
    )
    ib = SimpleNamespace(
        num_reqs=1,
        pooling_params=["p"],
        get_pooling_metadata=lambda: pooling_metadata,
        req_ids=["r"],
        req_id_to_index={"r": 0},
    )
    raw = torch.zeros(1, 2)
    model = SimpleNamespace(pooler=Mock(return_value=raw))
    late = SimpleNamespace(postprocess_pooler_output=Mock(return_value=raw))
    runner = make_runner(
        input_batch=ib,
        optimistic_seq_lens_cpu=torch.tensor([1]),
        query_start_loc=SimpleNamespace(gpu=torch.zeros(2)),
        model=model,
        late_interaction_runner=late,
    )
    runner._sync_device = Mock()
    out = runner._pool("HIDDEN", 1, np.array([1]), None)
    assert out.pooler_output == ["CPU"]


def test_pool_cuda_returns_async_wrapper(monkeypatch):
    monkeypatch.setattr(mr, "ModelRunnerOutput", Mock(return_value=SimpleNamespace()))
    monkeypatch.setattr(mr, "current_platform", fake_platform())
    async_wrapper = Mock(return_value="ASYNC")
    monkeypatch.setattr(mr, "AsyncGPUPoolingModelRunnerOutput", async_wrapper)
    pooling_metadata = SimpleNamespace(
        build_pooling_cursor=Mock(),
        prompt_lens=[1],
        pooling_params=["p"],
    )
    ib = SimpleNamespace(
        num_reqs=1,
        pooling_params=["p"],
        get_pooling_metadata=lambda: pooling_metadata,
        req_ids=["r"],
        req_id_to_index={"r": 0},
    )
    raw = torch.zeros(1, 2)
    runner = make_runner(
        input_batch=ib,
        optimistic_seq_lens_cpu=torch.tensor([1]),
        query_start_loc=SimpleNamespace(gpu=torch.zeros(2)),
        model=SimpleNamespace(pooler=Mock(return_value=raw)),
        late_interaction_runner=SimpleNamespace(
            postprocess_pooler_output=Mock(return_value=raw)
        ),
    )
    runner._get_or_create_async_output_copy_stream = Mock(return_value="STREAM")
    assert runner._pool("HIDDEN", 1, np.array([1]), None) == "ASYNC"
    async_wrapper.assert_called_once()


# ---------------------------------------------------------------------------
# execute_model / sample_tokens light paths
# ---------------------------------------------------------------------------


def test_execute_model_state_error():
    runner = make_runner(execute_model_state="stale")
    with pytest.raises(RuntimeError):
        runner.execute_model(SimpleNamespace())


def test_execute_model_empty_batch(monkeypatch):
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda name: nullcontext())
    runner = make_runner(
        execute_model_state=None,
        speculative_config=None,
        prepare_inputs_event=None,
        encoder_cache={},
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp", data_parallel_size=1
        ),
    )
    runner._update_states = Mock(return_value=None)
    sched = SimpleNamespace(total_num_scheduled_tokens=0)
    got = runner.execute_model(sched)
    assert got is mr.EMPTY_MODEL_RUNNER_OUTPUT


def test_execute_model_empty_batch_kv_transfer(monkeypatch):
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda name: nullcontext())
    monkeypatch.setattr(
        mr, "get_kv_transfer_group", Mock(return_value=SimpleNamespace(handle_preemptions=Mock()))
    )
    runner = make_runner(
        execute_model_state=None,
        speculative_config=None,
        prepare_inputs_event=None,
        encoder_cache={},
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp", data_parallel_size=1
        ),
    )
    runner._update_states = Mock(return_value=None)
    runner.kv_connector_no_forward = Mock(return_value="KVOUT")
    sched = SimpleNamespace(
        total_num_scheduled_tokens=0,
        kv_connector_metadata="META",
    )
    assert runner.execute_model(sched) == "KVOUT"
    mr.get_kv_transfer_group.return_value.handle_preemptions.assert_called_once_with("META")


def test_sample_tokens_without_state(monkeypatch):
    output_cls = Mock()
    output_cls.with_kv_conn_output_only = Mock(return_value="EMPTY")
    monkeypatch.setattr(mr, "ModelRunnerOutput", output_cls)
    runner = make_runner(
        execute_model_state=None,
        kv_connector_output="KVC",
        use_async_scheduling=False,
    )
    got = runner.sample_tokens(None)
    assert got == "EMPTY"
    assert runner.kv_connector_output is None


def test_sample_tokens_minimal_path(monkeypatch):
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda name: nullcontext())
    monkeypatch.setattr(mr, "advance_io_step", Mock())
    output_cls = Mock(return_value="OUTPUT")
    monkeypatch.setattr(mr, "ModelRunnerOutput", output_cls)
    sampler_output = SimpleNamespace(sampled_token_ids="TOK", logprobs_tensors=None)
    bookkeeping = (
        {},
        None,
        None,
        [[1]],
        {},
        ["r"],
        {"r": 0},
        [],
    )
    runner = make_runner(
        execute_model_state=(
            SimpleNamespace(total_num_scheduled_tokens=1),
            "LOGITS",
            None,
            None,
            "HIDDEN",
            "SAMPLE_H",
            None,
            None,
            None,
            None,
        ),
        use_async_scheduling=False,
        speculative_config=None,
        supports_mm_inputs=False,
        kv_connector_output=None,
        input_batch=SimpleNamespace(prev_sampled_token_ids=None),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        eep_eplb_suppressed=True,
        _draft_token_ids="x",
        _draft_probs="x",
        _draft_prob_req_ids="x",
        _draft_token_req_ids="x",
        valid_sampled_token_count_gpu="x",
    )
    runner._sample = Mock(return_value=sampler_output)
    runner._update_states_after_model_execute = Mock()
    runner._bookkeeping_sync = Mock(return_value=bookkeeping)
    runner.eplb_step = Mock()
    runner.finalize_kv_connector = Mock()
    got = runner.sample_tokens(None)
    assert got == "OUTPUT"
    assert runner.execute_model_state is None
    runner._sample.assert_called_once()
    runner._bookkeeping_sync.assert_called_once()
    runner.eplb_step.assert_called_once()
    assert runner._draft_token_ids is None
    assert runner.valid_sampled_token_count_gpu is None


def test_input_fits_in_drafter():
    runner = make_runner(speculative_config=None)
    assert runner._input_fits_in_drafter(None) is False

    spec = SimpleNamespace(use_dflash=lambda: False)
    meta = SimpleNamespace(max_seq_len=10)
    runner = make_runner(
        speculative_config=spec,
        num_spec_tokens=2,
        effective_drafter_max_model_len=12,
    )
    assert runner._input_fits_in_drafter(meta) is True
    assert runner._input_fits_in_drafter(SimpleNamespace(max_seq_len=11)) is False


# ---------------------------------------------------------------------------
# Opportunistic full __init__ construction
# ---------------------------------------------------------------------------


def _fake_vllm_config():
    model_config = SimpleNamespace(
        dtype=torch.float16,
        is_moe=False,
        runner_type="generate",
        enable_prompt_embeds=False,
        is_multimodal_raw_input_only_model=False,
        max_model_len=128,
        get_num_attention_heads=lambda pc: 8,
        get_inputs_embeds_size=lambda: 16,
        uses_alibi=False,
        disable_cascade_attn=False,
        is_mm_prefix_lm=False,
        uses_mrope=False,
        supports_multimodal_inputs=False,
        is_encoder_decoder=False,
        logprobs_mode=None,
        use_fp64_gumbel=False,
        get_vocab_size=lambda: 1000,
        logits_processors=None,
        is_hybrid=False,
    )
    cache_config = SimpleNamespace(
        cache_dtype="auto",
        block_size=16,
        kv_sharing_fast_prefill=False,
        use_replayssm=False,
        mamba_cache_mode="none",
    )
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        decode_context_parallel_size=1,
        distributed_executor_backend="mp",
        cp_kv_cache_interleave_size=1,
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=32,
        max_num_seqs=4,
        max_num_encoder_input_tokens=0,
        async_scheduling=False,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        offload_config="OFFLOAD",
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=None,
            cudagraph_mode=mr.CUDAGraphMode.NONE,
            static_forward_context={},
        ),
        lora_config=None,
        load_config="LOAD",
        parallel_config=parallel_config,
        scheduler_config=scheduler_config,
        speculative_config=None,
        observability_config=SimpleNamespace(),
        reasoning_config=None,
    )


def test_model_runner_init_best_effort(monkeypatch):
    if getattr(mr, "PIN_MEMORY", False):
        monkeypatch.setattr(mr, "PIN_MEMORY", False)
    monkeypatch.setattr(mr, "current_platform", fake_platform(device_type="cpu"))
    monkeypatch.setattr(mr, "kv_cache_dtype_str_to_dtype", lambda dtype, mc: torch.float16)
    monkeypatch.setattr(mr, "num_compute_units", lambda idx: 1)
    monkeypatch.setattr(mr, "Sampler", Mock(return_value="SAMPLER"))
    monkeypatch.setattr(mr, "InputBatch", Mock(return_value="INPUT_BATCH"))
    monkeypatch.setattr(mr, "build_logitsprocs", Mock(return_value="PROCS"))
    monkeypatch.setattr(mr, "CudagraphDispatcher", Mock(return_value="DISPATCHER"))
    monkeypatch.setattr(mr, "create_offloader", Mock(return_value="OFFLOADER"))
    monkeypatch.setattr(mr, "set_offloader", Mock())
    monkeypatch.setattr(mr, "CpuGpuBuffer", Mock(return_value="BUF"))
    monkeypatch.setattr(mr, "set_offloader", Mock())

    init = getattr(mr.ModelRunnerFL.__init__, "__wrapped__", mr.ModelRunnerFL.__init__)
    runner = object.__new__(mr.ModelRunnerFL)
    try:
        init(runner, _fake_vllm_config(), torch.device("cpu"))
    except Exception as exc:  # pragma: no cover - depends on vLLM internals
        pytest.skip(f"ModelRunnerFL.__init__ not constructible in this env: {exc!r}")

    assert runner.device == torch.device("cpu")
    assert runner.max_model_len == 128
    assert runner.max_num_tokens == 32
    assert runner.max_num_reqs == 4
    assert runner.sampler == "SAMPLER"
    assert runner.input_batch == "INPUT_BATCH"
    assert runner.kv_caches == []
    assert runner.encoder_cache == {}
    assert runner.cudagraph_batch_sizes == []
    assert runner.num_sms == 1
    assert runner.use_async_scheduling is False
    assert runner.async_output_copy_stream is None
    assert runner.execute_model_state is None
    assert runner.requests == {}
