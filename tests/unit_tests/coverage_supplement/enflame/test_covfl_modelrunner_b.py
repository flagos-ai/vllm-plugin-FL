"""Coverage tests for the latter portion of vllm_fl/worker/model_runner.py.

Self-contained: pytest + unittest.mock + stdlib + torch/numpy.  No tests/conftest
dependency.  The heavy ``ModelRunnerFL.__init__`` is bypassed with
``object.__new__`` and only the attributes each method touches are populated.

The whole module skips cleanly when vLLM is not importable.
"""

from __future__ import annotations

import sys
import threading
import types
from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
vllm = pytest.importorskip("vllm")
M = pytest.importorskip("vllm_fl.worker.model_runner")

from unittest import mock  # noqa: E402


def make_runner(**attrs):
    r = object.__new__(M.ModelRunnerFL)
    r.__dict__.update(attrs)
    return r


# ---------------------------------------------------------------------------
# Module level helpers
# ---------------------------------------------------------------------------
def test_count_nans_per_row():
    logits = torch.tensor([[0.0, float("nan"), 1.0], [1.0, 2.0, 3.0]])
    counts = M.count_nans_per_row(logits)
    assert counts.tolist() == [1, 0]
    assert counts.dtype == torch.int32


def test_nans_to_dict_pads_and_skips():
    out = M.nans_to_dict([3, 1], {"a": 0, "b": 5})
    assert out == {"a": 3, "b": 0}


def test_copy_pooler_tensor_all_none():
    raw = torch.zeros(2, 3)
    out = M._copy_pooler_output_to_cpu(raw, [False, False])
    assert out == [None, None]


def test_copy_pooler_tensor_all_finished():
    raw = torch.arange(6).reshape(2, 3)
    out = M._copy_pooler_output_to_cpu(raw, [True, True])
    assert len(out) == 2
    assert torch.equal(out[1], torch.tensor([3, 4, 5]))


def test_copy_pooler_tensor_partial():
    raw = torch.arange(6).reshape(2, 3)
    out = M._copy_pooler_output_to_cpu(raw, [True, False])
    assert out[0] is not None
    assert out[1] is None


def test_copy_pooler_tensor_mismatch_raises():
    with pytest.raises(ValueError):
        M._copy_pooler_output_to_cpu(torch.zeros(3, 2), [True, True])


def test_copy_pooler_list_partial():
    raw = [torch.ones(2), None, torch.zeros(2)]
    out = M._copy_pooler_output_to_cpu(raw, [True, True, False])
    assert out[0] is not None
    assert out[1] is None
    assert out[2] is None


def test_copy_pooler_list_mismatch_raises():
    with pytest.raises(ValueError):
        M._copy_pooler_output_to_cpu([torch.ones(2)], [True, False])


# ---------------------------------------------------------------------------
# Async output wrappers
# ---------------------------------------------------------------------------
def _async_output(**attrs):
    r = object.__new__(M.AsyncGPUModelRunnerOutput)
    base = dict(
        _invalid_req_indices=[],
        _logprobs_tensors=None,
        _logprobs_tensors_cpu=None,
        _num_nans=None,
        _num_nans_cpu=None,
        _has_fault=None,
        async_copy_ready_event=mock.MagicMock(),
    )
    base.update(attrs)
    r.__dict__.update(base)
    return r


def test_async_get_output_maxgen1():
    out = SimpleNamespace(req_id_to_index={})
    r = _async_output(
        sampled_token_ids_cpu=torch.tensor([[1], [2]]),
        _sampled_token_ids=torch.tensor([[1], [2]]),
        _model_runner_output=out,
    )
    got = r.get_output()
    assert got is out
    assert out.sampled_token_ids == [[1], [2]]
    assert out.logprobs is None
    r.async_copy_ready_event.synchronize.assert_called_once()


def test_async_get_output_num_nans():
    out = SimpleNamespace(req_id_to_index={"a": 0, "b": 1})
    r = _async_output(
        sampled_token_ids_cpu=torch.tensor([[1], [2]]),
        _sampled_token_ids=torch.tensor([[1], [2]]),
        _model_runner_output=out,
        _num_nans_cpu=torch.tensor([0, 2], dtype=torch.int32),
    )
    with mock.patch.object(M.envs, "VLLM_RAISE_ON_LOGIT_NANS", False):
        got = r.get_output()
    assert got.num_nans_in_logits == {"a": 0, "b": 2}


def test_async_get_output_fault_raises():
    out = SimpleNamespace(req_id_to_index={})
    r = _async_output(
        sampled_token_ids_cpu=torch.tensor([[1]]),
        _sampled_token_ids=torch.tensor([[1]]),
        _model_runner_output=out,
        _has_fault=torch.tensor(True),
    )

    mask = mock.MagicMock()
    mask.cpu.return_value.tolist.return_value = [1]
    mgr = mock.MagicMock()
    mgr.query_active_mask.return_value = mask
    with mock.patch.object(M, "get_ep_all2all_manager", return_value=mgr):
        with pytest.raises(RuntimeError):
            r.get_output()


def test_async_get_output_maxgen_gt1():
    out = SimpleNamespace(req_id_to_index={})
    r = _async_output(
        sampled_token_ids_cpu=torch.tensor([[1, 2], [3, 4]]),
        _sampled_token_ids=torch.tensor([[1, 2], [3, 4]]),
        _model_runner_output=out,
        _logprobs_tensors_cpu=mock.MagicMock(),
    )
    with mock.patch.object(
        M.RejectionSampler, "parse_output", return_value=(["v"], ["p"])
    ):
        got = r.get_output()
    assert got.sampled_token_ids == ["v"]
    assert got.logprobs == ["p"]


def test_async_pooling_get_output():
    out = SimpleNamespace()
    ev = mock.MagicMock()
    r = object.__new__(M.AsyncGPUPoolingModelRunnerOutput)
    r.__dict__.update(
        _model_runner_output=out,
        _raw_pooler_output=torch.zeros(2, 3),
        async_copy_ready_event=ev,
    )
    got = r.get_output()
    assert got is out
    ev.synchronize.assert_called_once()


# ---------------------------------------------------------------------------
# Spec decode helpers
# ---------------------------------------------------------------------------
def test_input_fits_in_drafter_none():
    r = make_runner()
    assert r._input_fits_in_drafter(None) is False


def test_input_fits_in_drafter_branches():
    r = make_runner(
        speculative_config=SimpleNamespace(use_dflash=lambda: True),
        num_spec_tokens=2,
        effective_drafter_max_model_len=10,
    )
    assert r._input_fits_in_drafter(SimpleNamespace(max_seq_len=7)) is True
    assert r._input_fits_in_drafter(SimpleNamespace(max_seq_len=8)) is False

    r2 = make_runner(
        speculative_config=SimpleNamespace(use_dflash=lambda: False),
        num_spec_tokens=2,
        effective_drafter_max_model_len=10,
    )
    assert r2._input_fits_in_drafter(SimpleNamespace(max_seq_len=8)) is True


def test_sample_tokens_empty_state():
    r = make_runner(
        execute_model_state=None,
        kv_connector_output="KV",
        use_async_scheduling=False,
    )
    sentinel = object()
    with mock.patch.object(
        M.ModelRunnerOutput, "with_kv_conn_output_only", return_value=sentinel
    ) as cb:
        got = r.sample_tokens(None)
    assert got is sentinel
    assert r.kv_connector_output is None
    cb.assert_called_once_with("KV")


def test_sample_tokens_empty_state_async_receive():
    r = make_runner(
        execute_model_state=None,
        kv_connector_output=None,
        use_async_scheduling=True,
    )
    pp = SimpleNamespace(is_last_rank=False)
    sentinel = object()
    recv = mock.MagicMock()
    r._pp_receive_prev_sampled_token_ids_to_input_batch = recv
    with mock.patch.object(M, "get_pp_group", return_value=pp), mock.patch.object(
        M.ModelRunnerOutput, "with_kv_conn_output_only", return_value=sentinel
    ):
        got = r.sample_tokens(None)
    assert got is sentinel
    recv.assert_called_once()


def test_take_draft_token_ids_no_spec():
    r = make_runner(num_spec_tokens=0, _draft_token_req_ids=None)
    assert r.take_draft_token_ids() is None


def test_take_draft_token_ids_returns():
    ev = mock.MagicMock()
    r = make_runner(
        num_spec_tokens=2,
        _draft_token_req_ids=["a"],
        _draft_token_ids=torch.tensor([[5, 6]]),
        draft_token_ids_event=ev,
        draft_token_ids_cpu=torch.tensor([[5, 6]]),
    )
    with mock.patch.object(
        M, "DraftTokenIds", side_effect=lambda req, ids: ("DTI", req, ids)
    ):
        got = r.take_draft_token_ids()
    assert got == ("DTI", ["a"], [[5, 6]])


def test_get_draft_token_ids_cpu_list():
    r = make_runner(
        _draft_token_ids=[[1, 2]],
        input_batch=SimpleNamespace(req_ids=["a"]),
    )
    ids, reqs = r._get_draft_token_ids_cpu()
    assert ids == [[1, 2]]
    assert reqs == ["a"]


def test_get_draft_token_ids_cpu_none_req():
    r = make_runner(
        _draft_token_ids=torch.tensor([[1]]),
        _draft_token_req_ids=None,
    )
    assert r._get_draft_token_ids_cpu() == ([], [])


def test_get_draft_token_ids_cpu_tensor():
    ev = mock.MagicMock()
    r = make_runner(
        _draft_token_ids=torch.tensor([[1, 2], [3, 4]]),
        _draft_token_req_ids=["a", "b"],
        draft_token_ids_event=ev,
        draft_token_ids_cpu=torch.tensor([[1, 2], [3, 4]]),
    )
    ids, reqs = r._get_draft_token_ids_cpu()
    assert ids == [[1, 2], [3, 4]]
    assert reqs == ["a", "b"]
    ev.synchronize.assert_called_once()


def test_valid_count_none_event():
    r = make_runner(valid_sampled_token_count_event=None, input_batch=mock.MagicMock())
    assert r._get_valid_sampled_token_count() == []


def test_valid_count_returns():
    ev = mock.MagicMock()
    r = make_runner(
        valid_sampled_token_count_event=ev,
        input_batch=SimpleNamespace(prev_sampled_token_ids=torch.tensor([[1], [2]])),
        valid_sampled_token_count_cpu=torch.tensor([[1], [2]]),
    )
    assert r._get_valid_sampled_token_count() == [[1], [2]]
    ev.synchronize.assert_called_once()


def test_get_spec_decode_draft_probs_none():
    r = make_runner(_draft_probs=None, _draft_prob_req_ids=None)
    assert r._get_spec_decode_draft_probs(mock.MagicMock()) is None


def test_get_spec_decode_draft_probs_builds():
    r = make_runner(
        _draft_probs=torch.arange(24).reshape(2, 3, 4).float(),
        _draft_prob_req_ids=["a", "b"],
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    meta = SimpleNamespace(num_draft_tokens=[2, 0])
    got = r._get_spec_decode_draft_probs(meta)
    assert got.shape == (2, 4)


def test_get_spec_decode_draft_probs_missing_req():
    r = make_runner(
        _draft_probs=torch.rand(1, 4),
        _draft_prob_req_ids=["a"],
        input_batch=SimpleNamespace(req_ids=["x"]),
    )
    meta = SimpleNamespace(num_draft_tokens=[1])
    assert r._get_spec_decode_draft_probs(meta) is None


# ---------------------------------------------------------------------------
# Config / eagle helpers
# ---------------------------------------------------------------------------
def test_update_config_invalid_raises():
    r = make_runner()
    with pytest.raises(ValueError):
        r.update_config({"bogus": {}})


def test_update_config_valid():
    r = make_runner(load_config="OLD")
    with mock.patch.object(M, "update_config", return_value="NEW") as upd:
        r.update_config({"load_config": {"x": 1}})
    assert r.load_config == "NEW"
    upd.assert_called_once_with("OLD", {"x": 1})


def test_eagle_layers_none():
    r = make_runner(speculative_config=None)
    assert r._get_eagle3_aux_layers_from_config() is None


def test_eagle_layers_direct():
    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 2])
    r = make_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        )
    )
    assert r._get_eagle3_aux_layers_from_config() == (1, 2)


def test_eagle_layers_dflash():
    hf = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config={"target_layer_ids": [0, 1]},
        eagle_config=None,
    )
    r = make_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        )
    )
    assert r._get_eagle3_aux_layers_from_config() == (1, 2)


def test_eagle_layers_eagle_dict():
    hf = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config=None,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 4]},
    )
    r = make_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        )
    )
    assert r._get_eagle3_aux_layers_from_config() == (3, 4)


def test_eagle_layers_not_sequence():
    hf = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids="bad",
        dflash_config=None,
        eagle_config=None,
    )
    r = make_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        )
    )
    assert r._get_eagle3_aux_layers_from_config() is None


def test_setup_eagle_not_aux():
    model = mock.MagicMock()
    r = make_runner(use_aux_hidden_state_outputs=False, model=model)
    r._setup_eagle3_aux_hidden_state_outputs()
    model.set_aux_hidden_state_layers.assert_not_called()


def test_setup_eagle_unsupported_raises():
    r = make_runner(use_aux_hidden_state_outputs=True, model=mock.MagicMock())
    with mock.patch.object(M, "supports_eagle3", return_value=False):
        with pytest.raises(RuntimeError):
            r._setup_eagle3_aux_hidden_state_outputs()


def test_setup_eagle_sets_layers():
    model = mock.MagicMock()
    r = make_runner(use_aux_hidden_state_outputs=True, model=model)
    r._get_eagle3_aux_layers_from_config = mock.MagicMock(return_value=(1, 2))
    with mock.patch.object(M, "supports_eagle3", return_value=True):
        r._setup_eagle3_aux_hidden_state_outputs()
    model.set_aux_hidden_state_layers.assert_called_once_with((1, 2))


def test_setup_eagle_defaults():
    model = mock.MagicMock()
    model.get_eagle3_default_aux_hidden_state_layers.return_value = ("x",)
    r = make_runner(use_aux_hidden_state_outputs=True, model=model)
    r._get_eagle3_aux_layers_from_config = mock.MagicMock(return_value=None)
    with mock.patch.object(M, "supports_eagle3", return_value=True):
        r._setup_eagle3_aux_hidden_state_outputs()
    model.set_aux_hidden_state_layers.assert_called_once_with(("x",))


# ---------------------------------------------------------------------------
# NaN / randomize / prompt logprobs
# ---------------------------------------------------------------------------
def test_get_nans_none_logits():
    r = make_runner(input_batch=SimpleNamespace(req_id_to_index={"a": 0}))
    with mock.patch.object(M, "gpu_sync_allowed", lambda: nullcontext()), mock.patch.object(
        M.envs, "VLLM_RAISE_ON_LOGIT_NANS", False
    ):
        got = r._get_nans_in_logits(None)
    assert got == {"a": 0}


def test_get_nans_tensor_raises_hook():
    r = make_runner(input_batch=SimpleNamespace(req_id_to_index={"a": 0}))
    logits = torch.tensor([[0.0, float("nan")]])
    with mock.patch.object(M, "gpu_sync_allowed", lambda: nullcontext()), mock.patch.object(
        M.envs, "VLLM_RAISE_ON_LOGIT_NANS", True
    ), mock.patch.object(M, "raise_if_nan_logits") as ril:
        got = r._get_nans_in_logits(logits)
    assert got == {"a": 1}
    ril.assert_called_once_with({"a": 1})


def test_get_nans_index_error():
    r = make_runner(input_batch=SimpleNamespace(req_id_to_index={"a": 0}))
    with mock.patch.object(M, "gpu_sync_allowed", lambda: nullcontext()), mock.patch.object(
        M, "count_nans_per_row", side_effect=IndexError
    ):
        got = r._get_nans_in_logits(torch.zeros(1, 2))
    assert got == {}


def test_randomize_noop():
    r = make_runner(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        )
    )
    with mock.patch.object(M.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False):
        with r.maybe_randomize_inputs(None, None, False):
            pass


def test_randomize_ids():
    r = make_runner(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        ),
        input_ids=SimpleNamespace(gpu=torch.zeros(4, dtype=torch.long)),
        model_config=SimpleNamespace(get_vocab_size=lambda: 10),
    )
    ids = torch.ones(4, dtype=torch.long)
    with mock.patch.object(M.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False):
        with r.maybe_randomize_inputs(ids, None, True):
            pass
    assert int(ids.sum()) == 0


def test_randomize_embeds():
    r = make_runner(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        ),
        inputs_embeds=SimpleNamespace(gpu=torch.zeros(3, 4)),
    )
    embeds = torch.ones(3, 4)
    with mock.patch.object(M.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False):
        with r.maybe_randomize_inputs(None, embeds, True):
            pass
    assert float(embeds.abs().sum()) == 0.0


def test_prompt_logprobs_empty():
    r = make_runner(num_prompt_logprobs={})
    assert r._get_prompt_logprobs_dict(torch.zeros(2, 4), {}) == {}


# ---------------------------------------------------------------------------
# GC / Kunlunxin budget helpers
# ---------------------------------------------------------------------------
def test_freeze_gc():
    fake_gc = mock.MagicMock()
    fake_gc.isenabled.return_value = True
    with mock.patch.object(M, "gc", fake_gc), mock.patch.object(
        M.envs, "VLLM_ENABLE_CUDAGRAPH_GC", False
    ):
        with M.ModelRunnerFL._freeze_gc():
            pass
    fake_gc.freeze.assert_called_once()
    fake_gc.unfreeze.assert_called_once()
    fake_gc.enable.assert_called_once()


def test_kl_guard_on():
    with mock.patch.object(M, "current_platform", SimpleNamespace(vendor_name="kunlunxin")):
        assert make_runner()._kl_guard_on() is True
    with mock.patch.object(M, "current_platform", SimpleNamespace(vendor_name="other")):
        assert make_runner()._kl_guard_on() is False


def test_kl_reset_note_overbudget():
    r = make_runner()
    r._kl_budget_reset()
    assert (r._kl_used, r._kl_budget, r._kl_stop) == (0, 0, False)
    r._kl_note(SimpleNamespace(name="FULL"), 8, 5)
    assert r._kl_used == 5
    assert r._kl_captured[("FULL", 8)] == 5
    r._kl_budget = 4
    assert r._kl_over_budget() is True
    assert r._kl_stop is True


def test_kl_filter_zero_budget():
    platform = SimpleNamespace(
        torch_device_fn=SimpleNamespace(mem_get_info=lambda: (2 << 30, 8 << 30))
    )
    r = make_runner(_kl_budget=0, _kl_used=0, _kl_stop=False)
    descs = [1, 2]
    with mock.patch.object(M, "current_platform", platform):
        got = r._kl_filter_descs(descs)
    assert got is descs
    assert r._kl_budget > 0
    r._kl_stop = True
    with mock.patch.object(M, "current_platform", platform):
        assert r._kl_filter_descs(descs) == []


def test_kl_apply_truncates():
    def desc(n):
        return SimpleNamespace(num_tokens=n)

    ma, mb = SimpleNamespace(name="A"), SimpleNamespace(name="B")
    capture_descs = [(ma, [desc(4), desc(8)]), (mb, [desc(4), desc(8)])]
    r = make_runner(
        _kl_budget=100,
        _kl_stop=True,
        _kl_used=50,
        _kl_captured={("A", 4): 1, ("B", 4): 1},
        compilation_config=SimpleNamespace(),
    )
    r._kl_apply(capture_descs)
    assert r.compilation_config.cudagraph_capture_sizes == [4]
    assert r.compilation_config.max_cudagraph_capture_size == 4


def test_kl_apply_noop_when_not_over_budget():
    r = make_runner(_kl_budget=0, _kl_stop=False, compilation_config=SimpleNamespace())
    r._kl_apply([(SimpleNamespace(name="A"), [SimpleNamespace(num_tokens=4)])])
    assert not hasattr(r.compilation_config, "cudagraph_capture_sizes")


# ---------------------------------------------------------------------------
# Attention group / KV cache spec helpers
# ---------------------------------------------------------------------------
def test_attn_group_iterator():
    g1, g2, g3 = object(), object(), object()
    r = make_runner(attn_groups=[[g1, g2], [g3]])
    assert list(r._attn_group_iterator()) == [g1, g2, g3]


def test_kv_spec_attn_iterator_empty():
    r = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        attn_groups=[[object()]],
    )
    assert list(r._kv_cache_spec_attn_group_iterator()) == []


def test_kv_spec_attn_iterator_values():
    g1, g2 = object(), object()
    r = make_runner(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[1]),
        attn_groups=[[g1], [g2]],
    )
    assert list(r._kv_cache_spec_attn_group_iterator()) == [g1, g2]


def test_reorder_threshold_empty():
    r = make_runner(attn_groups=[])
    r.calculate_reorder_batch_threshold()
    assert r.reorder_batch_threshold is None


def test_reorder_threshold_min():
    def grp(v):
        return SimpleNamespace(
            get_metadata_builder=lambda: SimpleNamespace(reorder_batch_threshold=v)
        )

    r = make_runner(attn_groups=[[grp(None)], [grp(4)], [grp(2)]])
    r.calculate_reorder_batch_threshold()
    assert r.reorder_batch_threshold == 2


def test_cleanup_profiling_kv_cache():
    layer = SimpleNamespace(kv_cache=torch.zeros(2))
    r = make_runner(
        kv_caches=[torch.zeros(2)],
        attn_groups=[],
        cache_config=SimpleNamespace(num_gpu_blocks=5),
        compilation_config=SimpleNamespace(static_forward_context={"l": layer}),
    )
    with mock.patch.object(
        M.torch.accelerator, "synchronize", mock.MagicMock(), create=True
    ), mock.patch.object(
        M.torch.accelerator, "empty_cache", mock.MagicMock(), create=True
    ):
        r._cleanup_profiling_kv_cache()
    assert r.kv_caches == []
    assert isinstance(layer.kv_cache, torch.Tensor) and layer.kv_cache.numel() == 0
    assert r.cache_config.num_gpu_blocks is None


def test_shutdown(monkeypatch):
    rope_mod = types.ModuleType("vllm.model_executor.layers.rotary_embedding")
    rope_mod._ROPE_DICT = {"k": 1}
    ws_mod = types.ModuleType("vllm.v1.worker.workspace")
    ws_mod.reset_workspace_manager = mock.MagicMock()
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.rotary_embedding", rope_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.workspace", ws_mod)

    r = make_runner(
        graph_runtime=None,
        model=object(),
        compilation_config=SimpleNamespace(static_forward_context={"a": 1}),
    )
    r._cleanup_profiling_kv_cache = mock.MagicMock()
    with mock.patch.object(
        M, "current_platform", SimpleNamespace(is_rocm=lambda: False, is_xpu=lambda: False)
    ):
        r.shutdown()
    assert r.model is None
    assert r.compilation_config.static_forward_context == {}
    assert rope_mod._ROPE_DICT == {}
    ws_mod.reset_workspace_manager.assert_called_once()
    r._cleanup_profiling_kv_cache.assert_called_once()


def test_initialize_attn_backend_asserts_when_ready():
    r = make_runner(attn_groups=[[]])
    with pytest.raises(AssertionError):
        r.initialize_attn_backend(mock.MagicMock())


def test_maybe_add_kv_sharing_empty():
    r = make_runner(shared_kv_cache_layers={})
    with mock.patch.object(M, "add_kv_sharing_layers_to_kv_cache_groups") as add:
        r.maybe_add_kv_sharing_layers_to_kv_cache_groups(mock.MagicMock())
    add.assert_not_called()


def test_maybe_add_kv_sharing_calls():
    r = make_runner(
        shared_kv_cache_layers={"l2": "l1"},
        runner_only_attn_layers=set(),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=False),
    )
    cfg = SimpleNamespace(kv_cache_groups=[])
    with mock.patch.object(M, "add_kv_sharing_layers_to_kv_cache_groups") as add:
        r.maybe_add_kv_sharing_layers_to_kv_cache_groups(cfg)
    add.assert_called_once_with({"l2": "l1"}, cfg.kv_cache_groups, r.runner_only_attn_layers)


def test_maybe_add_kv_sharing_fast_prefill():
    r = make_runner(
        shared_kv_cache_layers={"l2": "l1"},
        runner_only_attn_layers=set(),
        kv_sharing_fast_prefill_eligible_layers=set(),
        cache_config=SimpleNamespace(kv_sharing_fast_prefill=True),
        vllm_config=SimpleNamespace(),
    )
    cfg = SimpleNamespace(kv_cache_groups=[])
    with mock.patch.object(
        M, "add_kv_sharing_layers_to_kv_cache_groups"
    ), mock.patch.object(M, "get_layers_from_vllm_config", return_value={"l1": 0, "l2": 0}):
        r.maybe_add_kv_sharing_layers_to_kv_cache_groups(cfg)
    assert r.kv_sharing_fast_prefill_eligible_layers == {"l2"}


def test_get_kv_cache_spec_ec_consumer_empty():
    with mock.patch.object(M, "has_ec_transfer", return_value=True), mock.patch.object(
        M, "get_ec_transfer", return_value=SimpleNamespace(is_consumer=False)
    ):
        r = make_runner()
        assert r.get_kv_cache_spec() == {}


def test_get_kv_cache_spec_basic():
    import dataclasses

    @dataclasses.dataclass
    class FakeSpec:
        indexes_kv_by_block_stride: bool = False

    class FakeBackend:
        def indexes_kv_by_block_stride(self):
            return True

        def customize_spec(self, spec):
            return spec

    class FakeAttn:
        def __init__(self, target):
            self.kv_sharing_target_layer_name = target

        def get_kv_cache_spec(self, cfg):
            return FakeSpec()

        def get_attn_backend(self):
            return FakeBackend()

    shared = FakeAttn("l0")
    normal = FakeAttn(None)
    r = make_runner(
        shared_kv_cache_layers={},
        use_ascend_hybrid_cache=False,
        vllm_config=SimpleNamespace(),
    )
    with mock.patch.object(M, "has_ec_transfer", return_value=False), mock.patch.object(
        M, "get_layers_from_vllm_config", return_value={"l0": shared, "l1": normal}
    ), mock.patch.object(M, "Attention", FakeAttn), mock.patch.object(
        M, "AttentionSpec", FakeSpec
    ), mock.patch.object(
        M, "set_current_vllm_config", lambda cfg: nullcontext()
    ):
        spec = r.get_kv_cache_spec()
    assert r.shared_kv_cache_layers == {"l0": "l0"}
    assert "l1" in spec and isinstance(spec["l1"], FakeSpec)


def test_may_add_encoder_only_layers():
    class FakeSpec:
        def __init__(self, **kw):
            pass

    class FakeGroup:
        def __init__(self, **kw):
            self.kw = kw

    attn_mod = SimpleNamespace(
        attn_type=M.AttentionType.ENCODER_ONLY, num_kv_heads=4, head_size=8
    )
    r = make_runner(
        vllm_config=SimpleNamespace(cache_config=SimpleNamespace(block_size=16)),
        kv_cache_dtype=torch.float16,
        runner_only_attn_layers=set(),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
    )
    with mock.patch.object(
        M, "get_layers_from_vllm_config", return_value={"enc": attn_mod}
    ), mock.patch.object(M, "EncoderOnlyAttentionSpec", FakeSpec), mock.patch.object(
        M, "KVCacheGroupSpec", FakeGroup
    ):
        r.may_add_encoder_only_layers_to_kv_cache_config()
    assert "enc" in r.runner_only_attn_layers
    assert len(r.kv_cache_config.kv_cache_groups) == 1


def test_initialize_metadata_builders():
    group = mock.MagicMock()
    r = make_runner(
        attn_groups=[[group]],
        vllm_config="CFG",
        device="DEV",
        parallel_config=SimpleNamespace(use_ubatching=False),
        speculative_config=None,
    )
    r.calculate_reorder_batch_threshold = mock.MagicMock()
    cfg = SimpleNamespace(kv_cache_groups=[object()])
    r.initialize_metadata_builders(cfg, [64])
    group.create_metadata_builders.assert_called_once()
    args = group.create_metadata_builders.call_args.args
    assert args[0] == "CFG" and args[1] == "DEV" and args[2] == 64
    r.calculate_reorder_batch_threshold.assert_called_once()


def test_initialize_metadata_builders_drafter():
    class FakeEagle:
        def __init__(self):
            self.initialize_attn_backend = mock.MagicMock()

    drafter = FakeEagle()
    group = mock.MagicMock()
    spec = SimpleNamespace(
        use_eagle=lambda: True, uses_draft_model=lambda: False
    )
    r = make_runner(
        attn_groups=[[group]],
        vllm_config="CFG",
        device="DEV",
        parallel_config=SimpleNamespace(use_ubatching=True, num_ubatches=2),
        speculative_config=spec,
        drafter=drafter,
    )
    r.calculate_reorder_batch_threshold = mock.MagicMock()
    cfg = SimpleNamespace(kv_cache_groups=[object()])
    with mock.patch.object(M, "EagleProposer", FakeEagle):
        r.initialize_metadata_builders(cfg, [64])
    drafter.initialize_attn_backend.assert_called_once_with(cfg, [64])


def test_to_list():
    r = make_runner(
        sampled_token_ids_pinned_cpu=torch.zeros(2, 2, dtype=torch.int32),
        transfer_event=mock.MagicMock(),
    )
    got = r._to_list(torch.tensor([[1, 2], [3, 4]]))
    assert got == [[1, 2], [3, 4]]
    r.transfer_event.record.assert_called_once()
    r.transfer_event.synchronize.assert_called_once()


def test_get_encoder_timing_stats():
    r = make_runner(
        _encoder_timing_lock=threading.Lock(),
        encoder_timing_registry={"a": SimpleNamespace(to_dict=lambda: {"x": 1})},
    )
    got = r.get_encoder_timing_stats()
    assert got == {"a": {"x": 1}}
    assert r.encoder_timing_registry == {}


def test_timed_encoder_operation_no_time():
    r = make_runner()
    with r.timed_encoder_operation(False, [], 0, 0):
        pass


def test_timed_encoder_operation_times():
    class FakeStats:
        def __init__(self):
            self.encoder_forward_secs = 0.0
            self.num_encoder_calls = 0

    perf = mock.MagicMock(side_effect=[1.0, 3.0])
    r = make_runner(
        _encoder_timing_lock=threading.Lock(),
        encoder_timing_registry={},
    )
    with mock.patch.object(
        M.torch.accelerator, "synchronize", mock.MagicMock(), create=True
    ), mock.patch.object(M, "EncoderTimingStats", FakeStats), mock.patch.object(
        M, "time", SimpleNamespace(perf_counter=perf)
    ):
        with r.timed_encoder_operation(True, [("a", 0), ("a", 1)], 0, 2):
            pass
    stats = r.encoder_timing_registry["a"]
    assert stats.num_encoder_calls == 1
    assert stats.encoder_forward_secs == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Dummy / profile helpers
# ---------------------------------------------------------------------------
def test_mm_dummy_budget_path():
    r = make_runner(
        mm_budget=SimpleNamespace(
            get_dummy_encoder_profile_inputs=lambda m, n: [],
            cache=object(),
        ),
        device=torch.device("cpu"),
    )
    with mock.patch.object(
        M, "group_and_batch_mm_kwargs", return_value=[("m", 0, {"k": 1})]
    ):
        got = r._get_mm_dummy_batch("image", 3)
    assert got == {"k": 1}


def test_mm_dummy_registry_path():
    item = object()
    r = make_runner(
        mm_budget=SimpleNamespace(cache=object()),
        mm_registry=mock.MagicMock(),
        model_config=SimpleNamespace(),
        device=torch.device("cpu"),
    )
    r.mm_registry.get_dummy_mm_inputs.return_value = {"mm_kwargs": {"image": [item]}}
    with mock.patch.object(
        M, "group_and_batch_mm_kwargs", return_value=[("image", 0, {"k": 2})]
    ):
        got = r._get_mm_dummy_batch("image", 2)
    assert got == {"k": 2}


def test_dummy_pooler_encoder_only():
    r = make_runner(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = r._dummy_pooler_run(torch.zeros(2, 4))
    assert isinstance(out, torch.Tensor) and out.numel() == 0


def test_dummy_pooler_no_tasks_raises():
    r = make_runner(
        model_config=SimpleNamespace(model="m"),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=None, model="m"
            )
        ),
    )
    r.get_supported_pooling_tasks = mock.MagicMock(return_value=[])
    with pytest.raises(RuntimeError):
        r._dummy_pooler_run(torch.zeros(2, 4))


def test_dummy_sampler_encoder_only():
    r = make_runner(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                multimodal_config=SimpleNamespace(mm_encoder_only=True)
            )
        )
    )
    out = r._dummy_sampler_run(torch.zeros(2, 4))
    assert isinstance(out, torch.Tensor) and out.numel() == 0


# ---------------------------------------------------------------------------
# Draft token copy / PP plumbing
# ---------------------------------------------------------------------------
def test_copy_draft_async_skip():
    r = make_runner(
        use_async_scheduling=True,
        _draft_token_ids=[],
        _draft_token_req_ids="UNSET",
        input_batch=SimpleNamespace(
            sampling_metadata=SimpleNamespace(output_token_ids=[])
        ),
    )
    sched = SimpleNamespace(has_structured_output_requests=False)
    r._copy_draft_token_ids_to_cpu(sched)
    assert r._draft_token_req_ids == "UNSET"


def test_copy_draft_non_tensor_sets_req_ids():
    r = make_runner(
        use_async_scheduling=False,
        _draft_token_ids=[],
        _draft_token_req_ids=None,
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
    )
    r._copy_draft_token_ids_to_cpu(SimpleNamespace(has_structured_output_requests=False))
    assert r._draft_token_req_ids == ["a", "b"]


def test_copy_valid_count_none_event():
    r = make_runner(valid_sampled_token_count_event=None)
    r._copy_valid_sampled_token_count(torch.zeros(1), torch.zeros(1))
    assert not hasattr(r, "input_batch")


def test_pp_broadcast():
    pp = SimpleNamespace(is_last_rank=True, rank=0, device_group=object())
    r = make_runner()
    r._is_all_reqs_chunked_prefill = mock.MagicMock(return_value=False)
    with mock.patch.object(M, "get_pp_group", return_value=pp), mock.patch.object(
        M.torch.distributed, "broadcast", create=True
    ) as bc:
        r._pp_broadcast_prev_sampled_token_ids(torch.zeros(2, 1, dtype=torch.int32))
    bc.assert_called_once()


def test_pp_broadcast_chunked_skips():
    pp = SimpleNamespace(is_last_rank=True, rank=0, device_group=object())
    r = make_runner()
    r._is_all_reqs_chunked_prefill = mock.MagicMock(return_value=True)
    with mock.patch.object(M, "get_pp_group", return_value=pp), mock.patch.object(
        M.torch.distributed, "broadcast", create=True
    ) as bc:
        r._pp_broadcast_prev_sampled_token_ids(torch.zeros(2, 1, dtype=torch.int32))
    bc.assert_not_called()


def test_pp_receive_prev_sampled_tokens():
    pp = SimpleNamespace(is_last_rank=False, last_rank=0, device_group=object())
    input_batch = SimpleNamespace(
        num_reqs=2,
        req_ids=["a", "b"],
        num_tokens_no_spec=[1, 1],
        is_token_ids=torch.zeros(2, 5, dtype=torch.bool),
    )
    r = make_runner(
        device=torch.device("cpu"),
        input_batch=input_batch,
        discard_request_mask=SimpleNamespace(np=np.array([False, False])),
        requests={},
    )
    r._is_all_reqs_chunked_prefill = mock.MagicMock(return_value=False)
    with mock.patch.object(M, "get_pp_group", return_value=pp), mock.patch.object(
        M.torch.distributed, "broadcast", return_value=mock.MagicMock(), create=True
    ):
        r._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert r.input_batch.prev_req_id_to_index == {"a": 0, "b": 1}
    assert r.input_batch.num_tokens_no_spec == [2, 2]
    assert r.input_batch.prev_sampled_token_ids is not None


def test_is_all_reqs_chunked_prefill():
    r = make_runner(
        input_batch=SimpleNamespace(num_reqs=2),
        discard_request_mask=SimpleNamespace(np=np.array([True, True])),
    )
    assert r._is_all_reqs_chunked_prefill() is True
    r.discard_request_mask = SimpleNamespace(np=np.array([True, False]))
    assert r._is_all_reqs_chunked_prefill() is False


# ---------------------------------------------------------------------------
# KV cache tensor init / reinit
# ---------------------------------------------------------------------------
def test_initialize_kv_cache_tensors_v032(monkeypatch):
    gpu_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_mod.bind_kv_cache = mock.MagicMock()
    gpu_mod.map_kv_caches_to_kernel_blocks = mock.MagicMock(return_value={"m": 1})
    utils_mod = types.ModuleType("vllm.v1.worker.utils")
    allocated = {"l1": torch.zeros(2)}
    utils_mod.allocate_kv_cache = mock.MagicMock(return_value=allocated)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", gpu_mod)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.utils", utils_mod)

    r = make_runner(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: "LAYOUT"
        ),
        shared_kv_cache_layers={"l0": "l1"},
        model_config=SimpleNamespace(hf_config=SimpleNamespace(model_type="base")),
        attn_groups=[[object()]],
        speculative_config=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_caches={},
    )
    cfg = SimpleNamespace(kv_cache_groups=[object()])
    out = r._initialize_kv_cache_tensors_v032(cfg, [64])
    assert "l0" in out and out["l0"] is allocated["l1"]
    gpu_mod.bind_kv_cache.assert_called_once()


def test_initialize_kv_cache_tensors_wrapper():
    r = make_runner()
    r._initialize_kv_cache_tensors_v032 = mock.MagicMock(return_value="KV")
    assert r.initialize_kv_cache_tensors("cfg", [64]) == "KV"
    r._initialize_kv_cache_tensors_v032.assert_called_once_with("cfg", [64], None)


def test_may_reinitialize_input_batch_noop():
    kind = object()
    spec = SimpleNamespace(block_size=16, max_num_blocks_per_req=lambda v, m: 4)
    group = SimpleNamespace(kv_cache_spec=spec)
    r = make_runner(
        max_model_len=128,
        max_encoder_len=0,
        vllm_config=SimpleNamespace(),
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[64],
        _init_max_num_blocks=[4],
        _init_slot_mapping_modes=[M.SlotMappingMode.TOKEN_TO_KV_SLOT],
        input_batch="IB",
    )
    cfg = SimpleNamespace(kv_cache_groups=[group])
    with mock.patch.object(M, "get_kv_cache_spec_kind", return_value=kind):
        r.may_reinitialize_input_batch(cfg, [64])
    assert r.input_batch == "IB"
    assert r._init_block_sizes == [16]


def test_may_reinitialize_input_batch_reinit():
    kind = object()
    spec = SimpleNamespace(block_size=32, max_num_blocks_per_req=lambda v, m: 8)
    group = SimpleNamespace(kv_cache_spec=spec)
    new_ib = object()
    r = make_runner(
        max_model_len=128,
        max_encoder_len=0,
        max_num_reqs=4,
        max_num_tokens=256,
        device=torch.device("cpu"),
        num_spec_tokens=0,
        is_pooling_model=False,
        input_batch=SimpleNamespace(logitsprocs="LP", logitsprocs_need_output_token_ids=True),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        vllm_config=SimpleNamespace(reasoning_config=None),
        cache_config=SimpleNamespace(use_replayssm=False),
        _init_block_sizes=[16],
        _init_kernel_block_sizes=[64],
        _init_max_num_blocks=[4],
        _init_slot_mapping_modes=[M.SlotMappingMode.TOKEN_TO_KV_SLOT],
    )
    r.__dict__["_jit_warmup_registry"] = SimpleNamespace(activate=lambda: nullcontext())
    cfg = SimpleNamespace(kv_cache_groups=[group])
    with mock.patch.object(M, "get_kv_cache_spec_kind", return_value=kind), mock.patch.object(
        M, "InputBatch", return_value=new_ib
    ) as ib:
        r.may_reinitialize_input_batch(cfg, [64])
    assert r.input_batch is new_ib
    assert r._init_block_sizes == [32]
    assert r._init_max_num_blocks == [8]
    ib.assert_called_once()


# ---------------------------------------------------------------------------
# Encoder cudagraph / musa helpers
# ---------------------------------------------------------------------------
def test_encoder_cudagraph_none():
    r = make_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        supports_mm_inputs=False,
    )
    assert r._create_encoder_cudagraph_manager() is None


def test_encoder_cudagraph_creates(monkeypatch):
    iface = types.ModuleType("vllm.model_executor.models.interfaces")
    iface.SupportsEncoderCudaGraph = object
    iface.supports_encoder_cudagraph = lambda m: True
    enc = types.ModuleType("vllm.v1.worker.encoder_cudagraph")
    sentinel = object()
    enc.EncoderCudaGraphManager = mock.MagicMock(return_value=sentinel)
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.interfaces", iface
    )
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.encoder_cudagraph", enc)

    r = make_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        vllm_config="CFG",
        device="DEV",
        dtype="DT",
        model=object(),
    )
    assert r._create_encoder_cudagraph_manager() is sentinel


def test_maybe_init_encoder_cudagraph():
    r = make_runner(encoder_cudagraph_manager=None)
    sentinel = object()
    r._create_encoder_cudagraph_manager = mock.MagicMock(return_value=sentinel)
    r._maybe_init_encoder_cudagraph_manager()
    assert r.encoder_cudagraph_manager is sentinel


def test_musa_graph_capture_context_cached():
    stream = mock.MagicMock()
    sentinel = object()
    r = make_runner(device=torch.device("cpu"))
    platform = SimpleNamespace(torch_device_fn=SimpleNamespace(Stream=stream))
    with mock.patch.object(M, "current_platform", platform), mock.patch.object(
        M, "GraphCaptureContext", return_value=sentinel
    ) as gcc:
        first = r._get_musa_graph_capture_context()
        second = r._get_musa_graph_capture_context()
    assert first is sentinel and second is sentinel
    gcc.assert_called_once()
    stream.assert_called_once()

# ---------------------------------------------------------------------------
# execute_model early paths
# ---------------------------------------------------------------------------
def test_execute_model_state_error():
    r = make_runner(execute_model_state=object())
    with pytest.raises(RuntimeError):
        r.execute_model(SimpleNamespace())


def test_execute_model_no_tokens_empty_output():
    r = make_runner(
        execute_model_state=None,
        speculative_config=None,
        parallel_config=SimpleNamespace(
            distributed_executor_backend="mp", data_parallel_size=1
        ),
    )
    r._update_states = mock.MagicMock(return_value=None)
    r.synchronize_input_prep = lambda: nullcontext()
    sentinel = object()
    sched = SimpleNamespace(total_num_scheduled_tokens=0, num_scheduled_tokens={})
    with mock.patch.object(
        M, "has_kv_transfer_group", return_value=False
    ), mock.patch.object(M, "has_ec_transfer", return_value=False), mock.patch.object(
        M, "record_function_or_nullcontext", lambda name: nullcontext()
    ), mock.patch.object(M, "EMPTY_MODEL_RUNNER_OUTPUT", sentinel):
        got = r.execute_model(sched)
    assert got is sentinel
    r._update_states.assert_called_once()
