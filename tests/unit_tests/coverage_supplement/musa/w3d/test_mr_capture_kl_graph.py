"""Coverage tests for ModelRunnerFL graph-capture / KL guard / profiling helpers.

Source under test: vllm_fl/worker/model_runner.py @ d12e9d9e (lines ~5749-8224).

No real graph capture, attention backend or KV cache allocation is performed;
``torch.accelerator``, ``graph_capture`` and the vLLM graph managers are
replaced with local doubles.
"""

import importlib
import sys
import threading
import types
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from vllm_fl.worker import model_runner as mr

MR = mr.ModelRunnerFL


def new_runner(**attrs):
    runner = object.__new__(MR)
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


def _patch_accel(monkeypatch, infos=None):
    infos = list(infos or [])
    state = {"i": 0}

    def get_memory_info():
        value = infos[min(state["i"], len(infos) - 1)]
        state["i"] += 1
        return (value, 0)

    accel = SimpleNamespace(
        synchronize=MagicMock(),
        empty_cache=MagicMock(),
        get_memory_info=get_memory_info,
    )
    monkeypatch.setattr(mr.torch, "accelerator", accel, raising=False)
    return accel


# ---------------------------------------------------------------------------
# MUSA graph capture context
# ---------------------------------------------------------------------------
def test_musa_graph_capture_context_creates_and_reuses(monkeypatch):
    created = []

    def fake_ctx(stream):
        created.append(stream)
        return SimpleNamespace(stream=stream)

    monkeypatch.setattr(mr, "GraphCaptureContext", fake_ctx)
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(Stream=lambda device: ("stream", device))
        ),
    )
    runner = new_runner(device="musa:0")

    first = MR._get_musa_graph_capture_context(runner)
    second = MR._get_musa_graph_capture_context(runner)

    assert first is second
    assert created == [("stream", "musa:0")]


# ---------------------------------------------------------------------------
# KL guard helpers
# ---------------------------------------------------------------------------
def test_kl_guard_on_checks_vendor(monkeypatch):
    runner = new_runner()
    monkeypatch.setattr(
        mr, "current_platform", SimpleNamespace(vendor_name="kunlunxin")
    )
    assert MR._kl_guard_on(runner) is True
    monkeypatch.setattr(mr, "current_platform", SimpleNamespace(vendor_name="musa"))
    assert MR._kl_guard_on(runner) is False


def test_kl_budget_reset_and_note_and_over_budget():
    runner = new_runner()
    MR._kl_budget_reset(runner)
    assert runner._kl_captured == {}
    assert runner._kl_used == 0

    MR._kl_note(runner, SimpleNamespace(name="FULL"), 8, 128)
    assert runner._kl_used == 128
    assert runner._kl_captured[("FULL", 8)] == 128

    runner._kl_budget = 256
    assert MR._kl_over_budget(runner) is False
    runner._kl_used = 300
    assert MR._kl_over_budget(runner) is True
    assert runner._kl_stop is True


def test_kl_filter_descs_computes_budget_then_stops(monkeypatch):
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(
            torch_device_fn=SimpleNamespace(
                mem_get_info=lambda: (8 * 1024 ** 3, 16 * 1024 ** 3)
            )
        ),
    )
    runner = new_runner()
    MR._kl_budget_reset(runner)
    descs = [1, 2]

    assert MR._kl_filter_descs(runner, descs) is descs
    assert runner._kl_budget == int(min(16 * 1024 ** 3 * 0.30, 8 * 1024 ** 3 * 0.75))

    runner._kl_used = runner._kl_budget
    assert MR._kl_filter_descs(runner, descs) == []
    assert runner._kl_stop is True


def test_kl_apply_early_returns():
    runner = new_runner(compilation_config=SimpleNamespace())
    MR._kl_budget_reset(runner)
    MR._kl_apply(runner, [("mode", [1])])  # budget 0 -> no-op

    runner._kl_budget = 10
    runner._kl_stop = False
    MR._kl_apply(runner, [("mode", [1])])  # not stopped -> no-op

    runner._kl_stop = True
    MR._kl_apply(runner, [(SimpleNamespace(name="FULL"), [])])  # no modes -> no-op


def test_kl_apply_reduces_capture_sizes():
    runner = new_runner(
        compilation_config=SimpleNamespace(
            cudagraph_capture_sizes=[8, 16], max_cudagraph_capture_size=16
        )
    )
    MR._kl_budget_reset(runner)
    runner._kl_budget = 100
    runner._kl_stop = True
    runner._kl_used = 100
    runner._kl_captured = {("FULL", 16): 5}
    mode = SimpleNamespace(name="FULL")
    descs = [
        SimpleNamespace(num_tokens=8),
        SimpleNamespace(num_tokens=16),
    ]

    MR._kl_apply(runner, [(mode, descs)])

    assert runner.compilation_config.cudagraph_capture_sizes == [8]
    assert runner.compilation_config.max_cudagraph_capture_size == 8


def test_kl_apply_keeps_sizes_when_all_captured():
    cc = SimpleNamespace(cudagraph_capture_sizes=[8, 16], max_cudagraph_capture_size=16)
    runner = new_runner(compilation_config=cc)
    MR._kl_budget_reset(runner)
    runner._kl_budget = 100
    runner._kl_stop = True
    runner._kl_captured = {("FULL", 8): 1, ("FULL", 16): 1}
    descs = [SimpleNamespace(num_tokens=8), SimpleNamespace(num_tokens=16)]

    MR._kl_apply(runner, [(SimpleNamespace(name="FULL"), descs)])

    assert cc.cudagraph_capture_sizes == [8, 16]


# ---------------------------------------------------------------------------
# warmup / capture descriptors
# ---------------------------------------------------------------------------
def test_warmup_and_capture_runs_warmups_then_capture(monkeypatch):
    _patch_accel(monkeypatch)
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=2)
    )
    runner._dummy_run = MagicMock()
    desc = SimpleNamespace(num_tokens=8, uniform=True, num_active_loras=0)

    MR._warmup_and_capture(runner, desc, mr.CUDAGraphMode.FULL)

    assert runner._dummy_run.call_count == 3
    first = runner._dummy_run.call_args_list[0].kwargs
    last = runner._dummy_run.call_args_list[-1].kwargs
    assert first["cudagraph_runtime_mode"] == mr.CUDAGraphMode.NONE
    assert first["force_attention"] is True
    assert last["cudagraph_runtime_mode"] == mr.CUDAGraphMode.FULL
    assert last["is_graph_capturing"] is True


def test_warmup_and_capture_zero_warmups_skips_sync(monkeypatch):
    accel = _patch_accel(monkeypatch)
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_num_of_warmups=0)
    )
    runner._dummy_run = MagicMock()
    desc = SimpleNamespace(num_tokens=4, uniform=False, num_active_loras=0)
    profiler = nullcontext()

    MR._warmup_and_capture(
        runner, desc, mr.CUDAGraphMode.PIECEWISE, profiler=profiler
    )

    runner._dummy_run.assert_called_once()
    accel.synchronize.assert_not_called()


class _ValidMode:
    name = "FULL"

    def is_valid_runtime_mode(self):
        return True


def test_capture_cudagraphs_rejects_none_mode():
    runner = new_runner()
    with pytest.raises(AssertionError):
        MR._capture_cudagraphs(runner, [], mr.CUDAGraphMode.NONE)


def test_capture_cudagraphs_empty_batch_returns(monkeypatch):
    runner = new_runner()
    runner.maybe_remove_all_loras = MagicMock()
    MR._capture_cudagraphs(runner, [], _ValidMode())
    runner.maybe_remove_all_loras.assert_not_called()


def test_capture_cudagraphs_captures_each_descriptor(monkeypatch):
    _patch_accel(monkeypatch)
    monkeypatch.setattr(mr, "is_global_first_rank", lambda: False)
    runner = new_runner(
        parallel_config=SimpleNamespace(use_ubatching=False),
        load_config=SimpleNamespace(use_tqdm_on_load=False),
        lora_config=None,
    )
    runner._warmup_and_capture = MagicMock()
    runner.maybe_remove_all_loras = MagicMock()
    descs = [
        SimpleNamespace(num_tokens=8, uniform=True),
        SimpleNamespace(num_tokens=16, uniform=True),
    ]

    MR._capture_cudagraphs(runner, descs, _ValidMode())

    assert runner._warmup_and_capture.call_count == 2
    runner.maybe_remove_all_loras.assert_called_once()


def test_capture_cudagraphs_uses_tqdm_on_first_rank(monkeypatch):
    _patch_accel(monkeypatch)
    monkeypatch.setattr(mr, "is_global_first_rank", lambda: True)
    runner = new_runner(
        parallel_config=SimpleNamespace(use_ubatching=False),
        load_config=SimpleNamespace(use_tqdm_on_load=False),
        lora_config=None,
    )
    runner._warmup_and_capture = MagicMock()
    runner.maybe_remove_all_loras = MagicMock()

    MR._capture_cudagraphs(
        runner, [SimpleNamespace(num_tokens=8, uniform=False)], _ValidMode()
    )

    runner._warmup_and_capture.assert_called_once()


# ---------------------------------------------------------------------------
# capture_model
# ---------------------------------------------------------------------------
def test_capture_model_none_mode_returns_zero():
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mode=mr.CUDAGraphMode.NONE)
    )
    assert MR.capture_model(runner) == 0


def test_capture_model_full_path(monkeypatch):
    accel = _patch_accel(monkeypatch, infos=[1000, 400])
    monkeypatch.setattr(mr, "set_cudagraph_capturing_enabled", MagicMock())
    monkeypatch.setattr(mr, "lock_workspace", MagicMock())
    monkeypatch.setattr(mr, "graph_capture", lambda **kw: nullcontext())
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(device_type="cuda", graph_pool_handle=lambda: "pool"),
    )
    ps = importlib.import_module("vllm.distributed.parallel_state")
    monkeypatch.setattr(ps, "get_world_group", lambda: SimpleNamespace(local_rank=1))
    monkeypatch.setattr(
        mr, "compilation_counter", SimpleNamespace(num_gpu_runner_capture_triggers=0)
    )

    descs = [SimpleNamespace(num_tokens=8, uniform=True)]
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mode="FULL"),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(capture_torch_profiler=False)
        ),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: [("FULL", descs)]),
        encoder_cudagraph_manager=None,
        device="dev",
    )
    runner._maybe_init_encoder_cudagraph_manager = MagicMock()
    runner._capture_cudagraphs = MagicMock()

    result = MR.capture_model(runner)

    assert result == 600
    runner._capture_cudagraphs.assert_called_once()
    mr.lock_workspace.assert_called_once()


# ---------------------------------------------------------------------------
# gc freeze + profiling kv cache cleanup + shutdown
# ---------------------------------------------------------------------------
def test_freeze_gc_freezes_and_restores(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_ENABLE_CUDAGRAPH_GC", False)
    with MR._freeze_gc():
        pass
    monkeypatch.setattr(mr.envs, "VLLM_ENABLE_CUDAGRAPH_GC", True)
    with MR._freeze_gc():
        pass


def test_cleanup_profiling_kv_cache_releases_state(monkeypatch):
    _patch_accel(monkeypatch)

    layer_tensor = SimpleNamespace(kv_cache=mr.torch.zeros(2))
    layer_list = SimpleNamespace(kv_cache=[1, 2])
    layer_impl = SimpleNamespace(
        kv_cache=[],
        impl=SimpleNamespace(_k_scale_cache="k", _v_scale_cache="v"),
    )
    ctx = {
        "a": layer_tensor,
        "b": layer_list,
        "c": layer_impl,
    }
    runner = new_runner(
        kv_caches=[1, 2],
        cross_layers_kv_cache=object(),
        cross_layers_attn_backend=object(),
        attn_groups=[["g"]],
        kv_cache_config=object(),
        cache_config=SimpleNamespace(num_gpu_blocks=9),
        compilation_config=SimpleNamespace(static_forward_context=ctx),
    )

    MR._cleanup_profiling_kv_cache(runner)

    assert runner.kv_caches == []
    assert runner.cross_layers_kv_cache is None
    assert runner.cross_layers_attn_backend is None
    assert runner.cache_config.num_gpu_blocks is None
    assert not hasattr(runner, "kv_cache_config")
    assert isinstance(layer_tensor.kv_cache, mr.torch.Tensor)
    assert layer_tensor.kv_cache.numel() == 0
    assert layer_list.kv_cache == []
    assert layer_impl.impl._k_scale_cache is None
    assert layer_impl.impl._v_scale_cache is None


def test_shutdown_releases_resources(monkeypatch):
    rope = importlib.import_module("vllm.model_executor.layers.rotary_embedding")
    workspace = importlib.import_module("vllm.v1.worker.workspace")
    reset_ws = MagicMock()
    monkeypatch.setattr(workspace, "reset_workspace_manager", reset_ws)
    monkeypatch.setattr(rope, "_ROPE_DICT", {"x": 1}, raising=False)
    accel = _patch_accel(monkeypatch)
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: False, is_xpu=lambda: False),
    )
    graph_runtime = SimpleNamespace(clear_decoder_graphs=MagicMock())
    runner = new_runner(
        graph_runtime=graph_runtime,
        compilation_config=SimpleNamespace(
            static_forward_context=SimpleNamespace(clear=MagicMock())
        ),
        model=object(),
    )
    runner._cleanup_profiling_kv_cache = MagicMock()

    MR.shutdown(runner)

    graph_runtime.clear_decoder_graphs.assert_called_once()
    runner._cleanup_profiling_kv_cache.assert_called_once()
    assert runner.model is None
    assert rope._ROPE_DICT == {}
    reset_ws.assert_called_once()


def test_shutdown_rocm_clears_graphs(monkeypatch):
    rope = importlib.import_module("vllm.model_executor.layers.rotary_embedding")
    workspace = importlib.import_module("vllm.v1.worker.workspace")
    monkeypatch.setattr(workspace, "reset_workspace_manager", MagicMock())
    monkeypatch.setattr(rope, "_ROPE_DICT", {}, raising=False)
    _patch_accel(monkeypatch)
    monkeypatch.setattr(
        mr,
        "current_platform",
        SimpleNamespace(is_rocm=lambda: True, is_xpu=lambda: False),
    )
    monkeypatch.setattr(
        mr.CUDAGraphWrapper, "clear_all_graphs", MagicMock(), raising=False
    )
    monkeypatch.setattr(mr.GraphWrapper, "clear_all_graphs", MagicMock(), raising=False)
    monkeypatch.setattr(
        mr.BreakableCUDAGraphWrapper, "clear_all_graphs", MagicMock(), raising=False
    )
    runner = new_runner(
        graph_runtime=None,
        compilation_config=SimpleNamespace(
            static_forward_context=SimpleNamespace(clear=MagicMock())
        ),
        model=object(),
        encoder_cudagraph_manager=object(),
    )
    runner._cleanup_profiling_kv_cache = MagicMock()

    MR.shutdown(runner)

    mr.CUDAGraphWrapper.clear_all_graphs.assert_called_once()
    mr.GraphWrapper.clear_all_graphs.assert_called_once()
    mr.BreakableCUDAGraphWrapper.clear_all_graphs.assert_called_once()
    assert runner.encoder_cudagraph_manager is None


# ---------------------------------------------------------------------------
# encoder cudagraph manager
# ---------------------------------------------------------------------------
def test_create_encoder_manager_early_exits():
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=False),
        supports_mm_inputs=True,
    )
    assert MR._create_encoder_cudagraph_manager(runner) is None

    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=False,
    )
    assert MR._create_encoder_cudagraph_manager(runner) is None


def _install_encoder_modules(monkeypatch, supported):
    interfaces = types.ModuleType("vllm.model_executor.models.interfaces")
    interfaces.SupportsEncoderCudaGraph = object
    interfaces.supports_encoder_cudagraph = lambda model: supported
    monkeypatch.setitem(
        sys.modules, "vllm.model_executor.models.interfaces", interfaces
    )
    created = {}

    def manager(**kwargs):
        created.update(kwargs)
        return SimpleNamespace(**kwargs)

    encoder_mod = types.ModuleType("vllm.v1.worker.encoder_cudagraph")
    encoder_mod.EncoderCudaGraphManager = manager
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.encoder_cudagraph", encoder_mod)
    return created


def test_create_encoder_manager_unsupported_model(monkeypatch):
    _install_encoder_modules(monkeypatch, supported=False)
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
    )
    runner.get_model = lambda: "raw"
    assert MR._create_encoder_cudagraph_manager(runner) is None


def test_create_encoder_manager_success(monkeypatch):
    created = _install_encoder_modules(monkeypatch, supported=True)
    runner = new_runner(
        compilation_config=SimpleNamespace(cudagraph_mm_encoder=True),
        supports_mm_inputs=True,
        vllm_config="cfg",
        device="dev",
        dtype="dtype",
    )
    runner.get_model = lambda: "raw"

    out = MR._create_encoder_cudagraph_manager(runner)

    assert out is not None
    assert created == {
        "vllm_config": "cfg",
        "device": "dev",
        "dtype": "dtype",
        "model": "raw",
    }


def test_maybe_init_encoder_manager():
    runner = new_runner(encoder_cudagraph_manager=None)
    runner._create_encoder_cudagraph_manager = MagicMock(return_value="mgr")
    MR._maybe_init_encoder_cudagraph_manager(runner)
    assert runner.encoder_cudagraph_manager == "mgr"

    runner2 = new_runner(encoder_cudagraph_manager="existing")
    runner2._create_encoder_cudagraph_manager = MagicMock()
    MR._maybe_init_encoder_cudagraph_manager(runner2)
    runner2._create_encoder_cudagraph_manager.assert_not_called()


# ---------------------------------------------------------------------------
# eagle3 aux hidden states
# ---------------------------------------------------------------------------
def test_eagle3_aux_layers_from_config_variants():
    runner = new_runner(speculative_config=None)
    assert MR._get_eagle3_aux_layers_from_config(runner) is None

    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 2])
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) == (1, 2)

    hf2 = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        dflash_config={"target_layer_ids": [0, 1]},
    )
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf2)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) == (1, 2)

    hf3 = SimpleNamespace(
        eagle_aux_hidden_state_layer_ids=None,
        eagle_config={"eagle_aux_hidden_state_layer_ids": [3, 4]},
    )
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf3)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) == (3, 4)

    hf4 = SimpleNamespace(eagle_aux_hidden_state_layer_ids=5)
    runner = new_runner(
        speculative_config=SimpleNamespace(
            draft_model_config=SimpleNamespace(hf_config=hf4)
        )
    )
    assert MR._get_eagle3_aux_layers_from_config(runner) is None


def test_setup_eagle3_aux_early_return(monkeypatch):
    called = []
    monkeypatch.setattr(mr, "supports_eagle3", lambda m: called.append(m) or True)
    runner = new_runner(use_aux_hidden_state_outputs=False)
    runner.get_model = lambda: "model"
    MR._setup_eagle3_aux_hidden_state_outputs(runner)
    assert called == []


def test_setup_eagle3_aux_requires_support(monkeypatch):
    monkeypatch.setattr(mr, "supports_eagle3", lambda m: False)
    runner = new_runner(use_aux_hidden_state_outputs=True)
    runner.get_model = lambda: "model"
    with pytest.raises(RuntimeError):
        MR._setup_eagle3_aux_hidden_state_outputs(runner)


def test_setup_eagle3_aux_uses_config_and_default(monkeypatch):
    monkeypatch.setattr(mr, "supports_eagle3", lambda m: True)
    model = SimpleNamespace(
        set_aux_hidden_state_layers=MagicMock(),
        get_eagle3_default_aux_hidden_state_layers=MagicMock(return_value=(7, 8)),
    )
    runner = new_runner(use_aux_hidden_state_outputs=True, model=model)
    runner.get_model = lambda: model
    runner._get_eagle3_aux_layers_from_config = MagicMock(return_value=(1, 2))

    MR._setup_eagle3_aux_hidden_state_outputs(runner)
    model.set_aux_hidden_state_layers.assert_called_once_with((1, 2))

    runner._get_eagle3_aux_layers_from_config = MagicMock(return_value=None)
    MR._setup_eagle3_aux_hidden_state_outputs(runner)
    model.set_aux_hidden_state_layers.assert_called_with((7, 8))


# ---------------------------------------------------------------------------
# maybe_randomize_inputs
# ---------------------------------------------------------------------------
def _randomize_runner(**attrs):
    base = dict(
        input_ids=SimpleNamespace(gpu=mr.torch.zeros(8, dtype=mr.torch.long)),
        inputs_embeds=SimpleNamespace(gpu=mr.torch.zeros(8, 4)),
        model_config=SimpleNamespace(get_vocab_size=lambda: 100),
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=1)
        ),
    )
    base.update(attrs)
    return new_runner(**base)


def test_maybe_randomize_inputs_disabled(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)
    runner = _randomize_runner()
    ids = mr.torch.ones(4, dtype=mr.torch.long)
    with MR.maybe_randomize_inputs(runner, ids, None):
        assert (ids == 1).all()
    assert (ids == 1).all()


def test_maybe_randomize_inputs_ids_branch(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", True)
    runner = _randomize_runner(
        vllm_config=SimpleNamespace(
            parallel_config=SimpleNamespace(data_parallel_size=2)
        )
    )
    ids = mr.torch.ones(4, dtype=mr.torch.long)
    with MR.maybe_randomize_inputs(runner, ids, None):
        pass
    assert (ids == 0).all()


def test_maybe_randomize_inputs_embeds_branch(monkeypatch):
    monkeypatch.setattr(mr.envs, "VLLM_RANDOMIZE_DP_DUMMY_INPUTS", False)
    runner = _randomize_runner()
    embeds = mr.torch.ones(4, 4)
    with MR.maybe_randomize_inputs(runner, None, embeds, randomize_inputs=True):
        pass
    assert (embeds == 0).all()


# ---------------------------------------------------------------------------
# nans / prompt logprobs
# ---------------------------------------------------------------------------
def test_get_nans_in_logits_none(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(mr, "nans_to_dict", lambda counts, mapping: {"n": len(counts)})
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", False)
    runner = new_runner(input_batch=SimpleNamespace(req_id_to_index={}))
    assert MR._get_nans_in_logits(runner, None) == {"n": 0}


def test_get_nans_in_logits_counts_and_raises(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())
    monkeypatch.setattr(
        mr,
        "count_nans_per_row",
        lambda logits: SimpleNamespace(tolist=lambda: [0, 3]),
    )
    monkeypatch.setattr(mr, "nans_to_dict", lambda counts, mapping: {"r": counts[1]})
    raiser = MagicMock()
    monkeypatch.setattr(mr, "raise_if_nan_logits", raiser)
    monkeypatch.setattr(mr.envs, "VLLM_RAISE_ON_LOGIT_NANS", True)
    runner = new_runner(input_batch=SimpleNamespace(req_id_to_index={"r": 0}))

    assert MR._get_nans_in_logits(runner, object()) == {"r": 3}
    raiser.assert_called_once_with({"r": 3})


def test_get_nans_in_logits_index_error(monkeypatch):
    monkeypatch.setattr(mr, "gpu_sync_allowed", lambda: nullcontext())

    def boom(logits):
        raise IndexError("bad")

    monkeypatch.setattr(mr, "count_nans_per_row", boom)
    runner = new_runner(input_batch=SimpleNamespace(req_id_to_index={}))
    assert MR._get_nans_in_logits(runner, object()) == {}


class _FakeIndexable:
    def __getitem__(self, item):
        return self


class _FakeTensorSlot:
    def __getitem__(self, item):
        return self

    def copy_(self, src, non_blocking=False):
        return self


def _empty_logprobs_tensors(num_tokens, num_logprobs):
    return SimpleNamespace(
        logprob_token_ids=_FakeTensorSlot(),
        logprobs=_FakeTensorSlot(),
        selected_token_ranks=_FakeTensorSlot(),
    )


def test_get_prompt_logprobs_dict_early_return():
    runner = new_runner(num_prompt_logprobs={})
    assert MR._get_prompt_logprobs_dict(runner, object(), {}) == {}


def test_get_prompt_logprobs_dict_skips_missing_and_embedding(monkeypatch):
    runner = new_runner(
        num_prompt_logprobs={"r1": 1, "r2": 1},
        requests={
            "r2": SimpleNamespace(prompt_token_ids=None),
        },
    )
    assert MR._get_prompt_logprobs_dict(runner, object(), {"r2": 1}) == {}


def test_get_prompt_logprobs_dict_completes_request(monkeypatch):
    monkeypatch.setattr(mr, "async_tensor_h2d", lambda ids, device: _FakeIndexable())
    monkeypatch.setattr(
        mr,
        "LogprobsTensors",
        SimpleNamespace(empty_cpu=_empty_logprobs_tensors),
    )
    logits = SimpleNamespace(to=lambda dtype: logits)
    runner = new_runner(
        num_prompt_logprobs={"r": 1},
        requests={
            "r": SimpleNamespace(
                prompt_token_ids=[0, 1, 2, 3, 4, 5],
                in_progress_prompt_logprobs_cpu=None,
                num_computed_tokens=0,
            )
        },
        device="dev",
        input_batch=SimpleNamespace(req_id_to_index={"r": 0}),
        query_start_loc=SimpleNamespace(np=[SimpleNamespace(item=lambda: 0)]),
        model=SimpleNamespace(compute_logits=lambda hs: logits),
        model_config=SimpleNamespace(logprobs_mode="raw_logits"),
        sampler=SimpleNamespace(
            gather_logprobs=lambda scores, n, tgt: ("tok", "lp", "ranks", None)
        ),
    )
    runner._sync_device = MagicMock()

    out = MR._get_prompt_logprobs_dict(runner, mr.torch.zeros(8, 4), {"r": 7})

    assert set(out) == {"r"}
    assert runner.num_prompt_logprobs == {}
    runner._sync_device.assert_called_once()
    assert runner.requests["r"].in_progress_prompt_logprobs_cpu is None


# ---------------------------------------------------------------------------
# reload weights
# ---------------------------------------------------------------------------
def _reload_runner(model, **attrs):
    base = dict(
        load_config=SimpleNamespace(load_format="auto"),
        lora_config=None,
        model_config=SimpleNamespace(quantization=None, model="orig", revision="rev"),
    )
    base.update(attrs)
    runner = new_runner(**base)
    runner.get_model = lambda: model
    runner.reset_lora_state = MagicMock()
    runner.reset_encoder_cache = MagicMock()
    runner.reset_mm_cache = MagicMock()
    return runner


def test_reload_weights_checkpoint_format(monkeypatch):
    model = SimpleNamespace(
        named_parameters=lambda: [("layer.base_layer.w", object())],
        load_weights=MagicMock(return_value={"layer.w"}),
    )
    monkeypatch.setattr(mr, "initialize_layerwise_reload", MagicMock())
    monkeypatch.setattr(mr, "finalize_layerwise_reload", MagicMock())
    runner = _reload_runner(model, lora_config=object())

    MR.reload_weights(runner, weights_iterator=[("w", 1)], is_checkpoint_format=True)

    mr.initialize_layerwise_reload.assert_called_once_with(model)
    model.load_weights.assert_called_once()
    mr.finalize_layerwise_reload.assert_called_once_with(model, runner.model_config)
    runner.reset_encoder_cache.assert_called_once()
    runner.reset_mm_cache.assert_called_once()


def test_reload_weights_kernel_format(monkeypatch):
    param = SimpleNamespace(copy_=MagicMock())
    model = SimpleNamespace(
        named_parameters=lambda: [("w", object())],
    )
    monkeypatch.setattr(mr, "_get_parameter_for_reload", lambda m, name: param)
    runner = _reload_runner(model)

    MR.reload_weights(
        runner,
        weights_iterator=[("w", "value")],
        is_checkpoint_format=False,
    )

    param.copy_.assert_called_once_with("value")
    runner.reset_lora_state.assert_called_once()


def test_reload_weights_from_disk_sets_path(monkeypatch):
    model = SimpleNamespace(
        named_parameters=lambda: [("w", object())],
        load_weights=MagicMock(return_value={"w"}),
    )
    loader = SimpleNamespace(get_all_weights=MagicMock(return_value=[("w", 1)]))
    monkeypatch.setattr(mr, "get_model_loader", lambda cfg: loader)
    monkeypatch.setattr(mr, "initialize_layerwise_reload", MagicMock())
    monkeypatch.setattr(mr, "finalize_layerwise_reload", MagicMock())
    runner = _reload_runner(model)

    MR.reload_weights(runner, weights_path="new/path", is_checkpoint_format=True)

    assert runner.model_config.model == "new/path"
    assert runner.model_config.revision is None
    loader.get_all_weights.assert_called_once_with(runner.model_config, model)


def test_reload_weights_loader_without_get_all_weights(monkeypatch):
    model = SimpleNamespace(named_parameters=lambda: [("w", object())])
    monkeypatch.setattr(mr, "get_model_loader", lambda cfg: SimpleNamespace())
    runner = _reload_runner(model)
    with pytest.raises(NotImplementedError):
        MR.reload_weights(runner, weights_iterator=None, is_checkpoint_format=True)


# ---------------------------------------------------------------------------
# minimal kv cache profiling + profile_run
# ---------------------------------------------------------------------------
def test_init_minimal_kv_cache_for_profiling(monkeypatch):
    kcu = importlib.import_module("vllm.v1.core.kv_cache_utils")
    observed = {}

    def fake_config(cfg, groups, available_memory):
        observed["override"] = runner.cache_config.num_gpu_blocks_override
        observed["available"] = available_memory
        return SimpleNamespace(num_blocks=3)

    monkeypatch.setattr(kcu, "get_kv_cache_groups", lambda cfg, spec: ["grp"])
    monkeypatch.setattr(kcu, "get_kv_cache_config_from_groups", fake_config)
    monkeypatch.setattr(
        mr,
        "KVCacheSpecRegistry",
        SimpleNamespace(check_kv_cache_spec_registry=lambda spec: None),
    )
    runner = new_runner(
        vllm_config=SimpleNamespace(),
        max_num_reqs=4,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
        cache_config=SimpleNamespace(num_gpu_blocks_override=99, num_gpu_blocks=None),
    )
    runner.get_kv_cache_spec = MagicMock(return_value={"l": 1})
    runner.initialize_kv_cache = MagicMock()

    MR._init_minimal_kv_cache_for_profiling(runner)

    assert observed["override"] == 4
    assert observed["available"] == 0
    assert runner.cache_config.num_gpu_blocks_override == 99
    assert runner.cache_config.num_gpu_blocks == 3
    runner.initialize_kv_cache.assert_called_once()


def _profile_runner(**attrs):
    base = dict(
        max_num_tokens=16,
        encoder_cache={},
        supports_mm_inputs=False,
        is_pooling_model=False,
    )
    base.update(attrs)
    runner = new_runner(**base)
    runner._dummy_run = MagicMock(return_value=("hs", "lhs"))
    runner._sync_device = MagicMock()
    return runner


def test_profile_run_sampler_path(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    runner = _profile_runner()
    runner._dummy_sampler_run = MagicMock(return_value="out")

    MR.profile_run(runner)

    runner._dummy_run.assert_called_once_with(runner.max_num_tokens, is_profile=True)
    runner._dummy_sampler_run.assert_called_once_with("lhs")
    runner._sync_device.assert_called_once()
    assert runner.encoder_cache == {}


def test_profile_run_pooling_and_non_last_rank(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    pooled = _profile_runner(is_pooling_model=True)
    pooled._dummy_pooler_run = MagicMock(return_value="pool")
    MR.profile_run(pooled)
    pooled._dummy_pooler_run.assert_called_once_with("hs")

    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=False))
    tail = _profile_runner()
    tail._dummy_sampler_run = MagicMock()
    MR.profile_run(tail)
    tail._dummy_sampler_run.assert_not_called()


def _mm_profile_runner(budget, **attrs):
    model = SimpleNamespace(embed_multimodal=MagicMock(return_value=["e1", "e2"]))
    config = dict(
        supports_mm_inputs=True,
        mm_budget=budget,
        mm_registry=SimpleNamespace(),
        model=model,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=False)
        ),
        device="dev",
    )
    config.update(attrs)
    runner = _profile_runner(**config)
    runner._get_mm_dummy_batch = MagicMock(return_value={"m": 1})
    return runner


def test_profile_run_mm_encoder(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    monkeypatch.setattr(mr, "sanity_check_mm_encoder_outputs", MagicMock())
    budget = SimpleNamespace(
        get_encoder_budget=lambda: 10,
        mm_max_toks_per_item=5,
        get_modality_with_max_tokens=lambda: "image",
        mm_max_items_per_batch={"image": 2},
    )
    runner = _mm_profile_runner(budget)
    runner._dummy_sampler_run = MagicMock()

    MR.profile_run(runner)

    runner._get_mm_dummy_batch.assert_called_once_with("image", 2)
    runner.model.embed_multimodal.assert_called_once_with(m=1)
    mr.sanity_check_mm_encoder_outputs.assert_called_once()


def test_profile_run_mm_skips_embedding_only(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    budget = SimpleNamespace(
        get_encoder_budget=lambda: 10,
        mm_max_toks_per_item=0,
    )
    runner = _mm_profile_runner(budget)
    runner._dummy_sampler_run = MagicMock()

    MR.profile_run(runner)

    runner.model.embed_multimodal.assert_not_called()


def test_profile_run_mm_skips_when_disabled(monkeypatch):
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_last_rank=True))
    budget = SimpleNamespace(get_encoder_budget=lambda: 10, mm_max_toks_per_item=5)
    runner = _mm_profile_runner(
        budget,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=True)
        ),
    )
    runner._dummy_sampler_run = MagicMock()

    MR.profile_run(runner)

    runner.model.embed_multimodal.assert_not_called()


# ---------------------------------------------------------------------------
# profile_cudagraph_memory (zero-graph early exit)
# ---------------------------------------------------------------------------
def test_profile_cudagraph_memory_no_graphs(monkeypatch):
    monkeypatch.setattr(mr, "set_current_vllm_config", lambda cfg: nullcontext())
    monkeypatch.setattr(
        mr, "compilation_counter", SimpleNamespace(num_cudagraph_captured=7)
    )
    runner = new_runner(
        vllm_config=SimpleNamespace(),
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: []),
    )
    runner._init_minimal_kv_cache_for_profiling = MagicMock()
    runner._create_encoder_cudagraph_manager = MagicMock(return_value=None)
    runner._cleanup_profiling_kv_cache = MagicMock()

    assert MR.profile_cudagraph_memory(runner) == 0
    runner._init_minimal_kv_cache_for_profiling.assert_called_once()
    runner._cleanup_profiling_kv_cache.assert_called_once()
