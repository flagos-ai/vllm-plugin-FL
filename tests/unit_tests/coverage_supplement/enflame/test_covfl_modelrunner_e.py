"""Coverage tests ``e``: remaining small/branchy helpers of model_runner.py.

Targets the scattered methods not exercised by test_covfl_modelrunner_{a,b,c,d}.py:

* ``initialize_attn_backend`` (nested ``get_attn_backends_for_group`` /
  ``create_attn_groups``, fast-prefill backend, uniform specs, num_heads fallback)
* ``_check_and_update_cudagraph_mode`` (min support + drafter init)
* ``reload_weights`` (iterator / kernel format / disk load / errors)
* ``_get_prompt_logprobs_dict`` (skip, chunk, final-chunk, both logprob modes)
* ``init_routed_experts_capturer``
* ``propose_draft_token_ids`` (ngram / custom / gpu-ngram / suffix / medusa /
  extract-hidden-states / eagle padded & unpadded)
* ``profile_cudagraph_memory`` (early-return + a fully mocked one-graph run)

Self-contained: pytest + unittest.mock + stdlib + torch/numpy.  No real
accelerator work, no weights, no process spawning.  The whole module skips
cleanly when vLLM is unavailable.
"""

from __future__ import annotations

import threading
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


def _fake_ctx(*args, **kwargs):
    return nullcontext()


def _spec(**overrides):
    """A permissive speculative_config stand-in with all predicate methods."""
    values = dict(
        method="unknown",
        disable_padded_drafter_batch=False,
        use_ngram_gpu=lambda: False,
        use_eagle=lambda: False,
        use_dflash=lambda: False,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        use_ngram=lambda: False,
        uses_ngram=lambda: False,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


# ---------------------------------------------------------------------------
# initialize_attn_backend  (nested helpers)
# ---------------------------------------------------------------------------


def _fake_layer(backend, num_heads=None):
    layer = SimpleNamespace(get_attn_backend=lambda: backend)
    if num_heads is not None:
        layer.num_heads = num_heads
    else:
        layer.__dict__.pop("num_heads", None)
    return layer


def test_initialize_attn_backend_main_path():
    class Backend:
        def __init__(self, name="BackendA"):
            self.name = name

        def full_cls_name(self):
            return self.name

        def get_builder_cls(self):
            raise AssertionError("should not be called when mode mocked")

    b1 = Backend("B1")
    b2 = Backend("B2")
    layers = {"l1": _fake_layer(b1, num_heads=4), "l2": _fake_layer(b2, num_heads=8)}
    created = []

    class FakeAttentionGroup:
        def __init__(self, attn_backend, layer_names, spec, gid):
            created.append((attn_backend, tuple(layer_names), spec, gid))

    r = make_runner(
        attn_groups=[],
        vllm_config="CFG",
        kv_sharing_fast_prefill_eligible_layers=set(),
        uniform_decode_query_len=1,
        max_num_reqs=8,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        speculative_config=None,
        cudagraph_dispatcher=mock.MagicMock(),
    )
    r._check_and_update_cudagraph_mode = mock.MagicMock()

    spec = object()
    group_spec = SimpleNamespace(layer_names=["l1", "l2"], kv_cache_spec=spec)
    cfg = SimpleNamespace(kv_cache_groups=[group_spec])

    with mock.patch.object(M, "get_layers_from_vllm_config", return_value=layers), \
         mock.patch.object(M, "AttentionGroup", FakeAttentionGroup), \
         mock.patch.object(M, "check_attention_cp_compatibility") as cp_check:
        r.initialize_attn_backend(cfg)

    cp_check.assert_called_once_with("CFG")
    r._check_and_update_cudagraph_mode.assert_called_once()
    # Two distinct (backend, num_heads) keys -> two groups, each 1 layer.
    assert len(created) == 2
    assert len(r.attn_groups) == 1
    assert len(r.attn_groups[0]) == 2
    assert {g[3] for g in created} == {0}


def test_initialize_attn_backend_fast_prefill_and_uniform_spec():
    class Backend:
        def __init__(self, name):
            self.name = name

        def full_cls_name(self):
            return self.name

    base = Backend("Base")
    fast = Backend("FastPrefill")
    fast_calls = []
    layers = {"l1": _fake_layer(base), "l2": _fake_layer(base, num_heads=2)}

    class UniformTypeKVCacheSpecs:
        def __init__(self, mapping):
            self.kv_cache_specs = mapping

    sub_spec = SimpleNamespace(name="sub")
    uniform = UniformTypeKVCacheSpecs({"l1": sub_spec, "l2": sub_spec})

    class FakeAttentionGroup:
        def __init__(self, attn_backend, layer_names, spec, gid):
            self.backend = attn_backend
            self.layers = tuple(layer_names)
            self.spec = spec

    def fake_create_fast_prefill(name, backend):
        fast_calls.append((name, backend))
        return fast

    r = make_runner(
        attn_groups=[],
        vllm_config="CFG",
        kv_sharing_fast_prefill_eligible_layers={"l1"},
        uniform_decode_query_len=1,
        max_num_reqs=4,
        kv_cache_config=SimpleNamespace(kv_cache_groups=[]),
        speculative_config=None,
        cudagraph_dispatcher=mock.MagicMock(),
    )
    r._check_and_update_cudagraph_mode = mock.MagicMock()

    group_spec = SimpleNamespace(layer_names=["l1", "l2"], kv_cache_spec=uniform)
    cfg = SimpleNamespace(kv_cache_groups=[group_spec])

    with mock.patch.object(M, "get_layers_from_vllm_config", return_value=layers), \
         mock.patch.object(M, "AttentionGroup", FakeAttentionGroup), \
         mock.patch.object(M, "UniformTypeKVCacheSpecs", UniformTypeKVCacheSpecs), \
         mock.patch.object(M, "create_fast_prefill_custom_backend", fake_create_fast_prefill), \
         mock.patch.object(M, "check_attention_cp_compatibility"):
        r.initialize_attn_backend(cfg)

    assert fast_calls == [("FastPrefill", base)]
    # l1 -> fast backend with num_heads fallback 0; l2 -> base with 2 heads.
    backends = sorted(g.backend.name for g in r.attn_groups[0])
    assert backends == ["Base", "FastPrefill"]
    assert all(g.spec is sub_spec for g in r.attn_groups[0])


# ---------------------------------------------------------------------------
# _check_and_update_cudagraph_mode
# ---------------------------------------------------------------------------


def _cg_support_setup():
    ns = SimpleNamespace

    class FakeBackend:
        def __init__(self, name, support_value):
            self.__name__ = name
            self._support = ns(value=support_value)

        def get_builder_cls(self):
            backend = self

            class Builder:
                @staticmethod
                def get_cudagraph_support(vllm_config, kv_cache_spec):
                    return backend._support

            return Builder

    return FakeBackend, ns


def test_check_and_update_cudagraph_mode_min_and_dispatch():
    FakeBackend, ns = _cg_support_setup()
    b_hi = FakeBackend("High", 5)
    b_lo = FakeBackend("Low", 1)
    comp = SimpleNamespace(
        resolve_cudagraph_mode_and_sizes=mock.MagicMock(return_value="MODE")
    )
    dispatcher = mock.MagicMock()
    r = make_runner(
        vllm_config="CFG",
        compilation_config=comp,
        uniform_decode_query_len=3,
        kv_cache_config="KVCFG",
        max_num_reqs=9,
        cudagraph_dispatcher=dispatcher,
        speculative_config=None,
    )
    groups = [SimpleNamespace(kv_cache_spec="s1"), SimpleNamespace(kv_cache_spec="s2")]
    with mock.patch.object(M, "AttentionCGSupport", ns(ALWAYS=ns(value=5))):
        r._check_and_update_cudagraph_mode([{b_hi}, {b_lo}], groups, is_profiling=True)

    args = comp.resolve_cudagraph_mode_and_sizes.call_args
    assert args.args[0] is not None  # resolved min support object
    assert args.args[1] == "Low"  # min support backend name
    assert args.kwargs["is_profiling"] is True
    dispatcher.initialize_cudagraph_keys.assert_called_once_with("MODE", 3)


def test_check_and_update_cudagraph_mode_drafter_branch():
    FakeBackend, ns = _cg_support_setup()
    b = FakeBackend("Only", 5)
    comp = SimpleNamespace(
        resolve_cudagraph_mode_and_sizes=mock.MagicMock(return_value="MODE")
    )
    dispatcher = mock.MagicMock()

    class FakeEagle:
        def __init__(self):
            self.initialize_cudagraph_keys = mock.MagicMock()

    drafter = FakeEagle()
    spec = _spec(use_eagle=lambda: True)
    r = make_runner(
        vllm_config="CFG",
        compilation_config=comp,
        uniform_decode_query_len=1,
        kv_cache_config="KVCFG",
        max_num_reqs=2,
        cudagraph_dispatcher=dispatcher,
        speculative_config=spec,
        drafter=drafter,
    )
    groups = [SimpleNamespace(kv_cache_spec="s")]
    with mock.patch.object(M, "AttentionCGSupport", ns(ALWAYS=ns(value=5))), \
         mock.patch.object(M, "EagleProposer", FakeEagle):
        r._check_and_update_cudagraph_mode([{b}], groups)

    drafter.initialize_cudagraph_keys.assert_called_once_with("MODE")


# ---------------------------------------------------------------------------
# reload_weights
# ---------------------------------------------------------------------------


def _reload_runner(**overrides):
    params = {"w1": torch.nn.Parameter(torch.zeros(2)), "m.w2": torch.nn.Parameter(torch.ones(2))}
    model = SimpleNamespace(
        named_parameters=lambda: list(params.items()),
        load_weights=mock.MagicMock(return_value={"w1", "m.w2"}),
    )
    attrs = dict(
        get_model=lambda: model,
        lora_config=None,
        load_config=SimpleNamespace(load_format="safetensors"),
        model_config=SimpleNamespace(
            model="orig", revision="rev", quantization=None
        ),
        reset_lora_state=mock.MagicMock(),
        reset_encoder_cache=mock.MagicMock(),
        reset_mm_cache=mock.MagicMock(),
    )
    attrs.update(overrides)
    r = make_runner(**attrs)
    r._model = model
    return r


def test_reload_weights_checkpoint_iterator():
    r = _reload_runner()
    it = iter([("w1", torch.zeros(2))])
    with mock.patch.object(M, "initialize_layerwise_reload") as init, \
         mock.patch.object(M, "finalize_layerwise_reload") as fin:
        r.reload_weights(weights_iterator=it, is_checkpoint_format=True)
    init.assert_called_once_with(r._model)
    fin.assert_called_once_with(r._model, r.model_config)
    r._model.load_weights.assert_called_once_with(it)
    r.reset_lora_state.assert_called_once()
    r.reset_encoder_cache.assert_called_once()
    r.reset_mm_cache.assert_called_once()


def test_reload_weights_kernel_format_copies_params_and_lora_names():
    r = _reload_runner(lora_config=SimpleNamespace())
    params = dict(r._model.named_parameters())

    def fake_get_param(model, name):
        # mirrors the real name normalization done by reload_weights
        return params[name.replace(".base_layer.", ".")]

    new_w = torch.full((2,), 7.0)
    with mock.patch.object(M, "initialize_layerwise_reload") as init, \
         mock.patch.object(M, "finalize_layerwise_reload"), \
         mock.patch.object(M, "_get_parameter_for_reload", fake_get_param):
        r.reload_weights(
            weights_iterator=[("w1", new_w)], is_checkpoint_format=False
        )
    # kernel-format path never calls the layerwise helpers
    init.assert_not_called()
    assert torch.equal(params["w1"], new_w)


def test_reload_weights_from_disk_updates_model_config():
    r = _reload_runner()
    loader = SimpleNamespace(
        get_all_weights=mock.MagicMock(return_value=[("w1", torch.zeros(2))])
    )
    with mock.patch.object(M, "get_model_loader", return_value=loader), \
         mock.patch.object(M, "initialize_layerwise_reload"), \
         mock.patch.object(M, "finalize_layerwise_reload"):
        r.reload_weights(weights_path="newpath", is_checkpoint_format=True)
    assert r.model_config.model == "newpath"
    assert r.model_config.revision is None
    loader.get_all_weights.assert_called_once()


def test_reload_weights_disk_loader_unsupported_raises():
    r = _reload_runner()
    with mock.patch.object(M, "get_model_loader", return_value=SimpleNamespace()):
        with pytest.raises(NotImplementedError):
            r.reload_weights(is_checkpoint_format=True)


def test_reload_weights_warns_on_missing_weights():
    r = _reload_runner()
    r._model.load_weights = mock.MagicMock(return_value={"w1"})  # missing m.w2
    with mock.patch.object(M, "initialize_layerwise_reload"), \
         mock.patch.object(M, "finalize_layerwise_reload"), \
         mock.patch.object(M.logger, "warning") as warn:
        r.reload_weights(weights_iterator=[("w1", torch.zeros(2))])
    assert warn.called


# ---------------------------------------------------------------------------
# _get_prompt_logprobs_dict
# ---------------------------------------------------------------------------


class _FakeLogprobsTensors:
    def __init__(self, rows, cols):
        self.logprob_token_ids = torch.zeros(rows, cols, dtype=torch.int64)
        self.logprobs = torch.zeros(rows, cols)
        self.selected_token_ranks = torch.zeros(rows, cols, dtype=torch.int64)

    @staticmethod
    def empty_cpu(rows, cols):
        return _FakeLogprobsTensors(rows, cols)


def _prompt_runner(num_prompt_logprobs, prompt_token_ids, logprobs_mode="raw_logits",
                   existing=None, req_index=0, query_start=0):
    req = SimpleNamespace(
        prompt_token_ids=prompt_token_ids,
        in_progress_prompt_logprobs_cpu=existing,
        num_computed_tokens=0,
    )
    r = make_runner(
        num_prompt_logprobs=num_prompt_logprobs,
        requests={"r": req},
        input_batch=SimpleNamespace(req_id_to_index={"r": req_index}),
        query_start_loc=SimpleNamespace(np=np.array([query_start, query_start + 5])),
        model_config=SimpleNamespace(logprobs_mode=logprobs_mode),
        _sync_device=mock.MagicMock(),
    )
    return r, req


def _prompt_collaborators(logits_cols=4):
    def h2d(ids, device=None):
        return torch.tensor(list(ids))

    model = SimpleNamespace(
        compute_logits=mock.MagicMock(
            side_effect=lambda h: torch.zeros(h.shape[0], logits_cols)
        )
    )
    sampler = SimpleNamespace(
        compute_logprobs=lambda x: x,
        gather_logprobs=mock.MagicMock(
            side_effect=lambda scores, n, tgt: (
                torch.zeros(len(tgt), n + 1, dtype=torch.int64),
                torch.zeros(len(tgt), n + 1),
                torch.zeros(len(tgt), n + 1, dtype=torch.int64),
                None,
            )
        ),
    )
    return model, sampler, h2d


def test_get_prompt_logprobs_dict_empty():
    r, _ = _prompt_runner({}, [1, 2, 3])
    assert r._get_prompt_logprobs_dict(torch.zeros(2, 4), {}) == {}


def test_get_prompt_logprobs_dict_skips_preempted_and_embeds():
    r, req = _prompt_runner({"r": 2}, None)
    model, sampler, h2d = _prompt_collaborators()
    r.requests["gone"] = SimpleNamespace(
        prompt_token_ids=[1], in_progress_prompt_logprobs_cpu=None, num_computed_tokens=0
    )
    r.num_prompt_logprobs = {"r": 2, "gone": 2}
    with mock.patch.object(M, "async_tensor_h2d", h2d), \
         mock.patch.object(M, "LogprobsTensors", _FakeLogprobsTensors):
        # "r" is scheduled but prompt embeddings -> skipped; "gone" is absent
        # from the schedule (preempted) -> skipped.
        out = r._get_prompt_logprobs_dict(torch.zeros(4, 4), {"r": 1})
    assert out == {}
    r._sync_device.assert_not_called()


def test_get_prompt_logprobs_dict_final_chunk_raw_logits():
    r, req = _prompt_runner({"r": 2}, [1, 2, 3, 4], logprobs_mode="raw_logits")
    model, sampler, h2d = _prompt_collaborators()
    r.model = model
    r.sampler = sampler
    # num_computed_tokens=0 -> start_tok=1, remaining=3, num_tokens=5 > 3.
    with mock.patch.object(M, "async_tensor_h2d", h2d), \
         mock.patch.object(M, "LogprobsTensors", _FakeLogprobsTensors):
        out = r._get_prompt_logprobs_dict(torch.zeros(4, 4), {"r": 5})
    assert "r" in out
    assert req.in_progress_prompt_logprobs_cpu is None
    r._sync_device.assert_called_once()
    sampler.gather_logprobs.assert_called_once()


def test_get_prompt_logprobs_dict_chunk_processed_logprobs():
    existing = _FakeLogprobsTensors(3, 3)
    r, req = _prompt_runner(
        {"r": 2}, [1, 2, 3, 4], logprobs_mode="processed_logprobs", existing=existing
    )
    model, sampler, h2d = _prompt_collaborators()
    r.model = model
    r.sampler = sampler
    # num_tokens=1 <= remaining(3) -> partial chunk, no completion, no sync.
    with mock.patch.object(M, "async_tensor_h2d", h2d), \
         mock.patch.object(M, "LogprobsTensors", _FakeLogprobsTensors):
        out = r._get_prompt_logprobs_dict(torch.zeros(4, 4), {"r": 1})
    assert out == {}
    assert req.in_progress_prompt_logprobs_cpu is existing
    sampler.compute_logprobs.assert_called_once()
    r._sync_device.assert_not_called()


# ---------------------------------------------------------------------------
# init_routed_experts_capturer
# ---------------------------------------------------------------------------


def test_init_routed_experts_capturer():
    class FakeCapturer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.device_buffer = torch.zeros(4, 3, dtype=torch.float32)

    bound = []
    r = make_runner(
        model_config=SimpleNamespace(enable_return_routed_experts=True),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=16),
        vllm_config="CFG",
        kv_cache_config="KVCFG",
        model="MODEL",
        device=torch.device("cpu"),
    )
    with mock.patch.object(M, "RoutedExpertsCapturer", FakeCapturer), \
         mock.patch.object(M, "bind_routed_experts_capturer",
                           side_effect=lambda m, c: bound.append((m, c))), \
         mock.patch.object(M, "PIN_MEMORY", False):
        r.init_routed_experts_capturer()

    assert bound and bound[0][0] == "MODEL"
    assert r.routed_experts_initialized is True
    assert tuple(r.routed_experts_cpu.shape) == (4, 3)
    assert r.routed_experts_slot_mapping_cpu.shape == (16,)
    assert r.routed_experts_slot_mapping_cpu.dtype == torch.int64
    assert r.routed_experts_slot_mapping_device.shape == (16,)


# ---------------------------------------------------------------------------
# profile_cudagraph_memory
# ---------------------------------------------------------------------------


def test_profile_cudagraph_memory_no_graphs_early_return():
    r = make_runner(
        vllm_config="CFG",
        cudagraph_dispatcher=SimpleNamespace(get_capture_descs=lambda: []),
    )
    r._init_minimal_kv_cache_for_profiling = mock.MagicMock()
    r._cleanup_profiling_kv_cache = mock.MagicMock()
    r._create_encoder_cudagraph_manager = mock.MagicMock(return_value=None)
    with mock.patch.object(M, "set_current_vllm_config", _fake_ctx):
        got = r.profile_cudagraph_memory()
    assert got == 0
    r._cleanup_profiling_kv_cache.assert_called_once()


def test_profile_cudagraph_memory_single_graph_mocked():
    desc = SimpleNamespace(num_tokens=8, uniform=True)
    mode = SimpleNamespace(name="FULL")
    dispatcher = SimpleNamespace(
        get_capture_descs=lambda: [(mode, [desc])],
        cudagraph_keys={},
        keys_initialized=True,
    )
    r = make_runner(
        vllm_config="CFG",
        device=torch.device("cpu"),
        max_model_len=128,
        max_num_tokens=256,
        lora_config=None,
        graph_runtime=None,
        cudagraph_dispatcher=dispatcher,
        _kl_guard_on=None,
    )
    r._init_minimal_kv_cache_for_profiling = mock.MagicMock()
    r._cleanup_profiling_kv_cache = mock.MagicMock()
    r._create_encoder_cudagraph_manager = mock.MagicMock(return_value=None)
    r._warmup_and_capture = mock.MagicMock()
    r.maybe_remove_all_loras = mock.MagicMock()

    accel = SimpleNamespace(
        synchronize=lambda: None,
        empty_cache=lambda: None,
        get_memory_info=lambda: (1000, 2000),
    )

    class _Wrapper:
        _all_instances = []

        @classmethod
        def clear_all_graphs(cls):
            pass

    with mock.patch.object(M, "set_current_vllm_config", _fake_ctx), \
         mock.patch.object(M, "current_platform",
                           SimpleNamespace(device_type="cuda",
                                           graph_pool_handle=lambda: "POOL",
                                           is_rocm=lambda: False)), \
         mock.patch.object(M, "graph_capture", _fake_ctx), \
         mock.patch.object(M, "set_cudagraph_capturing_enabled"), \
         mock.patch.object(M, "CUDAGraphWrapper", _Wrapper), \
         mock.patch.object(M, "GraphWrapper", _Wrapper), \
         mock.patch.object(M, "BreakableCUDAGraphWrapper", _Wrapper), \
         mock.patch.object(M, "CUDAGraphMode", SimpleNamespace(FULL="FULL")), \
         mock.patch.object(torch, "accelerator", accel, create=True):
        got = r.profile_cudagraph_memory()

    r._warmup_and_capture.assert_called_once()
    r._cleanup_profiling_kv_cache.assert_called_once()
    assert isinstance(got, int)


# ---------------------------------------------------------------------------
# propose_draft_token_ids
# ---------------------------------------------------------------------------


def _propose_base(**overrides):
    attrs = dict(
        speculative_config=_spec(),
        input_batch=SimpleNamespace(
            num_tokens_no_spec=np.array([1]),
            token_ids_cpu=np.array([[1, 2]]),
            req_id_to_index={"r": 0},
            req_ids=np.array(["r"]),
            num_reqs=1,
        ),
        token_ids_gpu_tensor=torch.zeros(1, 2, dtype=torch.int64),
        num_tokens_no_spec_gpu=np.array([1]),
        discard_request_mask=SimpleNamespace(gpu=torch.zeros(1, dtype=torch.bool)),
        requests={"r": SimpleNamespace()},
        input_ids=SimpleNamespace(gpu=torch.zeros(8, dtype=torch.int64)),
        device=torch.device("cpu"),
        use_aux_hidden_state_outputs=False,
        supports_mm_inputs=False,
        _copy_valid_sampled_token_count=mock.MagicMock(),
        _num_valid_draft_tokens=None,
        _num_valid_draft_tokens_cpu=torch.zeros(1),
        _num_valid_draft_tokens_copy_stream="STREAM",
        _num_valid_draft_tokens_event="EVENT",
        get_model=lambda: SimpleNamespace(get_mtp_target_hidden_states=lambda: None),
    )
    attrs.update(overrides)
    r = make_runner(**attrs)
    r._get_positions = mock.MagicMock(return_value=torch.zeros(8, dtype=torch.int64))
    return r


_FakeSchedulerOutput = SimpleNamespace


def _sched(**overrides):
    values = dict(
        total_num_scheduled_tokens=4,
        num_spec_tokens_to_schedule=2,
        num_scheduled_tokens={"r": 4},
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def test_propose_draft_token_ids_ngram():
    ngram_mod = pytest.importorskip("vllm.v1.spec_decode.ngram_proposer")
    try:
        drafter = object.__new__(ngram_mod.NgramProposer)
        drafter.propose = mock.MagicMock(return_value=[[9]])
    except Exception:
        pytest.skip("NgramProposer cannot be built without __init__")
    r = _propose_base(speculative_config=_spec(method="ngram"), drafter=drafter)
    out = r.propose_draft_token_ids(
        _sched(), [[1]], None, torch.zeros(4, 2), torch.zeros(1, 2), None, None, None, None
    )
    assert out == [[9]]
    drafter.propose.assert_called_once()


def test_propose_draft_token_ids_custom_class():
    drafter = mock.MagicMock(return_value=None)
    drafter.propose = mock.MagicMock(return_value=[[7]])
    r = _propose_base(
        speculative_config=_spec(method="custom_class"), drafter=drafter
    )
    out = r.propose_draft_token_ids(
        _sched(), [[1]], None, torch.zeros(4, 2), torch.zeros(1, 2), None, None, None, None
    )
    assert out == [[7]]


def test_propose_draft_token_ids_gpu_ngram():
    class FakeGPU:
        def __init__(self):
            self.update_token_ids_ngram = mock.MagicMock(
                return_value=(
                    torch.zeros(1, dtype=torch.int64),
                    torch.zeros(1, dtype=torch.int64),
                    torch.zeros(1, 2, dtype=torch.int64),
                )
            )
            self.propose = mock.MagicMock(
                return_value=([[3]], torch.ones(1, dtype=torch.int64))
            )

    drafter = FakeGPU()
    r = _propose_base(
        speculative_config=_spec(method="gpu", use_ngram_gpu=lambda: True),
        drafter=drafter,
        _num_valid_draft_tokens_cpu=torch.zeros(1),
    )
    with mock.patch.object(M, "NgramProposerGPU", FakeGPU), \
         mock.patch.object(M, "copy_num_valid_draft_tokens") as copy_fn:
        out = r.propose_draft_token_ids(
            _sched(), torch.zeros(1, 2, dtype=torch.int64), None,
            torch.zeros(4, 2), torch.zeros(1, 2), None, None, None, None,
        )
    assert out == [[3]]
    r._copy_valid_sampled_token_count.assert_called_once()
    copy_fn.assert_called_once()


def test_propose_draft_token_ids_suffix():
    class FakeSuffix:
        def __init__(self):
            self.propose = mock.MagicMock(return_value=[[4]])

    drafter = FakeSuffix()
    r = _propose_base(speculative_config=_spec(method="suffix"), drafter=drafter)
    with mock.patch.object(M, "SuffixDecodingProposer", FakeSuffix):
        out = r.propose_draft_token_ids(
            _sched(), [[1]], None, torch.zeros(4, 2), torch.zeros(1, 2), None, None, None, None
        )
    assert out == [[4]]


def test_propose_draft_token_ids_medusa_direct_hidden():
    class FakeMedusa:
        def __init__(self):
            self.propose = mock.MagicMock(return_value=[[5]])

    drafter = FakeMedusa()
    r = _propose_base(speculative_config=_spec(method="medusa"), drafter=drafter)
    sample_hidden = torch.zeros(1, 2)
    with mock.patch.object(M, "MedusaProposer", FakeMedusa):
        out = r.propose_draft_token_ids(
            _sched(), [[1]], None, torch.zeros(4, 2), sample_hidden, None, None, None, None
        )
    assert out == [[5]]
    kwargs = drafter.propose.call_args.kwargs
    assert kwargs["target_hidden_states"] is sample_hidden


def test_propose_draft_token_ids_medusa_gather_indices():
    class FakeMedusa:
        def __init__(self):
            self.propose = mock.MagicMock(return_value=[[6]])

    drafter = FakeMedusa()
    r = _propose_base(speculative_config=_spec(method="medusa"), drafter=drafter)
    sample_hidden = torch.zeros(4, 2)
    meta = SimpleNamespace(num_draft_tokens=[1, 2])
    with mock.patch.object(M, "MedusaProposer", FakeMedusa), \
         mock.patch.object(M, "async_tensor_h2d",
                           lambda idx, device=None: torch.tensor(idx)) as h2d:
        out = r.propose_draft_token_ids(
            _sched(), [[1, 2], [3, 4, 5]], None, torch.zeros(4, 2),
            sample_hidden, None, meta, None, None,
        )
    assert out == [[6]]
    h2d.assert_called_once()


def test_propose_draft_token_ids_extract_hidden_requires_aux():
    class FakeExtract:
        def __init__(self):
            self.propose = mock.MagicMock()

    drafter = FakeExtract()
    r = _propose_base(
        speculative_config=_spec(uses_extract_hidden_states=lambda: True,
                                 method="extract_hidden_states"),
        drafter=drafter,
    )
    with mock.patch.object(M, "ExtractHiddenStatesProposer", FakeExtract):
        with pytest.raises(ValueError):
            r.propose_draft_token_ids(
                _sched(), torch.zeros(1, 2), None, torch.zeros(4, 2),
                torch.zeros(1, 2), None, None, None, None,
            )


def test_propose_draft_token_ids_extract_hidden_main():
    class FakeExtract:
        def __init__(self):
            self.propose = mock.MagicMock(return_value=[[8]])
            self.prepare_next_token_ids_padded = mock.MagicMock(
                return_value=(torch.zeros(1, dtype=torch.int64),
                              torch.ones(1, dtype=torch.int64))
            )

    drafter = FakeExtract()
    r = _propose_base(
        speculative_config=_spec(uses_extract_hidden_states=lambda: True,
                                 method="extract_hidden_states"),
        drafter=drafter,
        use_aux_hidden_state_outputs=True,
    )
    aux = [torch.zeros(8, 2), torch.zeros(8, 2)]
    with mock.patch.object(M, "ExtractHiddenStatesProposer", FakeExtract):
        out = r.propose_draft_token_ids(
            _sched(), torch.zeros(1, 2), None, torch.zeros(4, 2),
            torch.zeros(1, 2), aux, None, "CAM", None,
        )
    assert out == [[8]]
    drafter.prepare_next_token_ids_padded.assert_called_once()
    r._copy_valid_sampled_token_count.assert_called_once()


def test_propose_draft_token_ids_eagle_unpadded_no_spec_meta():
    class FakeEagle:
        def __init__(self):
            self.prepare_next_token_ids_cpu = mock.MagicMock(
                return_value=torch.zeros(1, dtype=torch.int64)
            )
            self.propose = mock.MagicMock(return_value=[[10]])

    drafter = FakeEagle()
    drafter.supports_mm_inputs = False
    r = _propose_base(
        speculative_config=_spec(use_eagle=lambda: True,
                                 disable_padded_drafter_batch=True),
        drafter=drafter,
    )
    with mock.patch.object(M, "EagleProposer", FakeEagle):
        out = r.propose_draft_token_ids(
            _sched(), [[1]], None, torch.zeros(4, 2), torch.zeros(1, 2),
            None, None, "CAM", None,
        )
    assert out == [[10]]
    drafter.prepare_next_token_ids_cpu.assert_called_once()
    r._copy_valid_sampled_token_count.assert_not_called()


def test_propose_draft_token_ids_eagle_padded_with_spec_meta_and_mm():
    class FakeEagle:
        def __init__(self):
            self.prepare_next_token_ids_padded = mock.MagicMock(
                return_value=(torch.zeros(1, dtype=torch.int64),
                              torch.ones(1, dtype=torch.int64))
            )
            self.prepare_inputs_padded = mock.MagicMock(
                return_value=(SimpleNamespace(num_actual_tokens=4),
                              torch.zeros(1, dtype=torch.int64),
                              torch.zeros(1, dtype=torch.int64))
            )
            self.propose = mock.MagicMock(return_value=[[11]])
            self.take_last_draft_probs = mock.MagicMock(return_value="PROBS")
            self.supports_mm_inputs = True

    drafter = FakeEagle()
    r = _propose_base(
        speculative_config=_spec(use_eagle=lambda: True,
                                 disable_padded_drafter_batch=False),
        drafter=drafter,
        use_aux_hidden_state_outputs=True,
        supports_mm_inputs=True,
    )
    r._gather_mm_embeddings = mock.MagicMock(return_value="MM")
    meta = SimpleNamespace(num_draft_tokens=[1])
    aux = [torch.zeros(8, 2)]
    with mock.patch.object(M, "EagleProposer", FakeEagle):
        out = r.propose_draft_token_ids(
            _sched(), torch.zeros(1, 2), None, torch.zeros(4, 2),
            torch.zeros(1, 2), aux, meta, "CAM", None,
        )
    assert out == [[11]]
    r._gather_mm_embeddings.assert_called_once()
    assert r._draft_probs == "PROBS"
    assert list(r._draft_prob_req_ids) == ["r"]
    kwargs = drafter.propose.call_args.kwargs
    assert kwargs["mm_embed_inputs"] == "MM"
    assert kwargs["target_hidden_states"].shape[-1] == 2
