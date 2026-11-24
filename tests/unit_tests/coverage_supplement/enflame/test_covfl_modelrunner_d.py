"""Coverage tests for model lifecycle / dummy / capture / kv-cache in
``vllm_fl/worker/model_runner.py`` (part _d only).

Self-contained: pytest + unittest.mock + stdlib + torch/numpy.  No conftest.
``ModelRunnerFL.__init__`` is bypassed with ``object.__new__`` and only the
attributes each method touches are populated.  Nothing touches a real
accelerator: every device / dist / loader / capture call is mocked.
"""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
M = pytest.importorskip("vllm_fl.worker.model_runner")

from unittest import mock  # noqa: E402


def runner(**attrs):
    r = object.__new__(M.ModelRunnerFL)
    r.__dict__.update(attrs)
    return r


def accel(get_memory_info=None):
    return SimpleNamespace(
        synchronize=mock.Mock(),
        empty_cache=mock.Mock(),
        get_memory_info=get_memory_info
        or mock.Mock(return_value=(1000, 2000)),
    )


class _FakeDMP:
    def __init__(self, *a, **k):
        self.consumed_memory = 123

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _OOMDMP:
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        raise torch.OutOfMemoryError("boom")

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Eagle3 aux hidden state helpers
# ---------------------------------------------------------------------------
def test_get_eagle3_aux_layers_variants():
    r = runner(speculative_config=None)
    assert r._get_eagle3_aux_layers_from_config() is None

    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=[1, 2])
    r = runner(speculative_config=SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf)))
    assert r._get_eagle3_aux_layers_from_config() == (1, 2)

    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=None,
                         dflash_config={"target_layer_ids": [3]})
    r = runner(speculative_config=SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf)))
    assert r._get_eagle3_aux_layers_from_config() == (4,)

    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids=None,
                         dflash_config=None,
                         eagle_config={"eagle_aux_hidden_state_layer_ids": [5]})
    r = runner(speculative_config=SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf)))
    assert r._get_eagle3_aux_layers_from_config() == (5,)

    hf = SimpleNamespace(eagle_aux_hidden_state_layer_ids="nope",
                         dflash_config=None,
                         eagle_config=None)
    r = runner(speculative_config=SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=hf)))
    assert r._get_eagle3_aux_layers_from_config() is None


def test_setup_eagle3_aux_hidden_states():
    r = runner(use_aux_hidden_state_outputs=False)
    r._setup_eagle3_aux_hidden_state_outputs()  # early return

    model = SimpleNamespace(
        get_eagle3_default_aux_hidden_state_layers=mock.Mock(return_value=(7,)),
        set_aux_hidden_state_layers=mock.Mock(),
    )
    r = runner(use_aux_hidden_state_outputs=True)
    r.get_model = lambda: model
    with mock.patch.object(M, "supports_eagle3", return_value=False):
        with pytest.raises(RuntimeError):
            r._setup_eagle3_aux_hidden_state_outputs()

    r._get_eagle3_aux_layers_from_config = mock.Mock(return_value=None)
    with mock.patch.object(M, "supports_eagle3", return_value=True):
        r._setup_eagle3_aux_hidden_state_outputs()
    model.set_aux_hidden_state_layers.assert_called_once_with((7,))

    model.set_aux_hidden_state_layers.reset_mock()
    r._get_eagle3_aux_layers_from_config = mock.Mock(return_value=(1,))
    with mock.patch.object(M, "supports_eagle3", return_value=True):
        r._setup_eagle3_aux_hidden_state_outputs()
    model.set_aux_hidden_state_layers.assert_called_once_with((1,))


# ---------------------------------------------------------------------------
# _dummy_run
# ---------------------------------------------------------------------------
def _fake_forward(**kw):
    ids = kw.get("input_ids")
    n = ids.shape[0] if ids is not None else 1
    return torch.zeros(n, 3)


def _dummy_runner(**over):
    mc = SimpleNamespace(multimodal_config=None, is_encoder_decoder=False,
                         use_mla=False)
    attrs = dict(
        vllm_config=SimpleNamespace(
            model_config=mc,
            parallel_config=SimpleNamespace(num_ubatches=1),
        ),
        model_config=mc,
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        uniform_decode_query_len=1,
        max_num_tokens=16,
        device=torch.device("cpu"),
        use_aux_hidden_state_outputs=False,
        enable_prompt_embeds=False,
        supports_mm_inputs=False,
        uses_mrope=False,
        use_ascend_attention=False,
        speculative_config=None,
        model=mock.Mock(side_effect=_fake_forward),
        input_ids=SimpleNamespace(gpu=torch.zeros(16, dtype=torch.int32)),
        positions=torch.zeros(16, dtype=torch.int64),
    )
    attrs.update(over)
    r = runner(**attrs)
    r._determine_batch_execution_and_padding = mock.Mock(
        side_effect=lambda num_tokens, **k: (
            M.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=num_tokens, num_reqs=None),
            False,
            None,
            None,
        )
    )
    r._get_slot_mappings = mock.Mock(return_value=(None, None))
    r._init_model_kwargs = mock.Mock(return_value={})
    r.synchronize_input_prep = lambda: nullcontext()
    r.maybe_dummy_run_with_lora = lambda *a, **k: nullcontext()
    r.maybe_randomize_inputs = lambda *a, **k: nullcontext()
    r._register_layerwise_nvtx_hooks = mock.Mock()
    r.eplb_step = mock.Mock()
    return r


def test_dummy_run_mm_encoder_only():
    mc = SimpleNamespace(multimodal_config=SimpleNamespace(mm_encoder_only=True))
    r = runner(vllm_config=SimpleNamespace(model_config=mc))
    hs, tok = r._dummy_run(4)
    assert hs.numel() == 0 and tok.numel() == 0


@mock.patch.object(M, "maybe_create_ubatch_slices", lambda *a, **k: (None, None))
@mock.patch.object(M, "get_dcp_dummy_context_len", lambda *a, **k: 0)
@mock.patch.object(M, "prepare_dcp_dummy_context_metadata", mock.Mock())
def test_dummy_run_main():
    r = _dummy_runner()
    with mock.patch.object(M, "set_forward_context", lambda *a, **k: nullcontext()):
        hs, tok = r._dummy_run(8, cudagraph_runtime_mode=M.CUDAGraphMode.NONE)
    assert hs.shape == (8, 3)
    assert tok.shape == (4, 3)
    r.eplb_step.assert_called_once()
    r._register_layerwise_nvtx_hooks.assert_called_once()


@mock.patch.object(M, "maybe_create_ubatch_slices", lambda *a, **k: (None, None))
@mock.patch.object(M, "get_dcp_dummy_context_len", lambda *a, **k: 0)
@mock.patch.object(M, "prepare_dcp_dummy_context_metadata", mock.Mock())
def test_dummy_run_force_attention():
    r = _dummy_runner(
        optimistic_seq_lens_cpu=torch.zeros(4, dtype=torch.int32),
        seq_lens=torch.zeros(4, dtype=torch.int32),
        query_start_loc=SimpleNamespace(
            np=np.zeros(5, dtype=np.int32), copy_to_gpu=mock.Mock()),
        input_batch=SimpleNamespace(
            block_table=SimpleNamespace(commit_block_table=mock.Mock())),
        use_ascend_attention=False,
    )
    r._get_cumsum_and_arange = mock.Mock(return_value=np.array([1, 2, 3, 4]))
    r._build_attention_metadata = mock.Mock(return_value=(SimpleNamespace(), None))
    with mock.patch.object(M, "set_forward_context", lambda *a, **k: nullcontext()):
        hs, _ = r._dummy_run(8, cudagraph_runtime_mode=M.CUDAGraphMode.NONE,
                             force_attention=True)
    assert hs.shape == (8, 3)
    r._build_attention_metadata.assert_called_once()
    r.input_batch.block_table.commit_block_table.assert_called_once()


@mock.patch.object(M, "maybe_create_ubatch_slices", lambda *a, **k: (None, None))
@mock.patch.object(M, "get_dcp_dummy_context_len", lambda *a, **k: 0)
@mock.patch.object(M, "prepare_dcp_dummy_context_metadata", mock.Mock())
def test_dummy_run_mixed_batch():
    r = _dummy_runner()
    with mock.patch.object(M, "set_forward_context", lambda *a, **k: nullcontext()):
        hs, tok = r._dummy_run(8, cudagraph_runtime_mode=M.CUDAGraphMode.NONE,
                               create_mixed_batch=True)
    assert hs.shape == (8, 3)
    assert tok.shape == (4, 3)


@mock.patch.object(M, "maybe_create_ubatch_slices", lambda *a, **k: (None, None))
@mock.patch.object(M, "get_dcp_dummy_context_len", lambda *a, **k: 0)
@mock.patch.object(M, "prepare_dcp_dummy_context_metadata", mock.Mock())
def test_dummy_run_uniform_decode():
    r = _dummy_runner(uniform_decode_query_len=2)
    with mock.patch.object(M, "set_forward_context", lambda *a, **k: nullcontext()):
        hs, tok = r._dummy_run(8, cudagraph_runtime_mode=M.CUDAGraphMode.NONE,
                               uniform_decode=True)
    assert hs.shape == (8, 3)


@mock.patch.object(M, "maybe_create_ubatch_slices", lambda *a, **k: (None, None))
@mock.patch.object(M, "get_dcp_dummy_context_len", lambda *a, **k: 0)
@mock.patch.object(M, "prepare_dcp_dummy_context_metadata", mock.Mock())
def test_dummy_run_drafter():
    class _Drafter:
        def __init__(self):
            self.dummy_run = mock.Mock()

    spec = SimpleNamespace(
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        enforce_eager=False,
    )
    r = _dummy_runner(speculative_config=spec,
                      compilation_config=SimpleNamespace(
                          cudagraph_specialize_lora=False),
                      drafter=_Drafter())
    with mock.patch.object(M, "EagleProposer", _Drafter), \
         mock.patch.object(M, "set_forward_context",
                           lambda *a, **k: nullcontext()):
        hs, _ = r._dummy_run(8, cudagraph_runtime_mode=M.CUDAGraphMode.NONE)
    assert hs.shape == (8, 3)
    r.drafter.dummy_run.assert_called_once()


# ---------------------------------------------------------------------------
# _dummy_sampler_run
# ---------------------------------------------------------------------------
def test_dummy_sampler_run_mm_encoder_only():
    mc = SimpleNamespace(multimodal_config=SimpleNamespace(mm_encoder_only=True))
    r = runner(vllm_config=SimpleNamespace(model_config=mc))
    out = r._dummy_sampler_run(torch.zeros(2, 3))
    assert out.numel() == 0


def _sampler_runner(**over):
    logits = torch.rand(2, 10)
    sampler = mock.Mock()
    sampler.logprobs_mode = "raw"
    sampler.return_value = "sampler-out"
    attrs = dict(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)),
        device=torch.device("cpu"),
        model=SimpleNamespace(compute_logits=mock.Mock(return_value=logits)),
        sampler=sampler,
        speculative_config=None,
    )
    attrs.update(over)
    return runner(**attrs)


@mock.patch.object(M, "replace", lambda obj, **kw: obj)
@mock.patch.object(M, "PROCESSED_LOGPROBS_MODES", set())
@mock.patch("vllm_fl.platform.PlatformFL",
            SimpleNamespace(vendor_name="other"))
def test_dummy_sampler_run_main():
    r = _sampler_runner()
    with mock.patch.object(M, "SamplingMetadata",
                           lambda **kw: SimpleNamespace(**kw)):
        out = r._dummy_sampler_run(torch.zeros(2, 3))
    assert out == "sampler-out"
    assert r.sampler.call_count == 2


@mock.patch.object(M, "replace", lambda obj, **kw: obj)
@mock.patch("vllm_fl.platform.PlatformFL",
            SimpleNamespace(vendor_name="other"))
def test_dummy_sampler_run_oom():
    r = _sampler_runner()
    r.sampler.side_effect = RuntimeError("out of memory")
    with mock.patch.object(M, "SamplingMetadata",
                           lambda **kw: SimpleNamespace(**kw)):
        with pytest.raises(RuntimeError, match="out of memory"):
            r._dummy_sampler_run(torch.zeros(2, 3))


@mock.patch.object(M, "replace", lambda obj, **kw: obj)
@mock.patch.object(M, "PROCESSED_LOGPROBS_MODES", set())
@mock.patch("vllm_fl.platform.PlatformFL",
            SimpleNamespace(vendor_name="other"))
def test_dummy_sampler_run_spec():
    spec = SimpleNamespace(rejection_sample_method="standard",
                           draft_sample_method="probabilistic")
    r = _sampler_runner(speculative_config=spec)
    r.rejection_sampler = mock.Mock()
    sm = SimpleNamespace(make_dummy=mock.Mock(return_value="meta"))
    with mock.patch.object(M, "SamplingMetadata",
                           lambda **kw: SimpleNamespace(**kw)), \
         mock.patch.object(M, "SpecDecodeMetadata", sm), \
         mock.patch.object(M.torch, "accelerator", accel()):
        out = r._dummy_sampler_run(torch.zeros(2, 3))
    assert out == "sampler-out"
    assert r.rejection_sampler.call_count == 2


# ---------------------------------------------------------------------------
# _dummy_pooler_run / task
# ---------------------------------------------------------------------------
class _FakePoolingParams:
    def __init__(self, task=None, **kw):
        self.task = task

    def verify(self, *a, **k):
        return None


class _FakePoolingMetadata:
    def __init__(self, **kw):
        self.kw = kw

    def build_pooling_cursor(self, *a, **k):
        return None


def test_dummy_pooler_run_main():
    r = runner(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)),
        model_config=SimpleNamespace(model="m"),
    )
    r.get_supported_pooling_tasks = mock.Mock(return_value=["embed", "cls"])
    r._dummy_pooler_run_task = mock.Mock(return_value=[torch.zeros(4)])
    out = r._dummy_pooler_run(torch.zeros(4, 3))
    assert r._dummy_pooler_run_task.call_count == 3
    assert out is not None

    r2 = runner(
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(multimodal_config=None)),
        model_config=SimpleNamespace(model="m"),
    )
    r2.get_supported_pooling_tasks = mock.Mock(return_value=[])
    with pytest.raises(RuntimeError):
        r2._dummy_pooler_run(torch.zeros(4, 3))

    mc = SimpleNamespace(multimodal_config=SimpleNamespace(mm_encoder_only=True))
    r3 = runner(vllm_config=SimpleNamespace(model_config=mc))
    assert r3._dummy_pooler_run(torch.zeros(3, 2)).numel() == 0


@mock.patch.object(M, "PoolingParams", _FakePoolingParams)
@mock.patch.object(M, "PoolingMetadata", _FakePoolingMetadata)
@mock.patch.object(M, "PoolingStates", lambda *a, **k: object())
def test_dummy_pooler_run_task_main_and_oom():
    pooler = mock.Mock()
    pooler.get_pooling_updates.return_value = mock.Mock()
    pooler.return_value = [torch.zeros(2)]
    model = SimpleNamespace(pooler=pooler)
    r = runner(
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        device=torch.device("cpu"),
        model_config=SimpleNamespace(model="m"),
    )
    r.get_model = lambda: model
    out = r._dummy_pooler_run_task(torch.zeros(4, 3), "embed")
    assert out is not None
    pooler.assert_called_once()

    pooler.side_effect = RuntimeError("out of memory")
    with pytest.raises(RuntimeError, match="out of memory"):
        r._dummy_pooler_run_task(torch.zeros(4, 3), "embed")


# ---------------------------------------------------------------------------
# profile_run
# ---------------------------------------------------------------------------
def test_profile_run_main():
    r = runner(
        supports_mm_inputs=False,
        model_config=SimpleNamespace(multimodal_config=None),
        is_pooling_model=False,
        encoder_cache={},
    )
    r._dummy_run = mock.Mock(return_value=(torch.zeros(2, 3), torch.zeros(2, 3)))
    r._dummy_sampler_run = mock.Mock(return_value="out")
    r._sync_device = mock.Mock()
    with mock.patch.object(M, "get_pp_group",
                           lambda: SimpleNamespace(is_last_rank=True)):
        r.profile_run()
    r._dummy_sampler_run.assert_called_once()
    r._sync_device.assert_called_once()

    r.is_pooling_model = True
    r._dummy_pooler_run = mock.Mock(return_value="pool")
    with mock.patch.object(M, "get_pp_group",
                           lambda: SimpleNamespace(is_last_rank=True)):
        r.profile_run()
    r._dummy_pooler_run.assert_called_once()


def test_profile_run_mm_profiling():
    budget = SimpleNamespace(
        get_encoder_budget=lambda: 10,
        mm_max_toks_per_item=True,
        get_modality_with_max_tokens=lambda: "image",
        mm_max_items_per_batch={"image": 2},
    )
    r = runner(
        supports_mm_inputs=True,
        model_config=SimpleNamespace(
            multimodal_config=SimpleNamespace(skip_mm_profiling=False)),
        mm_budget=budget,
        is_pooling_model=False,
        encoder_cache={},
        model=SimpleNamespace(embed_multimodal=mock.Mock(
            return_value=[torch.zeros(2, 3)])),
    )
    r._get_mm_dummy_batch = mock.Mock(return_value={"x": 1})
    r._dummy_run = mock.Mock(return_value=(torch.zeros(2, 3), torch.zeros(2, 3)))
    r._dummy_sampler_run = mock.Mock(return_value="out")
    r._sync_device = mock.Mock()
    with mock.patch.object(M, "sanity_check_mm_encoder_outputs", mock.Mock()), \
         mock.patch.object(M, "get_pp_group",
                           lambda: SimpleNamespace(is_last_rank=True)):
        r.profile_run()
    assert "tmp_0" in r.encoder_cache
    r.model.embed_multimodal.assert_called_once()


# ---------------------------------------------------------------------------
# KV cache init
# ---------------------------------------------------------------------------
def test_init_minimal_kv_cache_for_profiling():
    r = runner(
        max_num_reqs=4,
        compilation_config=SimpleNamespace(max_cudagraph_capture_size=8),
        cache_config=SimpleNamespace(num_gpu_blocks_override=99,
                                     num_gpu_blocks=None),
        vllm_config=SimpleNamespace(),
    )
    r.get_kv_cache_spec = mock.Mock(return_value={})
    r.initialize_kv_cache = mock.Mock()
    with mock.patch.object(M.KVCacheSpecRegistry,
                           "check_kv_cache_spec_registry", mock.Mock()), \
         mock.patch("vllm.v1.core.kv_cache_utils.get_kv_cache_groups",
                    mock.Mock(return_value=[])), \
         mock.patch("vllm.v1.core.kv_cache_utils."
                    "get_kv_cache_config_from_groups",
                    mock.Mock(return_value=SimpleNamespace(num_blocks=7))):
        r._init_minimal_kv_cache_for_profiling()
    r.initialize_kv_cache.assert_called_once()
    assert r.cache_config.num_gpu_blocks == 7
    assert r.cache_config.num_gpu_blocks_override == 99


def test_initialize_kv_cache_tensors():
    r = runner(
        device=torch.device("cpu"),
        cache_config=SimpleNamespace(
            get_resolved_kv_cache_layout=lambda: "NHD"),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(model_type="llama")),
        shared_kv_cache_layers={},
        attn_groups=[],
        speculative_config=None,
        kv_caches=[],
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    with mock.patch("vllm.v1.worker.utils.allocate_kv_cache",
                    mock.Mock(return_value={"a": torch.zeros(1)})), \
         mock.patch("vllm.v1.worker.gpu_model_runner.bind_kv_cache",
                    mock.Mock()), \
         mock.patch("vllm.v1.worker.gpu_model_runner."
                    "map_kv_caches_to_kernel_blocks",
                    mock.Mock(return_value={})):
        out = r.initialize_kv_cache_tensors(SimpleNamespace(kv_cache_groups=[]),
                                            [16])
    assert "a" in out


def _kv_runner(**over):
    attrs = dict(
        vllm_config=SimpleNamespace(mamba_config=object()),
        graph_runtime=None,
        speculative_config=None,
        cross_layers_kv_cache=None,
        cross_layers_attn_backend=None,
        attn_groups=[],
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
    )
    attrs.update(over)
    r = runner(**attrs)
    r.may_add_encoder_only_layers_to_kv_cache_config = mock.Mock()
    r.maybe_add_kv_sharing_layers_to_kv_cache_groups = mock.Mock()
    r.initialize_attn_backend = mock.Mock()
    r.initialize_metadata_builders = mock.Mock()
    r.may_reinitialize_input_batch = mock.Mock()
    r.initialize_kv_cache_tensors = mock.Mock(
        return_value={"a": torch.zeros(1)})
    return r


def test_initialize_kv_cache_main():
    r = _kv_runner()
    cfg = SimpleNamespace(kv_cache_groups=[])
    with mock.patch.object(M, "initialize_mamba_ssu_backend", mock.Mock()), \
         mock.patch.object(M, "prepare_kernel_block_sizes",
                           mock.Mock(return_value=[16])), \
         mock.patch.object(M, "has_kv_transfer_group",
                           mock.Mock(return_value=False)):
        r.initialize_kv_cache(cfg)
    assert r.kv_cache_config is not cfg
    assert r._kernel_block_sizes == [16]
    r.initialize_kv_cache_tensors.assert_called_once()


def test_initialize_kv_cache_transfer_group():
    r = _kv_runner(cross_layers_kv_cache=torch.zeros(1),
                   cross_layers_attn_backend=object())
    grp = mock.Mock()
    with mock.patch.object(M, "initialize_mamba_ssu_backend", mock.Mock()), \
         mock.patch.object(M, "prepare_kernel_block_sizes",
                           mock.Mock(return_value=[16])), \
         mock.patch.object(M, "has_kv_transfer_group",
                           mock.Mock(return_value=True)), \
         mock.patch.object(M, "get_kv_transfer_group",
                           mock.Mock(return_value=grp)), \
         mock.patch.object(M, "copy_kv_blocks", mock.Mock()):
        r.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[]))
    grp.register_cross_layers_kv_cache.assert_called_once()
    grp.set_host_xfer_buffer_ops.assert_called_once()


# ---------------------------------------------------------------------------
# capture_model / warmup / capture cudagraphs
# ---------------------------------------------------------------------------
def test_capture_model_skip_none():
    r = runner(compilation_config=SimpleNamespace(
        cudagraph_mode=M.CUDAGraphMode.NONE))
    assert r.capture_model() == 0


def test_capture_model_main():
    desc = SimpleNamespace(num_tokens=8)
    r = runner(
        compilation_config=SimpleNamespace(
            cudagraph_mode=M.CUDAGraphMode.FULL),
        device=torch.device("cpu"),
        vllm_config=SimpleNamespace(
            profiler_config=SimpleNamespace(capture_torch_profiler=False)),
        cudagraph_dispatcher=SimpleNamespace(
            get_capture_descs=mock.Mock(
                return_value=[(M.CUDAGraphMode.FULL, [desc])])),
        encoder_cudagraph_manager=None,
    )
    r._maybe_init_encoder_cudagraph_manager = mock.Mock()
    r._capture_cudagraphs = mock.Mock()
    r._freeze_gc = lambda: nullcontext()
    mem = mock.Mock(side_effect=[(1000, 2000), (400, 2000)])
    with mock.patch.object(M, "set_cudagraph_capturing_enabled", mock.Mock()), \
         mock.patch.object(M, "graph_capture",
                           lambda *a, **k: nullcontext()), \
         mock.patch.object(M, "lock_workspace", mock.Mock()), \
         mock.patch.object(M, "current_platform",
                           SimpleNamespace(device_type="cuda")), \
         mock.patch("vllm.distributed.parallel_state.get_world_group",
                    return_value=SimpleNamespace(local_rank=0)), \
         mock.patch.object(M.compilation_counter,
                           "num_gpu_runner_capture_triggers", 0), \
         mock.patch.object(M.torch, "accelerator", accel(mem)):
        size = r.capture_model()
    assert size == 600
    r._capture_cudagraphs.assert_called_once()


@mock.patch.object(M, "PROCESSED_LOGPROBS_MODES", set())
def test_warmup_and_capture():
    r = runner(compilation_config=SimpleNamespace(
        cudagraph_num_of_warmups=1))
    r._dummy_run = mock.Mock()
    desc = SimpleNamespace(num_tokens=8, uniform=False, num_active_loras=0)
    with mock.patch.object(M.torch, "accelerator", accel()), \
         mock.patch.object(M.torch.profiler, "record_function",
                           lambda name: nullcontext()):
        r._warmup_and_capture(desc, M.CUDAGraphMode.FULL)
    assert r._dummy_run.call_count == 2


def test_capture_cudagraphs():
    desc = SimpleNamespace(num_tokens=8, uniform=False)
    r = runner(
        parallel_config=SimpleNamespace(use_ubatching=False),
        lora_config=None,
    )
    r._warmup_and_capture = mock.Mock()
    r.maybe_remove_all_loras = mock.Mock()
    with mock.patch.object(M, "is_global_first_rank",
                           mock.Mock(return_value=False)), \
         mock.patch.object(M.torch, "accelerator", accel()):
        r._capture_cudagraphs([desc], M.CUDAGraphMode.FULL)
        r._capture_cudagraphs([], M.CUDAGraphMode.FULL)
    assert r._warmup_and_capture.call_count == 1
    r.maybe_remove_all_loras.assert_called_once()

    with pytest.raises(AssertionError):
        r._capture_cudagraphs([desc], M.CUDAGraphMode.NONE)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------
def _load_runner(**over):
    comp = SimpleNamespace(mode=M.CompilationMode.NONE,
                           cudagraph_mode=M.CUDAGraphMode.NONE)
    attrs = dict(
        model_config=SimpleNamespace(model="m", enforce_eager=False,
                                     multimodal_config=None),
        parallel_config=SimpleNamespace(enable_eplb=False,
                                        use_ubatching=False),
        device=torch.device("cpu"),
        load_config=SimpleNamespace(load_format="auto",
                                    use_tqdm_on_load=False),
        lora_config=None,
        compilation_config=comp,
        vllm_config=SimpleNamespace(compilation_config=comp,
                                    speculative_config=None),
        jit_warmup_registry=SimpleNamespace(activate=lambda: nullcontext()),
        graph_runtime=None,
    )
    attrs.update(over)
    r = runner(**attrs)
    r._setup_eagle3_aux_hidden_state_outputs = mock.Mock()
    r.get_model = lambda: getattr(r, "model", None)
    return r


@mock.patch.object(M, "get_offloader",
                   lambda: SimpleNamespace(post_init=mock.Mock()))
@mock.patch.object(M, "supports_multimodal_pruning", lambda m: False)
@mock.patch.object(M, "register_io_module_hooks", mock.Mock())
@mock.patch.object(M, "_prepare_comm_buffer", mock.Mock())
@mock.patch.object(M, "warmup_cached_ops", mock.Mock())
@mock.patch.object(M, "get_mixture_of_experts_model", lambda m: None)
@mock.patch.object(M, "get_model_loader")
@mock.patch.object(M, "DeviceMemoryProfiler", _FakeDMP)
@mock.patch.object(M, "init_io_dump_from_env", mock.Mock())
@mock.patch.object(M, "is_breakable_cudagraph_enabled", lambda: False)
@mock.patch.object(M, "current_platform", SimpleNamespace(device_type="cuda"))
@mock.patch.object(M, "logger", mock.Mock())
def test_load_model_main_and_dummy(get_loader):
    model = mock.Mock()
    get_loader.return_value.load_model.return_value = model
    r = _load_runner()
    r.load_model()
    assert r.model is model
    assert r.model_memory_usage == 123

    M._prepare_comm_buffer.reset_mock()
    r2 = _load_runner()
    r2.load_model(load_dummy_weights=True)
    assert r2.load_config.load_format == "dummy"
    M._prepare_comm_buffer.assert_not_called()


@mock.patch.object(M, "DeviceMemoryProfiler", _OOMDMP)
@mock.patch.object(M, "init_io_dump_from_env", mock.Mock())
@mock.patch.object(M, "get_model_loader")
@mock.patch.object(M, "logger", mock.Mock())
def test_load_model_oom(get_loader):
    get_loader.return_value.load_model.return_value = mock.Mock()
    r = _load_runner()
    with pytest.raises(torch.OutOfMemoryError):
        r.load_model()


@mock.patch.object(M, "get_offloader",
                   lambda: SimpleNamespace(post_init=mock.Mock()))
@mock.patch.object(M, "supports_multimodal_pruning", lambda m: False)
@mock.patch.object(M, "register_io_module_hooks", mock.Mock())
@mock.patch.object(M, "_prepare_comm_buffer", mock.Mock())
@mock.patch.object(M, "warmup_cached_ops", mock.Mock())
@mock.patch.object(M, "get_mixture_of_experts_model", lambda m: None)
@mock.patch.object(M, "get_model_loader")
@mock.patch.object(M, "DeviceMemoryProfiler", _FakeDMP)
@mock.patch.object(M, "init_io_dump_from_env", mock.Mock())
@mock.patch.object(M, "current_platform", SimpleNamespace(device_type="cuda"))
@mock.patch.object(M, "logger", mock.Mock())
@mock.patch("vllm.env_override._apply_constrain_to_fx_strides_patch",
            mock.Mock())
def test_load_model_stock_compile(get_loader):
    model = mock.Mock()
    get_loader.return_value.load_model.return_value = model
    comp = SimpleNamespace(
        mode=M.CompilationMode.STOCK_TORCH_COMPILE,
        cudagraph_mode=M.CUDAGraphMode.NONE,
        init_backend=mock.Mock(return_value="backend"),
    )
    r = _load_runner(compilation_config=comp,
                     vllm_config=SimpleNamespace(compilation_config=comp,
                                                 speculative_config=None))
    with mock.patch.object(M.compilation_counter,
                           "stock_torch_compile_count", 0):
        r.load_model()
    model.compile.assert_called_once()
    assert model.compile.call_args.kwargs["backend"] == "backend"


@mock.patch.object(M, "get_offloader",
                   lambda: SimpleNamespace(post_init=mock.Mock()))
@mock.patch.object(M, "supports_multimodal_pruning", lambda m: False)
@mock.patch.object(M, "register_io_module_hooks", mock.Mock())
@mock.patch.object(M, "_prepare_comm_buffer", mock.Mock())
@mock.patch.object(M, "warmup_cached_ops", mock.Mock())
@mock.patch.object(M, "get_mixture_of_experts_model", lambda m: None)
@mock.patch.object(M, "get_model_loader")
@mock.patch.object(M, "DeviceMemoryProfiler", _FakeDMP)
@mock.patch.object(M, "init_io_dump_from_env", mock.Mock())
@mock.patch.object(M, "is_breakable_cudagraph_enabled", lambda: False)
@mock.patch.object(M, "resolve_obj_by_qualname",
                   lambda q: mock.Mock(return_value="wrapped"))
@mock.patch.object(M, "current_platform", SimpleNamespace(
    device_type="cuda", get_static_graph_wrapper_cls=lambda: "x"))
@mock.patch.object(M, "logger", mock.Mock())
def test_load_model_full_graph_wrapper(get_loader):
    get_loader.return_value.load_model.return_value = mock.Mock()
    mode = SimpleNamespace(has_full_cudagraphs=lambda: True)
    comp = SimpleNamespace(mode=M.CompilationMode.NONE, cudagraph_mode=mode)
    r = _load_runner(compilation_config=comp,
                     vllm_config=SimpleNamespace(compilation_config=comp,
                                                 speculative_config=None))
    r.load_model()
    assert r.model == "wrapped"


# ---------------------------------------------------------------------------
# sample_tokens
# ---------------------------------------------------------------------------
def _sample_runner(**over):
    attrs = dict(
        use_async_scheduling=False,
        broadcast_pp_output=False,
        check_ep_fault=False,
        speculative_config=None,
        supports_mm_inputs=False,
        kv_connector_output=None,
        device=torch.device("cpu"),
        input_batch=SimpleNamespace(
            prev_sampled_token_ids=object(),
            sampling_metadata=None,
            vocab_size=100,
            req_ids=["a", "b"],
            set_async_sampled_token_ids=mock.Mock(),
        ),
    )
    attrs.update(over)
    r = runner(**attrs)
    r._bookkeeping_sync = mock.Mock(
        return_value=(0, None, None, [[1]], None, ["a"], {"a": 0}, []))
    r.eplb_step = mock.Mock()
    return r


def _state(spec_meta=None):
    return (
        SimpleNamespace(total_num_scheduled_tokens=4),
        torch.zeros(2, 10),
        None,
        spec_meta,
        torch.zeros(4, 3),
        torch.zeros(2, 3),
        None,
        None,
        None,
        None,
    )


@mock.patch.object(M, "ModelRunnerOutput")
def test_sample_tokens_no_state(mout):
    mout.with_kv_conn_output_only = mock.Mock(return_value="only")
    r = _sample_runner(execute_model_state=None, kv_connector_output="kv")
    assert r.sample_tokens(None) == "only"
    assert r.kv_connector_output is None


@mock.patch.object(M, "record_function_or_nullcontext",
                   lambda name: nullcontext())
@mock.patch.object(M, "advance_io_step", mock.Mock())
@mock.patch.object(M, "ModelRunnerOutput")
def test_sample_tokens_sync(mout):
    mout.return_value = "out"
    r = _sample_runner(execute_model_state=_state())
    r._sample = mock.Mock(return_value=SimpleNamespace(
        sampled_token_ids=torch.zeros(2, 1, dtype=torch.int64),
        logprobs_tensors=None))
    r._update_states_after_model_execute = mock.Mock()
    assert r.sample_tokens(None) == "out"
    r._sample.assert_called_once()
    r.eplb_step.assert_called_once()
    assert r.execute_model_state is None


@mock.patch.object(M, "record_function_or_nullcontext",
                   lambda name: nullcontext())
@mock.patch.object(M, "advance_io_step", mock.Mock())
@mock.patch.object(M, "ModelRunnerOutput")
@mock.patch.object(M, "AsyncGPUModelRunnerOutput")
def test_sample_tokens_async(arun, mout):
    mout.return_value = "out"
    async_obj = SimpleNamespace(
        sampled_token_ids_cpu=torch.zeros(2, 1),
        async_copy_ready_event=object(),
    )
    arun.return_value = async_obj
    r = _sample_runner(execute_model_state=_state(),
                       use_async_scheduling=True)
    r._sample = mock.Mock(return_value=SimpleNamespace(
        sampled_token_ids=torch.zeros(2, 1, dtype=torch.int64),
        logprobs_tensors=None))
    r._update_states_after_model_execute = mock.Mock()
    r._get_or_create_async_output_copy_stream = mock.Mock(return_value="s")
    with mock.patch.object(M, "get_pp_group",
                           lambda: SimpleNamespace(world_size=1,
                                                   is_last_rank=True)):
        got = r.sample_tokens(None)
    assert got is async_obj
    r.input_batch.set_async_sampled_token_ids.assert_called_once()


@mock.patch.object(M, "record_function_or_nullcontext",
                   lambda name: nullcontext())
@mock.patch.object(M, "advance_io_step", mock.Mock())
@mock.patch.object(M, "ModelRunnerOutput")
def test_sample_tokens_draft(mout):
    class _Drafter:
        pass

    mout.return_value = "out"
    spec = SimpleNamespace(
        use_eagle=lambda: True,
        uses_draft_model=lambda: False,
        uses_extract_hidden_states=lambda: False,
        disable_padded_drafter_batch=False,
    )
    r = _sample_runner(execute_model_state=_state(spec_meta=object()),
                       speculative_config=spec, drafter=_Drafter())
    r._sample = mock.Mock(return_value=SimpleNamespace(
        sampled_token_ids=torch.zeros(2, 1, dtype=torch.int64),
        logprobs_tensors=None))
    r._update_states_after_model_execute = mock.Mock()
    r._input_fits_in_drafter = mock.Mock(return_value=True)
    r.propose_draft_token_ids = mock.Mock(return_value=[[1], [2]])
    r._copy_draft_token_ids_to_cpu = mock.Mock()
    r.finalize_kv_connector = mock.Mock()
    with mock.patch.object(M, "EagleProposer", _Drafter):
        assert r.sample_tokens(None) == "out"
    r.propose_draft_token_ids.assert_called_once()
    r._copy_draft_token_ids_to_cpu.assert_called_once()
    r.finalize_kv_connector.assert_called_once()


# ---------------------------------------------------------------------------
# gc / cleanup / shutdown
# ---------------------------------------------------------------------------
def test_freeze_gc_both_paths():
    with mock.patch.object(M.envs, "VLLM_ENABLE_CUDAGRAPH_GC", True):
        with M.ModelRunnerFL._freeze_gc():
            pass
    with mock.patch.object(M.envs, "VLLM_ENABLE_CUDAGRAPH_GC", False):
        with M.ModelRunnerFL._freeze_gc():
            pass


def test_cleanup_profiling_kv_cache():
    layer = SimpleNamespace(kv_cache=torch.zeros(1))
    r = runner(
        kv_caches=[torch.zeros(1)],
        cross_layers_kv_cache=torch.zeros(1),
        cross_layers_attn_backend=object(),
        attn_groups=[],
        kv_cache_config=object(),
        cache_config=SimpleNamespace(num_gpu_blocks=5),
        compilation_config=SimpleNamespace(
            static_forward_context={"l": layer}),
        device=torch.device("cpu"),
    )
    with mock.patch.object(M.torch, "accelerator", accel()):
        r._cleanup_profiling_kv_cache()
    assert r.kv_caches == []
    assert not hasattr(r, "kv_cache_config")
    assert r.cache_config.num_gpu_blocks is None
    assert layer.kv_cache.numel() == 0


@mock.patch.object(M, "current_platform", SimpleNamespace(
    is_rocm=lambda: False, is_xpu=lambda: False))
def test_shutdown():
    r = runner(
        graph_runtime=None,
        compilation_config=SimpleNamespace(static_forward_context={}),
        model=mock.Mock(),
    )
    r._cleanup_profiling_kv_cache = mock.Mock()
    with mock.patch("vllm.v1.worker.workspace.reset_workspace_manager",
                    mock.Mock()):
        r.shutdown()
    r._cleanup_profiling_kv_cache.assert_called_once()
    assert r.model is None
