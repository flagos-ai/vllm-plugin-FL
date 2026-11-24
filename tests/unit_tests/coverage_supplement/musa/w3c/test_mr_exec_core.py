# SPDX-License-Identifier: Apache-2.0
"""CPU-only branch coverage for ModelRunnerFL execution helpers.

Targets the 3600-5600 line band of vllm_fl/worker/model_runner.py.  Every
test builds the runner with ``object.__new__`` and hand-set attributes; no
GPU, weights or network are required.
"""

import contextlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import vllm_fl.worker.model_runner as mr

ModelRunnerFL = mr.ModelRunnerFL


def _runner():
    return object.__new__(ModelRunnerFL)


# ---------------------------------------------------------------------------
# _is_uniform_decode
# ---------------------------------------------------------------------------


def test_is_uniform_decode_natural_true():
    assert ModelRunnerFL._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=2,
        num_tokens=6,
        num_reqs=3,
    )


def test_is_uniform_decode_natural_false_token_mismatch():
    assert not ModelRunnerFL._is_uniform_decode(
        max_num_scheduled_tokens=2,
        uniform_decode_query_len=2,
        num_tokens=5,
        num_reqs=3,
    )


@pytest.mark.parametrize("forced", [True, False])
def test_is_uniform_decode_forced_overrides(forced):
    # Contradictory token counts are ignored when force is provided.
    assert ModelRunnerFL._is_uniform_decode(9, 2, 1, 7, force_uniform_decode=forced) is forced


# ---------------------------------------------------------------------------
# _pad_query_start_loc_for_fia
# ---------------------------------------------------------------------------


def _qsl(size):
    calls = []

    def copy_to_gpu():
        calls.append(True)

    buf = SimpleNamespace(np=np.zeros(size, dtype=np.int64), copy_to_gpu=copy_to_gpu)
    return buf, calls


def test_pad_query_start_loc_uniform_decode():
    runner = _runner()
    runner.uniform_decode_query_len = 2
    runner.arange_np = np.arange(0, 20)
    qsl, calls = _qsl(8)
    qsl.np[:3] = [0, 2, 4]
    padded = runner._pad_query_start_loc_for_fia(qsl, 8, num_reqs=2, batch_desc_num_reqs=4)
    assert padded == 4
    assert qsl.np[3:5].tolist() == [6, 8]
    assert calls == [True]


def test_pad_query_start_loc_nonuniform_extends_when_short():
    runner = _runner()
    runner.uniform_decode_query_len = 1
    runner.arange_np = np.arange(0, 20)
    qsl, calls = _qsl(8)
    qsl.np[:3] = [0, 2, 2]
    padded = runner._pad_query_start_loc_for_fia(qsl, 5, num_reqs=2, batch_desc_num_reqs=2)
    assert padded == 3
    assert qsl.np[3] == 5
    assert calls == [True]


def test_pad_query_start_loc_nonuniform_no_extension_needed():
    runner = _runner()
    runner.uniform_decode_query_len = 1
    runner.arange_np = np.arange(0, 20)
    qsl, calls = _qsl(8)
    qsl.np[:3] = [0, 2, 5]
    padded = runner._pad_query_start_loc_for_fia(qsl, 5, num_reqs=2)
    assert padded == 2
    assert calls == [True]


# ---------------------------------------------------------------------------
# _prepare_mm_inputs
# ---------------------------------------------------------------------------


def test_prepare_mm_inputs_raw_tokens():
    runner = _runner()
    runner.model = SimpleNamespace(requires_raw_input_tokens=True)
    runner.input_ids = SimpleNamespace(gpu=torch.arange(6))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.arange(12).reshape(6, 2))
    input_ids, embeds = runner._prepare_mm_inputs(4)
    assert input_ids.tolist() == [0, 1, 2, 3]
    assert embeds.shape == (4, 2)


def test_prepare_mm_inputs_embeds_only():
    runner = _runner()
    runner.model = SimpleNamespace(requires_raw_input_tokens=False)
    runner.input_ids = SimpleNamespace(gpu=torch.arange(6))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.arange(12).reshape(6, 2))
    input_ids, embeds = runner._prepare_mm_inputs(3)
    assert input_ids is None
    assert embeds.shape == (3, 2)


# ---------------------------------------------------------------------------
# _is_all_reqs_chunked_prefill
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mask,expected", [([True, True], True), ([True, False], False), ([], True)]
)
def test_is_all_reqs_chunked_prefill(mask, expected):
    runner = _runner()
    runner.input_batch = SimpleNamespace(num_reqs=len(mask))
    runner.discard_request_mask = SimpleNamespace(np=np.array(mask, dtype=bool))
    assert runner._is_all_reqs_chunked_prefill() is expected


# ---------------------------------------------------------------------------
# _allow_microbatching
# ---------------------------------------------------------------------------


def _mb_runner(table, computed, query_lens, threshold=1):
    runner = _runner()
    runner.parallel_config = SimpleNamespace(use_ubatching=True)
    runner.reorder_batch_threshold = threshold
    runner.input_batch = SimpleNamespace(
        num_computed_tokens_cpu=np.array(computed, dtype=np.int32),
        block_table=SimpleNamespace(
            block_tables=[
                SimpleNamespace(
                    block_size=2, get_numpy_array=lambda t=table: np.array(t)
                )
            ]
        ),
    )
    return runner


def test_allow_microbatching_disabled():
    runner = _runner()
    runner.parallel_config = SimpleNamespace(use_ubatching=False)
    assert runner._allow_microbatching(4, np.array([2, 2, 2, 2])) is True


def test_allow_microbatching_single_req():
    runner = _runner()
    runner.parallel_config = SimpleNamespace(use_ubatching=True)
    assert runner._allow_microbatching(1, np.array([2])) is True


def test_allow_microbatching_no_readers():
    runner = _mb_runner(
        [[1, 2, 3], [4, 5, 6]], computed=[0, 0], query_lens=[1, 1]
    )
    assert runner._allow_microbatching(2, np.array([1, 1])) is True


def test_allow_microbatching_reader_shares_writer_block_vetoes():
    table = [[10, 11, 12, 13, 14, 15], [10, 21, 22, 23, 24, 25]]
    runner = _mb_runner(table, computed=[2, 0], query_lens=[2, 2])
    assert runner._allow_microbatching(2, np.array([2, 2], dtype=np.int32)) is False


def test_allow_microbatching_disjoint_blocks_allowed():
    table = [[10, 11, 12, 13, 14, 15], [20, 21, 22, 23, 24, 25]]
    runner = _mb_runner(table, computed=[2, 0], query_lens=[2, 2])
    assert runner._allow_microbatching(2, np.array([2, 2], dtype=np.int32)) is True


# ---------------------------------------------------------------------------
# synchronize_input_prep
# ---------------------------------------------------------------------------


def test_synchronize_input_prep_without_event():
    runner = _runner()
    runner.prepare_inputs_event = None
    seen = False
    with runner.synchronize_input_prep():
        seen = True
    assert seen


def test_synchronize_input_prep_tracks_event():
    runner = _runner()
    log = []
    runner.prepare_inputs_event = SimpleNamespace(
        synchronize=lambda: log.append("sync"), record=lambda: log.append("record")
    )
    with runner.synchronize_input_prep():
        log.append("body")
    assert log == ["sync", "body", "record"]


def test_synchronize_input_prep_records_on_error():
    runner = _runner()
    log = []
    runner.prepare_inputs_event = SimpleNamespace(
        synchronize=lambda: log.append("sync"), record=lambda: log.append("record")
    )
    with pytest.raises(ValueError):
        with runner.synchronize_input_prep():
            raise ValueError("boom")
    assert log == ["sync", "record"]


# ---------------------------------------------------------------------------
# _model_forward
# ---------------------------------------------------------------------------


def test_model_forward_forwards_kwargs():
    runner = _runner()
    captured = {}

    def model(**kwargs):
        captured.update(kwargs)
        return "OUT"

    runner.model = model
    out = runner._model_forward(
        input_ids=torch.zeros(2, dtype=torch.long),
        positions=torch.zeros(2, dtype=torch.long),
        extra="x",
    )
    assert out == "OUT"
    assert captured["extra"] == "x"
    assert captured["inputs_embeds"] is None


# ---------------------------------------------------------------------------
# _get_slot_mappings
# ---------------------------------------------------------------------------


def test_get_slot_mappings_without_cache_config():
    runner = _runner()
    assert runner._get_slot_mappings(4, 2, 4) == (None, None)


def test_get_slot_mappings_encoder_only_and_regular(monkeypatch):
    class _EncSpec:
        pass

    monkeypatch.setattr(mr, "EncoderOnlyAttentionSpec", _EncSpec)
    runner = _runner()
    runner.device = "cpu"
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=_EncSpec(), layer_names=["enc"]),
            SimpleNamespace(kv_cache_spec=object(), layer_names=["l1", "l2"]),
        ]
    )
    gpu_slot = torch.arange(6)
    runner.input_batch = SimpleNamespace(
        block_table={1: SimpleNamespace(slot_mapping=SimpleNamespace(gpu=gpu_slot))}
    )
    by_gid, by_layer = runner._get_slot_mappings(
        num_tokens_padded=4, num_reqs_padded=2, num_tokens_unpadded=3
    )
    assert set(by_gid) == {0, 1}
    # Unused slots are filled with -1 for both encoder-only and regular specs.
    assert by_gid[0].tolist() == [0, 0, 0, -1]
    assert by_gid[1].tolist() == [0, 1, 2, -1]
    assert set(by_layer) == {"enc", "l1", "l2"}
    assert by_layer["l1"] is by_gid[1]


def test_get_slot_mappings_ubatch_slices(monkeypatch):
    class _EncSpec:
        pass

    monkeypatch.setattr(mr, "EncoderOnlyAttentionSpec", _EncSpec)
    runner = _runner()
    runner.device = "cpu"
    runner.kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(kv_cache_spec=object(), layer_names=["l1"])]
    )
    runner.input_batch = SimpleNamespace(
        block_table={0: SimpleNamespace(slot_mapping=SimpleNamespace(gpu=torch.arange(6)))}
    )
    ubatches = [
        SimpleNamespace(token_slice=slice(0, 2)),
        SimpleNamespace(token_slice=slice(2, 4)),
    ]
    _, by_layer = runner._get_slot_mappings(4, 2, 4, ubatch_slices=ubatches)
    assert isinstance(by_layer, list) and len(by_layer) == 2
    assert by_layer[0]["l1"].tolist() == [0, 1]
    assert by_layer[1]["l1"].tolist() == [2, 3]


# ---------------------------------------------------------------------------
# _determine_batch_execution_and_padding
# ---------------------------------------------------------------------------


class _Dispatcher:
    def __init__(self, tokens=None, num_reqs=None):
        self.calls = []
        self.tokens = tokens
        self.num_reqs = num_reqs

    def dispatch(self, **kwargs):
        self.calls.append(kwargs)
        tokens = kwargs["num_tokens"] if self.tokens is None else self.tokens
        return (
            mr.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=tokens, num_reqs=self.num_reqs),
        )


def _batch_runner(dispatch, metrics=False, dp=1, dp_rank=0):
    runner = _runner()
    runner.uniform_decode_query_len = 1
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.input_batch = SimpleNamespace(lora_id_to_lora_request={})
    runner.parallel_config = SimpleNamespace(
        data_parallel_size=dp, data_parallel_rank=dp_rank
    )
    runner.vllm_config = SimpleNamespace(
        parallel_config=runner.parallel_config,
        observability_config=SimpleNamespace(cudagraph_metrics=metrics),
    )
    runner.cudagraph_dispatcher = dispatch
    return runner


def test_determine_batch_uniform_decode_and_metrics():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch, metrics=True)
    mode, desc, ubatch, dp_tokens, stats = runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=4,
        num_scheduled_tokens_np=np.array([1, 1, 1, 1]),
        max_num_scheduled_tokens=1,
        use_cascade_attn=False,
    )
    assert mode == mr.CUDAGraphMode.NONE
    assert desc.num_tokens == 4
    assert ubatch is False and dp_tokens is None
    assert stats.num_paddings == 0
    assert dispatch.calls[0]["uniform_decode"] is True
    assert dispatch.calls[0]["invalid_modes"] is None


def test_determine_batch_force_eager_and_disable_full():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch)
    runner._determine_batch_execution_and_padding(
        num_tokens=4,
        num_reqs=2,
        num_scheduled_tokens_np=np.array([2, 2]),
        max_num_scheduled_tokens=2,
        use_cascade_attn=True,
        force_eager=True,
    )
    call = dispatch.calls[0]
    assert call["valid_modes"] == {mr.CUDAGraphMode.NONE}
    assert call["invalid_modes"] == {mr.CUDAGraphMode.FULL}
    assert call["uniform_decode"] is False


def test_determine_batch_has_lora_overrides():
    dispatch = _Dispatcher()
    runner = _batch_runner(dispatch)
    runner.input_batch.lora_id_to_lora_request = {"a": 1, "b": 2}
    runner._determine_batch_execution_and_padding(
        4, 2, np.array([2, 2]), 2, False
    )
    assert dispatch.calls[0]["has_lora"] is True
    assert dispatch.calls[0]["num_active_loras"] == 2


def test_determine_batch_dp_coordination(monkeypatch):
    dispatch = _Dispatcher(num_reqs=None)
    runner = _batch_runner(dispatch, dp=2, dp_rank=1)
    monkeypatch.setattr(
        mr,
        "coordinate_batch_across_dp",
        lambda **kwargs: (True, torch.tensor([6, 8]), mr.CUDAGraphMode.NONE.value),
        raising=False,
    )
    # The re-dispatch must agree on the DP-synced token count.
    def dispatch_with_dp(**kwargs):
        dispatch.calls.append(kwargs)
        return (
            mr.CUDAGraphMode.NONE,
            SimpleNamespace(num_tokens=kwargs["num_tokens"], num_reqs=None),
        )

    runner.cudagraph_dispatcher = SimpleNamespace(dispatch=dispatch_with_dp)
    mode, desc, ubatch, dp_tokens, _ = runner._determine_batch_execution_and_padding(
        6, 3, np.array([2, 2, 2]), 2, False
    )
    assert ubatch is True
    assert desc.num_tokens == 8
    assert dp_tokens.tolist() == [6, 8]
    assert len(dispatch.calls) == 2
    assert dispatch.calls[1]["valid_modes"] == {mr.CUDAGraphMode.NONE}


# ---------------------------------------------------------------------------
# _preprocess
# ---------------------------------------------------------------------------


def _pre_runner():
    runner = _runner()
    runner.model_config = SimpleNamespace(is_encoder_decoder=False)
    runner.speculative_config = None
    runner.supports_mm_inputs = False
    runner.enable_prompt_embeds = False
    runner.uses_mrope = False
    runner.maybe_get_ec_connector_output = lambda *a, **k: contextlib.nullcontext()
    runner._init_model_kwargs = lambda: {"init": 1}
    runner._extract_mm_kwargs = lambda _so: {"mm": 2}
    runner._execute_mm_encoder = lambda _so: None
    return runner


def _sched(total=4, encoder_inputs=None):
    return SimpleNamespace(
        total_num_scheduled_tokens=total,
        scheduled_encoder_inputs=encoder_inputs or {},
    )


def test_preprocess_text_only_zeroes_padding(monkeypatch):
    runner = _pre_runner()
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(6))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(6, 2))
    runner.positions = torch.full((6,), 7, dtype=torch.long)
    input_ids, embeds, positions, inter, kwargs, ec = runner._preprocess(_sched(4), 6)
    assert input_ids.tolist() == [0, 1, 2, 3, 4, 5]
    assert embeds is None
    assert positions[4:6].tolist() == [0, 0]
    assert inter is None and kwargs == {"init": 1} and ec is None


def test_preprocess_prompt_embeds_fills_token_positions(monkeypatch):
    runner = _pre_runner()
    runner.enable_prompt_embeds = True
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([1, 2, 0, 0]))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(4, 2))
    runner.is_token_ids = SimpleNamespace(np=np.array([True, True, False, False]))
    runner.positions = torch.zeros(4, dtype=torch.long)
    runner.device = "cpu"

    def embed_input_ids(input_ids=None):
        return input_ids.to(torch.float32).unsqueeze(-1).expand(-1, 2) + 100

    runner.model = SimpleNamespace(embed_input_ids=embed_input_ids)
    # async_tensor_h2d is a module level helper; keep it local to CPU.
    monkeypatch.setattr(mr, "async_tensor_h2d", lambda arr, device=None: torch.as_tensor(arr), raising=False)
    input_ids, embeds, _, _, _, _ = runner._preprocess(_sched(4), 4)
    assert input_ids is None
    assert embeds[0].tolist() == [101.0, 101.0]
    assert embeds[2].tolist() == [0.0, 0.0]


def test_preprocess_non_first_rank_syncs_intermediate(monkeypatch):
    runner = _pre_runner()
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=False))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(3))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(3, 2))
    runner.positions = torch.zeros(3, dtype=torch.long)
    called = {}
    runner.sync_and_gather_intermediate_tensors = (
        lambda n, t, s: called.update(num=n, sync=s) or "SYNCED"
    )
    out = runner._preprocess(_sched(3), 3, intermediate_tensors=object())
    assert out[3] == "SYNCED"
    assert called == {"num": 3, "sync": True}


def test_preprocess_multimodal_embed_and_kwargs(monkeypatch):
    runner = _pre_runner()
    runner.supports_mm_inputs = True
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.encoder_cache = {}
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([1, 2, 3, 4]))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(4, 2))
    runner.positions = torch.zeros(4, dtype=torch.long)

    seen = {}

    def embed(ids, multimodal_embeddings=None, is_multimodal=None):
        seen["mm"] = multimodal_embeddings
        return ids.to(torch.float32).unsqueeze(-1).expand(-1, 2)

    runner.model = SimpleNamespace(
        requires_raw_input_tokens=False, embed_input_ids=embed
    )
    runner._gather_mm_embeddings = lambda _so: ("M", "I")
    input_ids, embeds, _, _, kwargs, _ = runner._preprocess(_sched(4), 4)
    assert input_ids is None
    assert embeds.shape == (4, 2)
    assert seen["mm"] == "M"
    assert kwargs == {"init": 1, "mm": 2}


def test_preprocess_encoder_decoder_outputs(monkeypatch):
    runner = _pre_runner()
    runner.model_config = SimpleNamespace(is_encoder_decoder=True)
    monkeypatch.setattr(mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    runner.input_ids = SimpleNamespace(gpu=torch.arange(2))
    runner.inputs_embeds = SimpleNamespace(gpu=torch.zeros(2, 2))
    runner.positions = torch.zeros(2, dtype=torch.long)
    runner._execute_mm_encoder = lambda _so: "ENCODER_OUT"
    _, _, _, _, kwargs, _ = runner._preprocess(
        _sched(2, encoder_inputs={"r": 1}), 2
    )
    assert kwargs["encoder_outputs"] == "ENCODER_OUT"


# ---------------------------------------------------------------------------
# execute_model guards / empty work path
# ---------------------------------------------------------------------------


def test_execute_model_rejects_stale_state():
    runner = _runner()
    runner.execute_model_state = object()
    with pytest.raises(RuntimeError, match="sample_tokens"):
        runner.execute_model(SimpleNamespace())


def _empty_exec_runner():
    runner = _runner()
    runner.execute_model_state = None
    runner.speculative_config = None
    runner.vllm_config = SimpleNamespace()
    runner.encoder_cache = {}
    runner._update_states = lambda _so: None
    runner.synchronize_input_prep = lambda: contextlib.nullcontext()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend="mp", data_parallel_size=1
    )
    runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
    return runner


def test_execute_model_empty_without_kv_group(monkeypatch):
    runner = _empty_exec_runner()
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda _n: contextlib.nullcontext())
    out = runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT


def test_execute_model_empty_with_kv_group(monkeypatch):
    runner = _empty_exec_runner()
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda _n: contextlib.nullcontext())
    handled = []
    monkeypatch.setattr(
        mr,
        "get_kv_transfer_group",
        lambda: SimpleNamespace(handle_preemptions=lambda m: handled.append(m)),
        raising=False,
    )
    runner.kv_connector_no_forward = lambda so, cfg: ("KVEMPTY", so)
    metadata = object()
    out = runner.execute_model(
        SimpleNamespace(total_num_scheduled_tokens=0, kv_connector_metadata=metadata)
    )
    assert out[0] == "KVEMPTY"
    assert handled == [metadata]


def test_execute_model_empty_external_launcher_dummy_run(monkeypatch):
    runner = _empty_exec_runner()
    runner.parallel_config = SimpleNamespace(
        distributed_executor_backend="external_launcher", data_parallel_size=2
    )
    dummy = []
    runner._dummy_run = lambda n: dummy.append(n)
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)
    monkeypatch.setattr(mr, "has_ec_transfer", lambda: False)
    monkeypatch.setattr(mr, "record_function_or_nullcontext", lambda _n: contextlib.nullcontext())
    out = runner.execute_model(SimpleNamespace(total_num_scheduled_tokens=0))
    assert out is mr.EMPTY_MODEL_RUNNER_OUTPUT
    assert dummy == [1]
