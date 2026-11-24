# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""G7-D (batch 2) coverage: attention metadata, multimodal encoder, pooling
and preprocessing entry points of ModelRunnerFL (814b line).

Targets (814b): _build_attention_metadata 2486-2828 (+ nested
_get_block_table 2529-2544 and _build_attn_group_metadata 2695-2766),
_execute_mm_encoder 3168-3384, _gather_mm_embeddings 3391-3500,
_pool 3620-3683, _pad_query_start_loc_for_fia 3686-3713,
_preprocess 3726-3868.

All collaborators (CommonAttentionMetadata, group_and_batch_mm_kwargs,
ModelRunnerOutput, get_pp_group, ...) are patched at the module boundary so the
tests stay on CPU and never touch a real attention backend or model weights.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import numpy as np
import torch

from vllm_fl.worker import model_runner as mr


class _Patch:
    def __init__(self, module, name, value):
        self.module, self.name, self.value = module, name, value

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        setattr(self.module, self.name, self.value)
        return self

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


def _stack(*pairs):
    stack = contextlib.ExitStack()
    for module, name, value in pairs:
        stack.enter_context(_Patch(module, name, value))
    return stack


# --------------------------------------------------------------------------
# _build_attention_metadata
# --------------------------------------------------------------------------
class _CM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _EncoderOnlyAttentionSpec:
    pass


class _UniformTypeKVCacheSpecs:
    pass


class _AttnGroup:
    def __init__(self, marker):
        self.layer_names = ["layer0"]
        self._marker = marker
        self.builder = SimpleNamespace(supports_update_block_table=False)

    def get_metadata_builder(self, ubid):
        return self.builder

    def build_metadata(self, common_attn_metadata, ubid, cascade_len, **extra):
        return ("meta", cascade_len, ubid)

    def build_metadata_for_cudagraph_capture(self, common_attn_metadata, ubid):
        return ("cg", ubid)


def _attn_runner():
    block_table = SimpleNamespace(
        get_device_tensor=lambda n: torch.zeros(n, 1, dtype=torch.int32)
    )
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[SimpleNamespace(kv_cache_spec=object())]
        ),
        max_model_len=100,
        optimistic_seq_lens_cpu=torch.tensor([5, 6, 0]),
        input_batch=SimpleNamespace(
            num_computed_tokens_cpu_tensor=torch.tensor([2, 3, 0]),
            num_prompt_tokens_cpu_tensor=torch.tensor([2, 5, 0]),
            block_table=[block_table],
        ),
        routed_experts_initialized=False,
        use_async_spec_decode=False,
        is_mm_prefix_lm=False,
        model_config=SimpleNamespace(rswa_window=None),
        cache_config=SimpleNamespace(
            use_replayssm=False, kv_sharing_fast_prefill=False
        ),
        query_start_loc=SimpleNamespace(
            gpu=torch.zeros(4, dtype=torch.int32),
            cpu=torch.zeros(4, dtype=torch.int32),
        ),
        seq_lens=torch.zeros(3, dtype=torch.int32),
        positions=torch.zeros(3, dtype=torch.int64),
        use_ascend_attention=False,
        dcp_world_size=1,
        attn_groups=[[_AttnGroup("g")]],
        speculative_config=None,
    )
    runner._get_encoder_seq_lens = (
        lambda num_sched, spec, num_reqs, for_cudagraph_capture=False: (None, None)
    )
    return runner


def _patch_attn_symbols():
    return (
        (mr, "CommonAttentionMetadata", _CM),
        (mr, "EncoderOnlyAttentionSpec", _EncoderOnlyAttentionSpec),
        (mr, "UniformTypeKVCacheSpecs", _UniformTypeKVCacheSpecs),
        (mr, "NULL_BLOCK_ID", -1),
    )


def test_build_attention_metadata_returns_empty_without_kv_groups():
    runner = SimpleNamespace(
        kv_cache_config=SimpleNamespace(kv_cache_groups=[])
    )
    attn_metadata, spec_cm = mr.ModelRunnerFL._build_attention_metadata(
        runner, num_tokens=3, num_reqs=2, max_query_len=2
    )
    assert attn_metadata == {}
    assert spec_cm is None


def test_build_attention_metadata_builds_group_metadata():
    runner = _attn_runner()
    with _stack(*_patch_attn_symbols()):
        attn_metadata, spec_cm = mr.ModelRunnerFL._build_attention_metadata(
            runner,
            num_tokens=3,
            num_reqs=2,
            max_query_len=2,
            slot_mappings={0: torch.zeros(3, dtype=torch.int32)},
            num_scheduled_tokens={"r0": 1},
        )
    assert attn_metadata == {"layer0": ("meta", 0, 0)}
    assert spec_cm is None


def test_build_attention_metadata_cudagraph_capture_path():
    runner = _attn_runner()
    with _stack(*_patch_attn_symbols()):
        attn_metadata, _ = mr.ModelRunnerFL._build_attention_metadata(
            runner,
            num_tokens=3,
            num_reqs=2,
            max_query_len=2,
            num_tokens_padded=3,
            num_reqs_padded=2,
            slot_mappings={0: torch.zeros(3, dtype=torch.int32)},
            for_cudagraph_capture=True,
        )
    assert attn_metadata == {"layer0": ("cg", 0)}


# --------------------------------------------------------------------------
# _execute_mm_encoder
# --------------------------------------------------------------------------
def _mm_sched(**extra):
    base = SimpleNamespace(
        ec_manager_metadata=None,
        free_encoder_mm_hashes=[],
        scheduled_encoder_inputs={"r0": [0]},
    )
    base.__dict__.update(extra)
    return base


def test_execute_mm_encoder_returns_empty_without_kwargs():
    calls = []
    runner = SimpleNamespace(
        _batch_mm_inputs_from_scheduler=lambda so: ([], [], [])
    )
    assert mr.ModelRunnerFL._execute_mm_encoder(runner, _mm_sched()) == []
    assert calls == []


def test_execute_mm_encoder_injects_prompt_embeds_only():
    calls = []
    pe_tensor = torch.ones(2, 4)
    mm_kwargs = [("prompt_embeds", {"embedding": SimpleNamespace(data=pe_tensor)})]
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        _batch_mm_inputs_from_scheduler=lambda so: (
            ["h1"],
            mm_kwargs,
            [("r0", "pos")],
        ),
        _cache_encoder_output=lambda h, o, ec, fr: calls.append((h, o)),
    )
    with _stack((mr, "async_tensor_h2d", lambda t, device: t)):
        result = mr.ModelRunnerFL._execute_mm_encoder(runner, _mm_sched())
    assert result == []
    assert calls[0][0] == "h1"
    assert torch.equal(calls[0][1], pe_tensor)


def test_execute_mm_encoder_runs_model_and_caches_outputs():
    out = torch.ones(2, 3)
    cached = []
    runner = SimpleNamespace(
        device=torch.device("cpu"),
        lora_config=None,
        observability_config=None,
        is_multimodal_pruning_enabled=False,
        requires_sequential_video_encoding=False,
        encoder_cudagraph_manager=None,
        model=SimpleNamespace(embed_multimodal=lambda **kw: [out]),
        _batch_mm_inputs_from_scheduler=lambda so: (
            ["h1"],
            [("image", {"pixel": 1})],
            [("r0", "pos")],
        ),
        _cache_encoder_output=lambda h, o, ec, fr: cached.append((h, o)),
    )
    runner.timed_encoder_operation = lambda *a, **k: contextlib.nullcontext()

    def _group(items, device, pin_memory):
        return [("image", 1, {"pixel": 1})]

    sanity = []
    with _stack(
        (mr, "group_and_batch_mm_kwargs", _group),
        (mr, "sanity_check_mm_encoder_outputs",
         lambda b, expected_num_items: sanity.append(expected_num_items)),
        (mr, "PIN_MEMORY", False),
    ):
        result = mr.ModelRunnerFL._execute_mm_encoder(runner, _mm_sched())
    assert result == [out]
    assert sanity == [1]
    assert cached == [("h1", out)]


# --------------------------------------------------------------------------
# _gather_mm_embeddings
# --------------------------------------------------------------------------
def test_gather_mm_embeddings_no_features_returns_empty_mask():
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0"]),
        requests={"r0": SimpleNamespace(num_computed_tokens=0, mm_features=[])},
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r0": 2}
    )
    with _stack(
        (mr, "get_mm_features_in_window", lambda feats, start, end: (0, 0)),
        (mr, "PIN_MEMORY", False),
    ):
        mm_embeds, mask = mr.ModelRunnerFL._gather_mm_embeddings(runner, sched)
    assert mm_embeds == []
    assert mask.tolist() == [False, False]


def test_gather_mm_embeddings_splices_encoder_output():
    pos_info = SimpleNamespace(
        offset=0,
        length=2,
        get_embeds_indices_in_range=lambda s, e: (s, e),
        is_embed=None,
    )
    feature = SimpleNamespace(
        mm_position=pos_info, identifier="h1", modality="image"
    )
    encoder_out = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0"]),
        requests={
            "r0": SimpleNamespace(num_computed_tokens=0, mm_features=[feature])
        },
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
        _get_encoder_output_from_cache=lambda h: encoder_out,
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r0": 2}
    )
    modalities = []
    with _stack(
        (mr, "get_mm_features_in_window", lambda feats, start, end: (0, 1)),
        (mr, "set_mm_embedding_modality", lambda item, mod: modalities.append(mod)),
        (mr, "PIN_MEMORY", False),
    ):
        mm_embeds, mask = mr.ModelRunnerFL._gather_mm_embeddings(runner, sched)
    assert len(mm_embeds) == 1
    assert torch.equal(mm_embeds[0], encoder_out)
    assert mask.tolist() == [True, True]
    assert modalities == ["image"]


def test_gather_mm_embeddings_raises_on_encoder_cache_miss():
    pos_info = SimpleNamespace(
        offset=0,
        length=2,
        get_embeds_indices_in_range=lambda s, e: (s, e),
        is_embed=None,
    )
    feature = SimpleNamespace(
        mm_position=pos_info, identifier="h1", modality="image"
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r0"]),
        requests={
            "r0": SimpleNamespace(num_computed_tokens=0, mm_features=[feature])
        },
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
        _get_encoder_output_from_cache=lambda h: None,
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=2, num_scheduled_tokens={"r0": 2}
    )
    with _stack(
        (mr, "get_mm_features_in_window", lambda feats, start, end: (0, 1)),
        (mr, "PIN_MEMORY", False),
    ):
        try:
            mr.ModelRunnerFL._gather_mm_embeddings(runner, sched)
        except RuntimeError as exc:
            assert "Encoder cache miss" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected RuntimeError")


# --------------------------------------------------------------------------
# _pool
# --------------------------------------------------------------------------
class _MRO:
    def __init__(self, req_ids, req_id_to_index, kv_connector_output):
        self.req_ids = req_ids
        self.req_id_to_index = req_id_to_index
        self.kv_connector_output = kv_connector_output


def _pool_runner(prompt_lens, raw_output):
    pooling_metadata = SimpleNamespace(
        prompt_lens=prompt_lens,
        pooling_params=["p0", "p1"],
        build_pooling_cursor=lambda *a, **k: None,
    )
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            num_reqs=2,
            pooling_params=["p0", "p1"],
            req_ids=["r0", "r1"],
            req_id_to_index={"r0": 0, "r1": 1},
            get_pooling_metadata=lambda: pooling_metadata,
        ),
        optimistic_seq_lens_cpu=torch.tensor([2, 3]),
        query_start_loc=SimpleNamespace(gpu=torch.zeros(3, dtype=torch.int32)),
        model=SimpleNamespace(pooler=lambda hidden_states, pooling_metadata: raw_output),
        late_interaction_runner=SimpleNamespace(
            postprocess_pooler_output=lambda **kw: raw_output
        ),
    )
    calls = []
    runner._sync_device = lambda: calls.append("sync")
    return runner, calls


def test_pool_no_finished_requests_returns_none_outputs():
    fake_platform = SimpleNamespace(is_cuda_alike=lambda: False)
    runner, calls = _pool_runner([5, 5], torch.ones(2, 4))
    with _stack(
        (mr, "ModelRunnerOutput", _MRO),
        (mr, "current_platform", fake_platform),
    ):
        out = mr.ModelRunnerFL._pool(
            runner,
            torch.ones(5, 4),
            5,
            np.array([2, 3]),
            None,
        )
    assert out.pooler_output == [None, None]
    assert calls == ["sync"]


def test_pool_finished_requests_copies_output_to_cpu():
    fake_platform = SimpleNamespace(is_cuda_alike=lambda: False)
    raw = torch.ones(2, 4)
    runner, calls = _pool_runner([2, 3], raw)
    copied = []
    with _stack(
        (mr, "ModelRunnerOutput", _MRO),
        (mr, "current_platform", fake_platform),
        (mr, "_copy_pooler_output_to_cpu",
         lambda raw_pooler_output, finished_mask: (
             copied.append(finished_mask) or ["c0", "c1"]
         )),
    ):
        out = mr.ModelRunnerFL._pool(
            runner,
            torch.ones(5, 4),
            5,
            np.array([2, 3]),
            None,
        )
    assert out.pooler_output == ["c0", "c1"]
    assert copied == [[True, True]]
    assert calls == ["sync"]


# --------------------------------------------------------------------------
# _pad_query_start_loc_for_fia
# --------------------------------------------------------------------------
def test_pad_query_start_loc_uniform_decode_pads_tail():
    qsl = SimpleNamespace(
        np=np.zeros(6, dtype=np.int32),
        copy_to_gpu=lambda: None,
    )
    runner = SimpleNamespace(
        uniform_decode_query_len=2,
        arange_np=np.arange(10),
    )
    result = mr.ModelRunnerFL._pad_query_start_loc_for_fia(
        runner, qsl, num_tokens_padded=6, num_reqs=2, batch_desc_num_reqs=3
    )
    assert result == 3
    assert qsl.np[3] == 2


def test_pad_query_start_loc_non_uniform_appends_token_boundary():
    qsl = SimpleNamespace(
        np=np.zeros(6, dtype=np.int32),
        copy_to_gpu=lambda: None,
    )
    runner = SimpleNamespace(
        uniform_decode_query_len=2,
        arange_np=np.arange(10),
    )
    result = mr.ModelRunnerFL._pad_query_start_loc_for_fia(
        runner, qsl, num_tokens_padded=5, num_reqs=2, batch_desc_num_reqs=2
    )
    assert result == 3
    assert qsl.np[3] == 5


# --------------------------------------------------------------------------
# _preprocess
# --------------------------------------------------------------------------
def _preprocess_runner(*, is_encoder_decoder):
    runner = SimpleNamespace(
        speculative_config=None,
        model_config=SimpleNamespace(is_encoder_decoder=is_encoder_decoder),
        supports_mm_inputs=False,
        enable_prompt_embeds=False,
        uses_mrope=False,
        input_ids=SimpleNamespace(gpu=torch.zeros(6, dtype=torch.int32)),
        positions=torch.arange(6, dtype=torch.int64),
    )
    runner._init_model_kwargs = lambda: {"k": 1}
    return runner


def test_preprocess_text_only_returns_token_inputs():
    runner = _preprocess_runner(is_encoder_decoder=False)
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_encoder_inputs={}
    )
    with _stack(
        (mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    ):
        (input_ids, inputs_embeds, positions, intermediate,
         model_kwargs, ec_out) = mr.ModelRunnerFL._preprocess(
            runner, sched, num_input_tokens=6
        )
    assert input_ids is not None
    assert inputs_embeds is None
    assert model_kwargs == {"k": 1}
    assert intermediate is None
    assert ec_out is None
    assert positions.tolist() == [0, 1, 2, 3, 0, 0]


def test_preprocess_encoder_decoder_runs_encoder():
    runner = _preprocess_runner(is_encoder_decoder=True)
    runner._execute_mm_encoder = lambda so: ["enc"]
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_encoder_inputs={"r0": [0]}
    )
    with _stack(
        (mr, "get_pp_group", lambda: SimpleNamespace(is_first_rank=True))
    ):
        result = mr.ModelRunnerFL._preprocess(
            runner, sched, num_input_tokens=4
        )
    assert result[4] == {"k": 1, "encoder_outputs": ["enc"]}
