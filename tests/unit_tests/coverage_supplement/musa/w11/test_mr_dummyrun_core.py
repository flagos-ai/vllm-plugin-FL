"""Branch coverage for ModelRunnerFL._dummy_run (model_runner.py 6111-6558)."""
import os
import sys
import types
from unittest.mock import MagicMock
import numpy as np
import pytest
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mr_dummyrun_common import CUDAGraphMode, LightModel, build_dummy_run_runner, install_module_stubs, mr, set_dummy_state
MR = mr.ModelRunnerFL

def _prepare(monkeypatch, **kwargs):
    runner = build_dummy_run_runner(**kwargs)
    install_module_stubs(monkeypatch, runner)
    return runner

class FakeEagle(mr.EagleProposer):
    pass

def _drafter():
    drafter = object.__new__(FakeEagle)
    drafter.dummy_run = MagicMock()
    return drafter

def _spec_config(**over):
    cfg = dict(use_eagle=lambda : True, uses_draft_model=lambda : False, uses_extract_hidden_states=lambda : False, enforce_eager=False, method='eagle')
    cfg.update(over)
    return types.SimpleNamespace(**cfg)

def test_dummy_run_mm_encoder_only_returns_empty(monkeypatch):
    runner = _prepare(monkeypatch)
    runner.model_config.multimodal_config = types.SimpleNamespace(mm_encoder_only=True)
    (out, logits) = MR._dummy_run(runner, 8)
    assert out.numel() == 0
    assert logits.numel() == 0
    assert runner.model.calls == []

def test_dummy_run_mode_mismatch_asserts(monkeypatch):
    runner = _prepare(monkeypatch, mode=CUDAGraphMode.FULL)
    with pytest.raises(AssertionError):
        MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)

def test_dummy_run_non_uniform_remainder_and_eplb(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=6, max_num_seqs=4)
    ids = torch.ones(6, dtype=torch.long)
    runner.input_ids = types.SimpleNamespace(gpu=ids)
    (hidden, logits) = MR._dummy_run(runner, 6)
    call = runner.determine_calls[0]
    assert call['num_reqs'] == 4
    assert call['max_num_scheduled_tokens'] == 6
    assert np.array_equal(call['num_scheduled_tokens_np'], np.array([1, 1, 1, 3], dtype=np.int32))
    assert hidden.shape == (6, 4)
    assert logits.shape == (4, 4)
    assert runner.model.calls[0]['input_ids'] is not None
    assert runner.model.calls[0]['inputs_embeds'] is None
    assert runner.forwards[0]['num_tokens'] == 6
    runner.eplb_step.assert_called_once_with(is_dummy=True, is_profile=False)
    assert runner.attention_calls == []

def test_dummy_run_skip_eplb(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    MR._dummy_run(runner, 4, skip_eplb=True)
    runner.eplb_step.assert_not_called()

def test_dummy_run_uniform_decode_padding_and_query_loc(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=5, max_num_seqs=4, uniform_decode_query_len=2)
    MR._dummy_run(runner, 5, uniform_decode=True, force_attention=True)
    call = runner.determine_calls[0]
    assert call['max_num_scheduled_tokens'] == 2
    assert call['force_uniform_decode'] is True
    assert np.array_equal(call['num_scheduled_tokens_np'], np.array([2, 2, 1], dtype=np.int32))
    assert torch.equal(runner.optimistic_seq_lens_cpu[:3], torch.tensor([2, 2, 2], dtype=torch.int32))
    assert torch.equal(runner.optimistic_seq_lens_cpu[3:], torch.zeros(1, dtype=torch.int32))
    assert np.array_equal(runner.query_start_loc.np[:5], [0, 2, 4, 5, 5])
    assert runner.query_start_loc.copy_count == 1
    assert mr.prepare_dcp_dummy_context_metadata.call_count == 2

def test_dummy_run_mixed_batch_with_dcp_context(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, dcp_len=2)
    MR._dummy_run(runner, 8, create_mixed_batch=True, force_attention=True)
    call = runner.determine_calls[0]
    assert np.array_equal(call['num_scheduled_tokens_np'], np.array([1, 1, 1, 5], dtype=np.int32))
    assert call['max_num_scheduled_tokens'] == 5
    assert torch.equal(runner.optimistic_seq_lens_cpu[:4], torch.tensor([3, 3, 3, 7], dtype=torch.int32))

def test_dummy_run_mixed_batch_without_dcp(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, dcp_len=0)
    MR._dummy_run(runner, 8, create_mixed_batch=True, force_attention=True)
    assert torch.equal(runner.optimistic_seq_lens_cpu[:4], torch.tensor([1, 1, 1, 6], dtype=torch.int32))

def test_dummy_run_uniform_decode_adds_dcp_len(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=6, max_num_seqs=4, uniform_decode_query_len=2, dcp_len=2)
    MR._dummy_run(runner, 6, uniform_decode=True, force_attention=True)
    assert torch.equal(runner.optimistic_seq_lens_cpu[:3], torch.tensor([4, 4, 4], dtype=torch.int32))

def test_dummy_run_profile_seq_lens_override(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=6, max_num_seqs=4)
    MR._dummy_run(runner, 6, force_attention=True, profile_seq_lens=3)
    assert torch.equal(runner.optimistic_seq_lens_cpu[:4], torch.tensor([3, 3, 3, 3], dtype=torch.int32))

def test_dummy_run_mixed_batch_asserts_not_uniform(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4)
    with pytest.raises(AssertionError):
        MR._dummy_run(runner, 8, create_mixed_batch=True, uniform_decode=True)

def test_dummy_run_mode_none_skips_attention_but_fills_slots(monkeypatch):
    slot = torch.zeros(4, dtype=torch.long)
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=4, slot_mappings=({'gid': slot}, {'layer': slot}))
    MR._dummy_run(runner, 4)
    assert runner.attention_calls == []
    mr.prepare_dcp_dummy_context_metadata.assert_not_called()
    assert torch.equal(slot, torch.full((4,), -1, dtype=torch.long))

def test_dummy_run_full_mode_builds_attention(monkeypatch):
    slot_group = {'gid': torch.zeros(8, dtype=torch.long)}
    slot_layer = {'layer': torch.zeros(8, dtype=torch.long)}
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, mode=CUDAGraphMode.FULL, batch_num_tokens=8, batch_num_reqs=4, slot_mappings=(slot_group, slot_layer))
    runner._attn_metadata = 'ATTMETA'
    MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    call = runner.attention_calls[0]
    assert call['num_tokens'] == 8
    assert call['num_tokens_padded'] == 8
    assert call['num_reqs'] == 4
    assert call['for_cudagraph_capture'] is True
    assert call['use_spec_decode'] is False
    assert runner.forwards[0]['attn_metadata'] == 'ATTMETA'
    assert runner.forwards[0]['cudagraph_runtime_mode'] == CUDAGraphMode.FULL
    assert torch.equal(slot_group['gid'], torch.full((8,), -1, dtype=torch.long))
    runner.input_batch.block_table.commit_block_table.assert_called_once_with(4)
    assert mr.prepare_dcp_dummy_context_metadata.call_count == 2

def test_dummy_run_ascend_attention_pads_query_start_loc(monkeypatch):
    calls = []

    def get_ascend_dummy_attention_state(create_mixed_batch, speculative_method=None, use_mla=False):
        calls.append((create_mixed_batch, speculative_method, use_mla))
        return 'ascend-state'
    fake_mod = types.ModuleType('vllm_fl.dispatch.backends.vendor.ascend.impl.attention')
    fake_mod.get_ascend_dummy_attention_state = get_ascend_dummy_attention_state
    monkeypatch.setitem(sys.modules, 'vllm_fl.dispatch.backends.vendor.ascend.impl.attention', fake_mod)
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, mode=CUDAGraphMode.FULL, batch_num_tokens=8, batch_num_reqs=4)
    runner.use_ascend_attention = True
    runner.model_config.use_mla = True
    runner.speculative_config = types.SimpleNamespace(method='eagle', use_eagle=lambda : False, uses_draft_model=lambda : False, uses_extract_hidden_states=lambda : False, enforce_eager=True)
    runner._pad_query_start_loc_for_fia = MagicMock(return_value=5)
    MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.FULL)
    assert calls == [(False, 'eagle', True)]
    runner._pad_query_start_loc_for_fia.assert_called_once_with(runner.query_start_loc, 8, 4, 4)
    assert runner.attention_calls[0]['num_reqs'] == 5

def test_dummy_run_multimodal_inputs_branch(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.supports_mm_inputs = True
    runner.model_config.is_encoder_decoder = False
    MR._dummy_run(runner, 4)
    assert runner.mm_calls == [4]
    assert runner.model_kwargs_calls == [2]
    assert runner.model.calls[0]['mm_arg'] == 2
    assert runner.model.calls[0]['inputs_embeds'] is not None

def test_dummy_run_encoder_decoder_uses_input_ids(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.supports_mm_inputs = True
    runner.model_config.is_encoder_decoder = True
    MR._dummy_run(runner, 4)
    assert runner.mm_calls == []
    assert runner.model.calls[0]['input_ids'] is not None

def test_dummy_run_prompt_embeds_branch(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.enable_prompt_embeds = True
    MR._dummy_run(runner, 4)
    call = runner.model.calls[0]
    assert call['input_ids'] is None
    assert call['inputs_embeds'].shape[0] == 4

def test_dummy_run_mrope_positions_branch(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.uses_mrope = True
    MR._dummy_run(runner, 4)
    assert runner.model.calls[0]['positions'].shape == (3, 4)

def test_dummy_run_aux_hidden_state_outputs(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.use_aux_hidden_state_outputs = True
    runner.model.aux_hidden_state = True
    (hidden, logits) = MR._dummy_run(runner, 4)
    assert hidden.shape == (4, 4)
    assert logits.shape == (2, 4)

def test_dummy_run_non_first_rank_reuses_intermediate_tensors(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    set_dummy_state(pp_first_rank=False)
    runner._pp_first_rank = False
    runner.intermediate_tensors = {'h': torch.ones(32, 2)}
    MR._dummy_run(runner, 4)
    assert runner.model.intermediate_shapes == []
    assert torch.equal(runner.model.calls[0]['intermediate_tensors']['h'], torch.ones(4, 2))

def test_dummy_run_ubatch_rewrites_padded_size(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_tokens=8, max_num_seqs=4, ubatch_slices=(None, [types.SimpleNamespace(num_tokens=4)]))
    runner._num_tokens_across_dp = torch.zeros(1, dtype=torch.int32)
    MR._dummy_run(runner, 8)
    ctx = runner.forwards[0]
    assert ctx['num_tokens'] == 4
    assert ctx['ubatch_slices'] is runner._ubatch_slices[1]
    assert int(runner._num_tokens_across_dp[0]) == 4

def test_dummy_run_randomizes_placeholder_input_ids(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    monkeypatch.setattr(mr.envs, 'VLLM_RANDOMIZE_DP_DUMMY_INPUTS', True)
    runner.vllm_config.parallel_config.data_parallel_size = 2
    MR._dummy_run(runner, 4, randomize_inputs=True)
    assert torch.equal(runner.input_ids.gpu[:4], torch.zeros(4, dtype=torch.long))

def test_dummy_run_spec_decode_captures_drafter_with_cudagraphs(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, mode=CUDAGraphMode.PIECEWISE)
    runner.speculative_config = _spec_config()
    runner.drafter = _drafter()
    MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, is_graph_capturing=True)
    runner.drafter.dummy_run.assert_called_once_with(8, use_cudagraphs=True, is_graph_capturing=True, slot_mappings=None)

def test_dummy_run_spec_decode_enforce_eager_disables_cudagraphs(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, mode=CUDAGraphMode.PIECEWISE)
    runner.speculative_config = _spec_config(enforce_eager=True)
    runner.drafter = _drafter()
    MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE)
    (_, kwargs) = runner.drafter.dummy_run.call_args
    assert kwargs['use_cudagraphs'] is False

def test_dummy_run_spec_decode_lora_specialization_disables_cudagraphs(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=8, max_num_seqs=4, mode=CUDAGraphMode.PIECEWISE)
    runner.speculative_config = _spec_config()
    runner.drafter = _drafter()
    runner.compilation_config.cudagraph_specialize_lora = True
    MR._dummy_run(runner, 8, cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE, num_active_loras=1)
    (_, kwargs) = runner.drafter.dummy_run.call_args
    assert kwargs['use_cudagraphs'] is False

def test_dummy_run_spec_decode_rejects_wrong_drafter(monkeypatch):
    runner = _prepare(monkeypatch, num_tokens=4, max_num_seqs=2)
    runner.speculative_config = _spec_config()
    runner.drafter = object()
    with pytest.raises(AssertionError):
        MR._dummy_run(runner, 4)