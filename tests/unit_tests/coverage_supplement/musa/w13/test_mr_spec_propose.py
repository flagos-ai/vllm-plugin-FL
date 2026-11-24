"""CPU-only coverage for ModelRunnerFL draft-token helpers and proposers.

Covers ``_copy_draft_token_ids_to_cpu``, ``_get_draft_token_ids_cpu``,
``take_draft_token_ids``, the valid-sampled-count helpers, the PP broadcast /
receive helpers and the full ``propose_draft_token_ids`` dispatch table.
"""
import contextlib
import sys
import types
from types import SimpleNamespace
import numpy as np
import pytest
import torch
import vllm_fl.worker.model_runner as mr
ModelRunnerFL = mr.ModelRunnerFL

def _fake_platform(rec=None):

    def _current_stream():
        return 'DEFAULT'

    def _stream(_s):
        return contextlib.nullcontext()
    return SimpleNamespace(torch_device_fn=SimpleNamespace(current_stream=_current_stream, stream=_stream))

def test_take_draft_token_ids_none_when_no_spec_tokens():
    r = object.__new__(ModelRunnerFL)
    r.num_spec_tokens = 0
    r._draft_token_req_ids = ['r0']
    assert r.take_draft_token_ids() is None

def test_take_draft_token_ids_none_when_req_ids_missing():
    r = object.__new__(ModelRunnerFL)
    r.num_spec_tokens = 2
    r._draft_token_req_ids = None
    assert r.take_draft_token_ids() is None

def test_take_draft_token_ids_returns_draft_ids(monkeypatch):

    class _DraftIds:

        def __init__(self, req_ids, draft_token_ids):
            self.req_ids = req_ids
            self.draft_token_ids = draft_token_ids
    monkeypatch.setattr(mr, 'DraftTokenIds', _DraftIds)
    r = object.__new__(ModelRunnerFL)
    r.num_spec_tokens = 2
    r._draft_token_req_ids = ['r0']
    r._get_draft_token_ids_cpu = lambda : ([[7, 8]], ['r0'])
    out = r.take_draft_token_ids()
    assert out.req_ids == ['r0']
    assert out.draft_token_ids == [[7, 8]]

def test_copy_draft_token_ids_non_async_copies(monkeypatch):
    rec = []
    monkeypatch.setattr(mr, 'current_platform', _fake_platform(rec))
    r = object.__new__(ModelRunnerFL)
    r.use_async_scheduling = False
    r._draft_token_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    r.input_batch = SimpleNamespace(req_ids=['r0', 'r1'], sampling_metadata=SimpleNamespace(output_token_ids=None))
    r.draft_token_ids_event = SimpleNamespace(record=lambda : rec.append('record'))
    r.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda s: rec.append(('wait', s)))
    r.draft_token_ids_cpu = torch.zeros(2, 2, dtype=torch.int32)
    r._copy_draft_token_ids_to_cpu(SimpleNamespace(has_structured_output_requests=False))
    assert r.prev_num_spec_tokens == 2
    assert r._draft_token_req_ids == ['r0', 'r1']
    assert r.draft_token_ids_cpu.tolist() == [[1, 2], [3, 4]]
    assert ('wait', 'DEFAULT') in rec
    assert 'record' in rec

def test_copy_draft_token_ids_async_structured_forces_copy(monkeypatch):
    monkeypatch.setattr(mr, 'current_platform', _fake_platform())
    r = object.__new__(ModelRunnerFL)
    r.use_async_scheduling = True
    r._draft_token_ids = torch.tensor([[9, 9]], dtype=torch.int32)
    r.input_batch = SimpleNamespace(req_ids=['r0'], sampling_metadata=SimpleNamespace(output_token_ids=None))
    r.draft_token_ids_event = SimpleNamespace(record=lambda : None)
    r.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda s: None)
    r.draft_token_ids_cpu = torch.zeros(1, 2, dtype=torch.int32)
    r._copy_draft_token_ids_to_cpu(SimpleNamespace(has_structured_output_requests=True))
    assert r._draft_token_req_ids == ['r0']
    assert r.draft_token_ids_cpu.tolist() == [[9, 9]]

def test_copy_draft_token_ids_async_output_token_ids_forces_copy(monkeypatch):
    monkeypatch.setattr(mr, 'current_platform', _fake_platform())
    r = object.__new__(ModelRunnerFL)
    r.use_async_scheduling = True
    r._draft_token_ids = torch.tensor([[4]], dtype=torch.int32)
    r.input_batch = SimpleNamespace(req_ids=['r0'], sampling_metadata=SimpleNamespace(output_token_ids=[1, 2]))
    r.draft_token_ids_event = SimpleNamespace(record=lambda : None)
    r.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda s: None)
    r.draft_token_ids_cpu = torch.zeros(1, 1, dtype=torch.int32)
    r._copy_draft_token_ids_to_cpu(SimpleNamespace(has_structured_output_requests=False))
    assert r.draft_token_ids_cpu.tolist() == [[4]]

def test_copy_draft_token_ids_zeros_only(monkeypatch):
    rec = []
    monkeypatch.setattr(mr, 'current_platform', _fake_platform(rec))
    r = object.__new__(ModelRunnerFL)
    r.use_async_scheduling = False
    r._draft_token_ids = torch.ones(2, 2, dtype=torch.int32)
    r.input_batch = SimpleNamespace(req_ids=['r0', 'r1'], sampling_metadata=SimpleNamespace(output_token_ids=None))
    r.draft_token_ids_event = SimpleNamespace(record=lambda : rec.append('record'))
    r.draft_token_ids_copy_stream = SimpleNamespace(wait_stream=lambda s: rec.append('wait'))
    r.draft_token_ids_cpu = torch.ones(2, 2, dtype=torch.int32)
    r._copy_draft_token_ids_to_cpu(SimpleNamespace(has_structured_output_requests=False), zeros_only=True)
    assert r.draft_token_ids_cpu.tolist() == [[0, 0], [0, 0]]
    assert 'wait' not in rec
    assert 'record' in rec

def test_get_draft_token_ids_cpu_list_branch():
    r = object.__new__(ModelRunnerFL)
    r._draft_token_ids = [[1, 2]]
    r.input_batch = SimpleNamespace(req_ids=['r0'])
    assert r._get_draft_token_ids_cpu() == ([[1, 2]], ['r0'])

def test_get_draft_token_ids_cpu_no_req_ids():
    r = object.__new__(ModelRunnerFL)
    r._draft_token_ids = torch.zeros(1, 2, dtype=torch.int32)
    r._draft_token_req_ids = None
    assert r._get_draft_token_ids_cpu() == ([], [])

def test_get_draft_token_ids_cpu_syncs_and_slices():
    rec = []
    r = object.__new__(ModelRunnerFL)
    r._draft_token_ids = torch.zeros(2, 3, dtype=torch.int32)
    r._draft_token_req_ids = ['r0']
    r.draft_token_ids_event = SimpleNamespace(synchronize=lambda : rec.append('sync'))
    r.draft_token_ids_cpu = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int32)
    (ids, req_ids) = r._get_draft_token_ids_cpu()
    assert rec == ['sync']
    assert ids == [[1, 2, 3]]
    assert req_ids == ['r0']

def test_copy_valid_sampled_token_count_no_event_returns_early():
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = None
    r.input_batch = SimpleNamespace()
    r._copy_valid_sampled_token_count(torch.zeros(1), torch.zeros(1))
    assert not hasattr(r.input_batch, 'prev_sampled_token_ids')

def test_copy_valid_sampled_token_count_copies_and_sets_prev(monkeypatch):
    rec = []
    monkeypatch.setattr(mr, 'current_platform', _fake_platform(rec))
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = SimpleNamespace(record=lambda : rec.append('record'))
    r.valid_sampled_token_count_copy_stream = SimpleNamespace(wait_stream=lambda s: rec.append('wait'))
    r.valid_sampled_token_count_cpu = torch.zeros(3, dtype=torch.int32)
    r.use_async_spec_decode = False
    r.input_batch = SimpleNamespace()
    next_ids = torch.tensor([1, 2], dtype=torch.int32)
    r._copy_valid_sampled_token_count(next_ids, torch.tensor([5, 6], dtype=torch.int32))
    assert r.valid_sampled_token_count_cpu.tolist() == [5, 6, 0]
    assert 'wait' in rec and 'record' in rec
    assert r.input_batch.prev_sampled_token_ids.tolist() == [[1], [2]]
    assert not hasattr(r, 'valid_sampled_token_count_gpu')

def test_copy_valid_sampled_token_count_async_stashes_gpu(monkeypatch):
    monkeypatch.setattr(mr, 'current_platform', _fake_platform())
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = SimpleNamespace(record=lambda : None)
    r.valid_sampled_token_count_copy_stream = SimpleNamespace(wait_stream=lambda s: None)
    r.valid_sampled_token_count_cpu = torch.zeros(2, dtype=torch.int32)
    r.use_async_spec_decode = True
    r.input_batch = SimpleNamespace()
    counts = torch.tensor([4, 4], dtype=torch.int32)
    r._copy_valid_sampled_token_count(torch.tensor([1, 2], dtype=torch.int32), counts)
    assert r.valid_sampled_token_count_gpu is counts

def test_get_valid_sampled_token_count_no_event():
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = None
    r.input_batch = SimpleNamespace(prev_sampled_token_ids=torch.zeros(2, 1))
    assert r._get_valid_sampled_token_count() == []

def test_get_valid_sampled_token_count_no_prev():
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = SimpleNamespace(synchronize=lambda : None)
    r.input_batch = SimpleNamespace(prev_sampled_token_ids=None)
    assert r._get_valid_sampled_token_count() == []

def test_get_valid_sampled_token_count_syncs_and_slices():
    rec = []
    r = object.__new__(ModelRunnerFL)
    r.valid_sampled_token_count_event = SimpleNamespace(synchronize=lambda : rec.append('sync'))
    r.valid_sampled_token_count_cpu = torch.tensor([1, 2, 3], dtype=torch.int32)
    r.input_batch = SimpleNamespace(prev_sampled_token_ids=torch.zeros(2, 1))
    assert r._get_valid_sampled_token_count() == [1, 2]
    assert rec == ['sync']

def _probs_runner():
    r = object.__new__(ModelRunnerFL)
    r.input_batch = SimpleNamespace(req_ids=['r0', 'r1'])
    return r

def test_get_spec_decode_draft_probs_none_inputs():
    r = _probs_runner()
    r._draft_probs = None
    r._draft_prob_req_ids = ['r0']
    assert r._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=[1, 1])) is None
    r._draft_probs = torch.zeros(2, 1, 3)
    r._draft_prob_req_ids = None
    assert r._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=[1, 1])) is None

def test_get_spec_decode_draft_probs_missing_request_returns_none():
    r = _probs_runner()
    r._draft_probs = torch.zeros(2, 2, 3)
    r._draft_prob_req_ids = ['r0']
    out = r._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=[1, 1]))
    assert out is None

def test_get_spec_decode_draft_probs_empty_rows_returns_none():
    r = _probs_runner()
    r._draft_probs = torch.zeros(2, 2, 3)
    r._draft_prob_req_ids = ['r0', 'r1']
    out = r._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=[0, 0]))
    assert out is None

def test_get_spec_decode_draft_probs_concatenates_rows():
    r = _probs_runner()
    probs = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
    r._draft_probs = probs
    r._draft_prob_req_ids = ['r1', 'r0']
    out = r._get_spec_decode_draft_probs(SimpleNamespace(num_draft_tokens=[2, 1]))
    expected = torch.cat([probs[1, :2], probs[0, :1]], dim=0)
    assert torch.equal(out, expected)
    assert out.is_contiguous()

def test_pp_broadcast_calls_distributed(monkeypatch):
    rec = []
    monkeypatch.setattr(mr, 'get_pp_group', lambda : SimpleNamespace(is_last_rank=True, rank=2, device_group='GROUP'))
    monkeypatch.setattr(mr.torch.distributed, 'broadcast', lambda tensor, **k: rec.append((tensor, k)))
    r = object.__new__(ModelRunnerFL)
    r._is_all_reqs_chunked_prefill = lambda : False
    ids = torch.zeros(2, 1, dtype=torch.int32)
    r._pp_broadcast_prev_sampled_token_ids(ids)
    assert len(rec) == 1
    assert rec[0][0] is ids
    assert rec[0][1] == {'src': 2, 'group': 'GROUP'}

def test_pp_broadcast_rejects_bad_shape(monkeypatch):
    monkeypatch.setattr(mr, 'get_pp_group', lambda : SimpleNamespace(is_last_rank=True, rank=0, device_group='G'))
    r = object.__new__(ModelRunnerFL)
    r._is_all_reqs_chunked_prefill = lambda : False
    with pytest.raises(AssertionError):
        r._pp_broadcast_prev_sampled_token_ids(torch.zeros(2, 2, dtype=torch.int32))

def _pp_recv_runner():
    r = object.__new__(ModelRunnerFL)
    r.device = 'cpu'
    r.input_batch = SimpleNamespace(num_reqs=2, req_ids=['r0', 'r1'], num_tokens_no_spec=np.array([3, 4], dtype=np.int32), is_token_ids=np.zeros((2, 8), dtype=bool), prev_sampled_token_ids=None, prev_req_id_to_index=None)
    r.discard_request_mask = SimpleNamespace(np=np.array([False, True], dtype=bool))
    r.requests = {'r0': SimpleNamespace(output_token_ids=[]), 'r1': SimpleNamespace(output_token_ids=[])}
    return r

def _patch_pp(monkeypatch, rec):
    monkeypatch.setattr(mr, 'get_pp_group', lambda : SimpleNamespace(is_last_rank=False, last_rank=3, device_group='GROUP'))
    monkeypatch.setattr(mr.torch.distributed, 'broadcast', lambda tensor, **k: rec.append((tensor, k)) or 'WORK')

def test_pp_receive_skips_chunked_but_updates_bookkeeping(monkeypatch):
    rec = []
    _patch_pp(monkeypatch, rec)
    r = _pp_recv_runner()
    r._is_all_reqs_chunked_prefill = lambda : True
    r._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert rec == []
    assert r.input_batch.prev_sampled_token_ids.shape == (2, 1)
    assert r.input_batch.prev_req_id_to_index == {'r0': 0}
    assert r.requests['r0'].output_token_ids == [-1]
    assert r.requests['r1'].output_token_ids == []
    assert r.input_batch.is_token_ids[0, 3]
    assert r.input_batch.num_tokens_no_spec.tolist() == [4, 4]
    assert getattr(r, '_pp_recv_work', None) is None

def test_pp_receive_starts_async_broadcast(monkeypatch):
    rec = []
    _patch_pp(monkeypatch, rec)
    r = _pp_recv_runner()
    r._is_all_reqs_chunked_prefill = lambda : False
    r._pp_receive_prev_sampled_token_ids_to_input_batch()
    assert r._pp_recv_work == 'WORK'
    assert len(rec) == 1
    assert rec[0][1]['src'] == 3
    assert rec[0][1]['group'] == 'GROUP'
    assert rec[0][1]['async_op'] is True
    assert r.input_batch.prev_req_id_to_index == {'r0': 0}

def _spec(**kw):
    base = dict(method='eagle', use_eagle=lambda : False, use_dflash=lambda : False, uses_draft_model=lambda : False, uses_extract_hidden_states=lambda : False, use_ngram_gpu=lambda : False, disable_padded_drafter_batch=False)
    base.update(kw)
    return SimpleNamespace(**base)

def _sched_prop():
    return SimpleNamespace(total_num_scheduled_tokens=4, num_spec_tokens_to_schedule=3, num_scheduled_tokens={'r0': 2, 'r1': 2})

class _Drafter:
    supports_mm_inputs = False

    def __init__(self, result='DRAFT'):
        self.result = result
        self.calls = []

    def propose(self, *a, **k):
        self.calls.append((a, k))
        return self.result

class _NgramCPU(_Drafter):
    pass

class _NgramGPU(_Drafter):

    def update_token_ids_ngram(self, *a):
        self.calls.append(('update', a))
        return (torch.tensor([1]), 'NG_COUNT', 'NG_VALID')

    def propose(self, *a, **k):
        self.calls.append(('propose', a, k))
        return ('NG_DRAFT', 'NG_NUMVALID')

class _Suffix(_Drafter):
    pass

class _Medusa(_Drafter):
    pass

class _Extract(_Drafter):

    def prepare_next_token_ids_padded(self, *a):
        self.calls.append(('prepared', a))
        return ('EX_NEXT', 'EX_COUNT')

class _EagleF(_Drafter):

    def prepare_next_token_ids_cpu(self, *a):
        self.calls.append(('prepare_cpu', a))
        return 'CPU_NEXT'

    def prepare_next_token_ids_padded(self, *a):
        self.calls.append(('prepare_padded', a))
        return ('PAD_NEXT', 'PAD_COUNT')

    def prepare_inputs(self, *a):
        self.calls.append(('prepare_inputs', a))
        return ('CAM_IN', torch.tensor([0, 1, 2]))

    def prepare_inputs_padded(self, *a):
        self.calls.append(('prepare_padded_inputs', a))
        return (SimpleNamespace(num_actual_tokens=3), 'TIS', 'NREJ')

def _prop_runner(spec, drafter):
    r = object.__new__(ModelRunnerFL)
    r.speculative_config = spec
    r.drafter = drafter
    r.device = 'cpu'
    r.num_spec_tokens = 2
    r.use_aux_hidden_state_outputs = False
    r.supports_mm_inputs = False
    r.discard_request_mask = SimpleNamespace(gpu='DISCARD')
    r.token_ids_gpu_tensor = torch.zeros((2, 8), dtype=torch.int32)
    r.num_tokens_no_spec_gpu = torch.tensor([4, 4], dtype=torch.int32)
    r._num_valid_draft_tokens_cpu = 'NDV_CPU'
    r._num_valid_draft_tokens_copy_stream = 'NDV_STREAM'
    r._num_valid_draft_tokens_event = 'NDV_EVENT'
    r._copy_valid_calls = []
    r._copy_valid_sampled_token_count = lambda n, c: r._copy_valid_calls.append((n, c))
    r._gather_calls = []
    r._gather_mm_embeddings = lambda so, shift_computed_tokens=0: r._gather_calls.append(shift_computed_tokens) or 'MM_EMBEDS'
    r.get_model = lambda : SimpleNamespace()
    r.input_batch = SimpleNamespace(req_ids=['r0', 'r1'], num_tokens_no_spec=np.array([4, 4], dtype=np.int32), token_ids_cpu=np.zeros((2, 8), dtype=np.int32), num_reqs=2, sampling_metadata='SMETA')
    r.requests = {'r0': SimpleNamespace(), 'r1': SimpleNamespace()}
    r.input_ids = SimpleNamespace(gpu=torch.arange(8))
    r._get_positions = lambda n: torch.arange(n) if isinstance(n, int) else torch.arange(len(n))
    return r

def _call_propose(r, sampled, metadata=None, aux=None, **overrides):
    kwargs = dict(hidden_states=torch.zeros(4, 4), sample_hidden_states=torch.zeros(4, 4), aux_hidden_states=aux, spec_decode_metadata=metadata, common_attn_metadata='CAM', slot_mappings='SLOTS')
    kwargs.update(overrides)
    return r.propose_draft_token_ids(_sched_prop(), sampled, 'SMETA', kwargs['hidden_states'], kwargs['sample_hidden_states'], kwargs['aux_hidden_states'], kwargs['spec_decode_metadata'], kwargs['common_attn_metadata'], kwargs['slot_mappings'])

def test_propose_ngram(monkeypatch):
    mod = types.ModuleType('vllm.v1.spec_decode.ngram_proposer')
    mod.NgramProposer = _NgramCPU
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    drafter = _NgramCPU('NGRAM_DRAFT')
    r = _prop_runner(_spec(method='ngram'), drafter)
    out = _call_propose(r, [[1, 2], [3]])
    assert out == 'NGRAM_DRAFT'
    (args, kwargs) = drafter.calls[0]
    assert args[0] == 3
    assert args[1] == [[1, 2], [3]]
    assert kwargs['slot_mappings'] == 'SLOTS'
    assert r._draft_probs is None

def test_propose_custom_class(monkeypatch):
    drafter = _Drafter('CUSTOM_DRAFT')
    r = _prop_runner(_spec(method='custom_class'), drafter)
    out = _call_propose(r, [[1]])
    assert out == 'CUSTOM_DRAFT'
    (args, kwargs) = drafter.calls[0]
    assert args[0] == [[1]]
    assert kwargs['slot_mappings'] == 'SLOTS'

def test_propose_ngram_gpu(monkeypatch):
    monkeypatch.setattr(mr, 'NgramProposerGPU', _NgramGPU)
    rec = []
    monkeypatch.setattr(mr, 'copy_num_valid_draft_tokens', lambda *a: rec.append(a))
    drafter = _NgramGPU()
    r = _prop_runner(_spec(method='ngram_gpu', use_ngram_gpu=lambda : True), drafter)
    out = r.propose_draft_token_ids(_sched_prop(), torch.tensor([[1], [2]], dtype=torch.int32), 'SMETA', torch.zeros(4, 4), torch.zeros(4, 4), None, None, 'CAM', 'SLOTS')
    assert out == 'NG_DRAFT'
    assert len(r._copy_valid_calls) == 1
    (copied_next, copied_counts) = r._copy_valid_calls[0]
    assert torch.equal(copied_next, torch.tensor([1]))
    assert copied_counts == 'NG_COUNT'
    assert r._num_valid_draft_tokens == 'NG_NUMVALID'
    assert len(rec) == 1
    assert rec[0][0] == 'NDV_CPU'
    assert rec[0][4] == 2

def test_propose_suffix(monkeypatch):
    monkeypatch.setattr(mr, 'SuffixDecodingProposer', _Suffix)
    drafter = _Suffix('SUFFIX_DRAFT')
    r = _prop_runner(_spec(method='suffix'), drafter)
    out = _call_propose(r, [[1, 2]])
    assert out == 'SUFFIX_DRAFT'
    (args, kwargs) = drafter.calls[0]
    assert args[0] == 3
    assert args[1] is r.input_batch
    assert kwargs['slot_mappings'] == 'SLOTS'

def test_propose_medusa_equal_shapes(monkeypatch):
    monkeypatch.setattr(mr, 'MedusaProposer', _Medusa)
    drafter = _Medusa('MEDUSA_DRAFT')
    r = _prop_runner(_spec(method='medusa'), drafter)
    sample_hidden = torch.zeros(2, 4)
    out = _call_propose(r, [[1], [2]], sample_hidden_states=sample_hidden)
    assert out == 'MEDUSA_DRAFT'
    (_, kwargs) = drafter.calls[0]
    assert kwargs['num_speculative_tokens'] == 3
    assert kwargs['target_hidden_states'] is sample_hidden

def test_propose_medusa_indexes_hidden_states(monkeypatch):
    monkeypatch.setattr(mr, 'MedusaProposer', _Medusa)
    monkeypatch.setattr(mr, 'async_tensor_h2d', lambda arr, device=None: torch.tensor(arr))
    drafter = _Medusa('MEDUSA_DRAFT')
    r = _prop_runner(_spec(method='medusa'), drafter)
    sample_hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    metadata = SimpleNamespace(num_draft_tokens=[2, 1])
    _call_propose(r, [[1, 2, 3], [4, 5]], metadata=metadata, sample_hidden_states=sample_hidden)
    (_, kwargs) = drafter.calls[0]
    expected = sample_hidden[torch.tensor([2, 4])]
    assert torch.equal(kwargs['target_hidden_states'], expected)

def test_propose_extract_hidden_requires_aux(monkeypatch):
    monkeypatch.setattr(mr, 'ExtractHiddenStatesProposer', _Extract)
    drafter = _Extract('EXTRACT_DRAFT')
    r = _prop_runner(_spec(method='extract', uses_extract_hidden_states=lambda : True), drafter)
    r.use_aux_hidden_state_outputs = False
    with pytest.raises(ValueError, match='aux_hidden_states'):
        _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32))

def test_propose_extract_hidden_success(monkeypatch):
    monkeypatch.setattr(mr, 'ExtractHiddenStatesProposer', _Extract)
    drafter = _Extract('EXTRACT_DRAFT')
    r = _prop_runner(_spec(method='extract', uses_extract_hidden_states=lambda : True), drafter)
    r.use_aux_hidden_state_outputs = True
    aux = [torch.arange(8, dtype=torch.float32).reshape(4, 2)]
    out = _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32), aux=aux)
    assert out == 'EXTRACT_DRAFT'
    (_, kwargs) = drafter.calls[0]
    assert torch.equal(kwargs['target_hidden_states'][0], aux[0][:4])
    assert r._copy_valid_calls == [('EX_NEXT', 'EX_COUNT')]

def test_propose_eagle_disable_padded_no_metadata(monkeypatch):
    monkeypatch.setattr(mr, 'EagleProposer', _EagleF)
    monkeypatch.setattr(mr, 'DFlashProposer', _EagleF)
    monkeypatch.setattr(mr, 'DraftModelProposer', _EagleF)
    monkeypatch.setattr(mr, 'Gemma4Proposer', _EagleF)
    drafter = _EagleF('EAGLE_DRAFT')
    r = _prop_runner(_spec(use_eagle=lambda : True, disable_padded_drafter_batch=True), drafter)
    out = _call_propose(r, [[1, 2], [3]])
    assert out == 'EAGLE_DRAFT'
    assert drafter.calls[0][0] == 'prepare_cpu'
    (_, kwargs) = drafter.calls[-1]
    assert kwargs['next_token_ids'] == 'CPU_NEXT'
    assert kwargs['token_indices_to_sample'] is None
    assert kwargs['target_token_ids'].tolist() == [0, 1, 2, 3]
    assert kwargs['target_hidden_states'].shape == (4, 4)
    assert kwargs['mm_embed_inputs'] is None

def test_propose_eagle_padded_no_metadata_alt_and_probs(monkeypatch):
    monkeypatch.setattr(mr, 'EagleProposer', _EagleF)
    drafter = _EagleF('EAGLE_DRAFT')
    drafter.take_last_draft_probs = lambda : 'DRAFT_PROBS'
    r = _prop_runner(_spec(use_eagle=lambda : True), drafter)
    alt = torch.full((4, 4), 7.0)
    r.get_model = lambda : SimpleNamespace(get_mtp_target_hidden_states=lambda : alt)
    _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32))
    assert drafter.calls[0][0] == 'prepare_padded'
    assert r._copy_valid_calls == [('PAD_NEXT', 'PAD_COUNT')]
    (_, kwargs) = drafter.calls[-1]
    assert torch.equal(kwargs['target_hidden_states'], alt[:4])
    assert r._draft_probs == 'DRAFT_PROBS'
    assert r._draft_prob_req_ids == ['r0', 'r1']

def test_propose_eagle_probabilities_none_leaves_cache_empty(monkeypatch):
    monkeypatch.setattr(mr, 'EagleProposer', _EagleF)
    drafter = _EagleF('EAGLE_DRAFT')
    drafter.take_last_draft_probs = lambda : None
    r = _prop_runner(_spec(use_eagle=lambda : True), drafter)
    _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32))
    assert r._draft_probs is None
    assert r._draft_prob_req_ids is None

def test_propose_eagle_mm_embeddings(monkeypatch):
    monkeypatch.setattr(mr, 'EagleProposer', _EagleF)
    drafter = _EagleF('EAGLE_DRAFT')
    drafter.supports_mm_inputs = True
    r = _prop_runner(_spec(use_eagle=lambda : True), drafter)
    r.supports_mm_inputs = True
    _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32))
    assert r._gather_calls == [1]
    (_, kwargs) = drafter.calls[-1]
    assert kwargs['mm_embed_inputs'] == 'MM_EMBEDS'

def test_propose_eagle_dflash_and_draft_model_accept(monkeypatch):
    monkeypatch.setattr(mr, 'DFlashProposer', _EagleF)
    drafter = _EagleF('DFLASH_DRAFT')
    r = _prop_runner(_spec(use_dflash=lambda : True), drafter)
    out = _call_propose(r, torch.tensor([[1], [2]], dtype=torch.int32))
    assert out == 'DFLASH_DRAFT'