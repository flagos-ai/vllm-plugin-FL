"""Coverage for vllm_fl.worker.model_runner._prepare_inputs (w10, local CPU)."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_fl.worker import model_runner as mr


def make_runner(**attrs):
    runner = object.__new__(mr.ModelRunnerFL)
    runner.__dict__.update(attrs)
    return runner


def buf(size, np_dtype=np.int32, torch_dtype=torch.int32, value=0):
    """CpuGpuBuffer stand-in whose .np is a view of .cpu."""
    cpu = torch.full((size,), value, dtype=torch_dtype)
    npv = cpu.numpy()
    gpu = torch.full((size,), value, dtype=torch_dtype)

    def copy_to_gpu(n=None):
        n = gpu.numel() if n is None else n
        gpu[:n].copy_(cpu[:n])

    return SimpleNamespace(np=npv, cpu=cpu, gpu=gpu, copy_to_gpu=copy_to_gpu)


def base_setup(num_reqs=2, num_sched=(2, 2), max_len=8, buf_size=16, **over):
    num_scheduled_tokens = np.array(num_sched, dtype=np.int32)
    token_ids = np.arange(num_reqs * max_len, dtype=np.int32).reshape(num_reqs, max_len)
    token_ids_t = torch.from_numpy(token_ids.copy())
    input_batch = SimpleNamespace(
        num_reqs=num_reqs,
        block_table=SimpleNamespace(
            commit_block_table=lambda *a, **k: None,
            compute_slot_mapping=lambda *a, **k: None,
        ),
        token_ids_cpu=token_ids,
        token_ids_cpu_tensor=token_ids_t,
        is_token_ids_tensor=torch.zeros_like(token_ids_t),
        req_prompt_embeds={},
        num_computed_tokens_cpu=np.zeros(num_reqs, dtype=np.int32),
        num_computed_tokens_cpu_tensor=torch.zeros(num_reqs, dtype=torch.int32),
        prev_req_id_to_index=None,
        req_ids=[f"r{i}" for i in range(num_reqs)],
        req_id_to_index={f"r{i}": i for i in range(num_reqs)},
        num_accepted_tokens_cpu=np.zeros(num_reqs, dtype=np.int32),
        prev_sampled_token_ids=None,
    )
    requests = {
        f"r{i}": SimpleNamespace(num_tokens=100) for i in range(num_reqs)
    }
    attrs = dict(
        input_batch=input_batch,
        requests=requests,
        arange_np=np.arange(buf_size),
        query_pos=buf(buf_size),
        query_start_loc=buf(buf_size),
        req_indices=buf(buf_size),
        num_scheduled_tokens=buf(buf_size),
        discard_request_mask=buf(buf_size, bool, torch.bool),
        prev_positions=buf(buf_size, np.int64, torch.int64, -1),
        num_accepted_tokens=buf(buf_size),
        num_decode_draft_tokens=buf(buf_size),
        prev_num_draft_tokens=buf(4),
        input_ids=buf(buf_size),
        inputs_embeds=None,
        is_token_ids=None,
        positions=torch.zeros(buf_size, dtype=torch.int64),
        seq_lens=torch.zeros(buf_size, dtype=torch.int32),
        num_computed_tokens=torch.zeros(buf_size, dtype=torch.int32),
        optimistic_seq_lens_cpu=torch.zeros(buf_size, dtype=torch.int32),
        device=torch.device("cpu"),
        use_ascend_attention=False,
        uses_mrope=False,
        enable_prompt_embeds=False,
        use_async_scheduling=False,
        use_async_spec_decode=False,
        valid_sampled_token_count_gpu=None,
        num_accepted_tokens_event=None,
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        scheduler_config=SimpleNamespace(enable_chunked_prefill=True),
        speculative_config=None,
        lora_config=None,
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(max_num_batched_tokens=8192)
        ),
    )
    attrs.update(over)
    return make_runner(**attrs), num_scheduled_tokens


@pytest.fixture(autouse=True)
def _cpu_pinned(monkeypatch):
    monkeypatch.setattr(mr, "PIN_MEMORY", False, raising=False)


def test_prepare_inputs_plain_decode_basic():
    r, nst = base_setup()
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    r._kl_guard_on = lambda: False
    seen = {}

    def rec_prepare(*a, **k):
        seen["args"] = a

    r._prepare_input_ids = rec_prepare
    logits, meta, max_sampled = r._prepare_inputs(sched, nst)

    assert seen["args"][1] == 2  # num_reqs
    assert seen["args"][2] == 4  # total
    assert logits.tolist() == [1, 3]
    assert meta is None
    assert max_sampled == 1
    # input_ids gathered via index_select from [0,1,8,9]
    assert r.input_ids.cpu[:4].tolist() == [0, 1, 8, 9]
    assert r.positions[:4].tolist() == [0, 1, 0, 1]
    assert r.seq_lens[:2].tolist() == [2, 2]
    assert r.seq_lens[2:].tolist() == [0] * (r.seq_lens.numel() - 2)
    assert r.optimistic_seq_lens_cpu[:2].tolist() == [2, 2]
    assert r.discard_request_mask.np[:2].tolist() == [True, True]
    assert r.query_start_loc.gpu[:3].tolist() == [0, 2, 4]
    assert r.num_accepted_tokens.np[:4].tolist() == [1, 1, 1, 1]


def test_prepare_inputs_kl_guard_uses_cpu_positions():
    r, nst = base_setup()
    r._kl_guard_on = lambda: True
    r._prepare_input_ids = lambda *a, **k: None
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    r._prepare_inputs(sched, nst)
    assert r.positions[:4].tolist() == [0, 1, 0, 1]


def test_prepare_inputs_spec_decode_path():
    r, nst = base_setup()
    r._kl_guard_on = lambda: False
    r._prepare_input_ids = lambda *a, **k: None
    sentinel = SimpleNamespace(logits_indices=torch.tensor([5, 6, 7]))
    r._calc_spec_decode_metadata = lambda ndt, cu: sentinel
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={"r0": [99]},
    )
    logits, meta, max_sampled = r._prepare_inputs(sched, nst)
    assert logits is sentinel.logits_indices
    assert meta is sentinel
    assert max_sampled == 2  # num_draft_tokens [1,0] + 1
    # r0 had 2 scheduled == draft_len+1 -> decode draft kept; r1 -> -1
    assert r.num_decode_draft_tokens.np[:2].tolist() == [1, -1]
    assert r.num_decode_draft_tokens.np[2:].tolist() == [-1] * (
        r.num_decode_draft_tokens.np.size - 2
    )


def test_prepare_inputs_lora_sets_active_loras():
    r, nst = base_setup()
    r._kl_guard_on = lambda: False
    r._prepare_input_ids = lambda *a, **k: None
    r.lora_config = SimpleNamespace()
    calls = {}
    r.set_active_loras = lambda ib, nst_, nsp: calls.setdefault(
        "ok", (int(np.sum(nsp)), nst_.tolist())
    )
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    r._prepare_inputs(sched, nst)
    assert calls["ok"] == (2, [2, 2])


def test_prepare_inputs_async_spec_decode_fixup_and_align_accepted(monkeypatch):
    r, nst = base_setup()
    r.use_async_spec_decode = True
    r.use_async_scheduling = True
    r.cache_config = SimpleNamespace(mamba_cache_mode="align")
    r.valid_sampled_token_count_gpu = torch.ones(4, dtype=torch.int32)
    r.prev_positions.np[:2] = np.array([1, -1], dtype=np.int64)
    r.input_batch.prev_req_id_to_index = {"r0": 1}
    r.input_batch.num_accepted_tokens_cpu = np.array([3, 4], dtype=np.int32)
    r.num_accepted_tokens_event = SimpleNamespace(synchronize=lambda: None)
    r._kl_guard_on = lambda: False
    r._prepare_input_ids = lambda *a, **k: None
    fixup = {}
    monkeypatch.setattr(
        mr,
        "update_num_computed_tokens_for_batch_change",
        lambda *a, **k: fixup.setdefault("called", True),
    )
    r.num_computed_tokens[:2] = 5
    positions_before = r.positions.clone()
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    r._prepare_inputs(sched, nst)
    assert fixup.get("called") is True
    # align async accepted branch: row 0 pulled from prev idx 1 -> 4; new row -> 1
    assert r.num_accepted_tokens.np[:2].tolist() == [4, 1]
    # async fixup path keeps the GPU-derived positions (not CPU positions)
    assert not torch.equal(r.positions, positions_before)


def test_prepare_inputs_non_async_accepted_counts():
    r, nst = base_setup()
    r.num_accepted_tokens_event = SimpleNamespace(synchronize=lambda: None)
    r.input_batch.num_accepted_tokens_cpu = np.array([7, 8], dtype=np.int32)
    r._kl_guard_on = lambda: False
    r._prepare_input_ids = lambda *a, **k: None
    sched = SimpleNamespace(
        total_num_scheduled_tokens=4, scheduled_spec_decode_tokens={}
    )
    r._prepare_inputs(sched, nst)
    assert r.num_accepted_tokens.np[:2].tolist() == [7, 8]
    assert r.num_accepted_tokens.np[2:].tolist() == [1] * (
        r.num_accepted_tokens.np.size - 2
    )
