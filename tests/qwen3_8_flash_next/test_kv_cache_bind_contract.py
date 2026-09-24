# SPDX-License-Identifier: Apache-2.0
"""C6: vLLM 0.24 KV binding must call the cache-owner hook exactly once, after
the upstream bind, and identify owners by registration rather than module name."""

import pytest
import torch

import vllm_fl.worker.model_runner as mr
from vllm_fl.compat.vllm024 import kv_cache as kv_bind
from vllm_fl.models.qwen3_8_flash_next.common.qsa_cache import QSAKeyStateCache


class _Owner:
    def __init__(self):
        self.bound = []

    def bind_kv_cache(self, kv_cache):
        self.bound.append(kv_cache)


class _UnregisteredOwner:
    """Same interface as _Owner but never registered (no shared base)."""

    def __init__(self):
        self.bound = []

    def bind_kv_cache(self, kv_cache):
        self.bound.append(kv_cache)


kv_bind.register_kv_cache_owner(_Owner)


class _FailingOwner:
    def __init__(self):
        self.kv_cache = None

    def validate_kv_cache(self, kv_cache):
        pass

    def bind_kv_cache(self, kv_cache):
        self.kv_cache = kv_cache
        raise RuntimeError("hook failed after partial bind")


kv_bind.register_kv_cache_owner(_FailingOwner)


def _noop_upstream(monkeypatch, *, mutates=True):
    calls = []

    def fake(kv_caches, forward_context, runner_kv_caches, num_attn_module=1):
        calls.append(list(kv_caches))
        if mutates:
            for layer_name, kv in kv_caches.items():
                forward_context[layer_name].kv_cache = kv
                runner_kv_caches.append(kv)

    import vllm.v1.worker.utils as vllm_utils

    monkeypatch.setattr(vllm_utils, "bind_kv_cache", fake)
    monkeypatch.setattr(kv_bind, "resolve_kv_bind_abi", lambda: False)
    return calls


def test_owner_hook_called_once_after_upstream(monkeypatch):
    upstream_calls = _noop_upstream(monkeypatch)
    owner = _Owner()
    kv_caches = {"qsa": torch.zeros(2, 4, 1, 8)}
    forward_context = {"qsa": owner}

    kv_bind.bind_kv_cache(kv_caches, forward_context, [], 1)
    assert len(upstream_calls) == 1
    assert len(owner.bound) == 1
    assert owner.bound[0] is kv_caches["qsa"]


def test_unregistered_layer_is_not_called(monkeypatch):
    _noop_upstream(monkeypatch)
    unregistered = _UnregisteredOwner()
    kv_caches = {"qsa": torch.zeros(1)}
    kv_bind.bind_kv_cache(kv_caches, {"qsa": unregistered}, [], 1)
    assert unregistered.bound == []


def test_native_upstream_hook_is_not_duplicated(monkeypatch):
    _noop_upstream(monkeypatch)
    monkeypatch.setattr(kv_bind, "resolve_kv_bind_abi", lambda: True)
    owner = _Owner()
    kv_bind.bind_kv_cache({"qsa": torch.zeros(1)}, {"qsa": owner}, [], 1)
    assert owner.bound == []


def test_validation_failure_leaves_no_mutation(monkeypatch):
    _noop_upstream(monkeypatch)
    owner = _Owner()
    owner.validate_kv_cache = lambda kv: (_ for _ in ()).throw(ValueError("bad dtype"))
    runner: list = []
    kv_caches = {"qsa": torch.zeros(1)}
    with pytest.raises(ValueError, match="bad dtype"):
        kv_bind.bind_kv_cache(kv_caches, {"qsa": owner}, runner, 1)
    assert runner == []
    assert owner.bound == []


def test_hook_failure_aborts_initialization_and_does_not_run_later_owners(monkeypatch):
    _noop_upstream(monkeypatch)
    failing, later = _FailingOwner(), _Owner()
    caches = {"failing": torch.zeros(1), "later": torch.zeros(1)}
    with pytest.raises(RuntimeError, match="hook failed"):
        kv_bind.bind_kv_cache(caches, {"failing": failing, "later": later}, [])
    assert later.bound == []


def test_runner_binds_registered_owner_exactly_once(monkeypatch):
    upstream_calls = _noop_upstream(monkeypatch)
    owner = _Owner()
    kv_cache = torch.zeros(2, 4, 1, 8, dtype=torch.bfloat16)

    runner = type("Runner", (), {})()
    runner.cache_config = type("C", (), {"cache_dtype": "auto"})()
    runner.attn_groups = []
    runner.use_uniform_kv_cache = lambda groups: False
    runner._allocate_kv_cache_tensors = lambda cfg: {"qsa": torch.empty(0)}
    runner._reshape_kv_cache_tensors = lambda raw, kbs: {"qsa": kv_cache}
    runner.shared_kv_cache_layers = {}
    runner.compilation_config = type(
        "CC", (), {"static_forward_context": {"qsa": owner}}
    )()
    runner.model_config = type(
        "MC", (), {"hf_config": type("HF", (), {"model_type": "qwen3"})()}
    )()
    runner.kv_caches = []
    runner.device = "cpu"

    mr.ModelRunnerFL.initialize_kv_cache_tensors(runner, type("KVC", (), {})(), [])

    assert len(upstream_calls) == 1
    assert len(owner.bound) == 1


def test_qsa_validate_rejects_non_contiguous_last_dim():
    """Finding 5: reject a layout whose typed int64 view cannot be built."""
    obj = object.__new__(QSAKeyStateCache)
    obj.head_size = 512
    obj.key_head_size = 512
    obj.cache_rope_positions = False
    strided = torch.zeros(2, 4, 1, 1024, dtype=torch.bfloat16)[..., ::2]
    assert strided.stride(-1) == 2
    with pytest.raises(ValueError, match="contiguous last dimension"):
        obj.validate_kv_cache(strided)


def test_qsa_validate_rejects_view_incompatible_outer_stride():
    """Finding P2-2: offset alignment alone lets a bad outer stride through."""
    key_head = 140
    obj = object.__new__(QSAKeyStateCache)
    obj.key_head_size = key_head
    obj.rope_position_offset = ((key_head + 3) // 4) * 4
    obj._BF16_PER_INT64 = 4
    obj.cache_rope_positions = True
    obj.head_size = obj.rope_position_offset + 2 * obj._BF16_PER_INT64  # 148

    width = obj.head_size
    base = torch.zeros(2048, dtype=torch.bfloat16)
    bad = torch.as_strided(base, (2, 2, 1, width), (width * 2 + 1, width, width, 1))
    assert bad.stride(-1) == 1
    assert (bad.storage_offset() + obj.rope_position_offset) % 4 == 0
    with pytest.raises(ValueError, match="cannot be viewed as int64"):
        obj.validate_kv_cache(bad)

    good = torch.zeros(2, 2, 1, width, dtype=torch.bfloat16)
    obj.validate_kv_cache(good)


def test_rebind_replaces_typed_views():
    key_head_size = 512
    obj = object.__new__(QSAKeyStateCache)
    obj.head_size = ((key_head_size + 3) // 4) * 4 + 3 * 4
    obj.key_head_size = key_head_size
    obj.cache_rope_positions = True
    obj.rope_position_offset = ((key_head_size + 3) // 4) * 4
    obj.kv_cache = torch.tensor([])

    first = torch.zeros(2, 4, 1, obj.head_size, dtype=torch.bfloat16)
    obj.bind_kv_cache(first)
    assert (
        obj.key_cache.untyped_storage().data_ptr() == first.untyped_storage().data_ptr()
    )

    second = torch.zeros(2, 4, 1, obj.head_size, dtype=torch.bfloat16)
    obj.bind_kv_cache(second)
    assert (
        obj.key_cache.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )
    assert (
        obj.rope_position_cache.untyped_storage().data_ptr()
        == second.untyped_storage().data_ptr()
    )
    assert obj.rope_position_cache.dtype == torch.int64
    assert obj.rope_position_cache.storage_offset() == obj.rope_position_offset // 4


def test_unknown_abi_fails_before_upstream_bind(monkeypatch):
    calls = _noop_upstream(monkeypatch)

    def fail_abi():
        raise RuntimeError("Unverified ABI")

    monkeypatch.setattr(kv_bind, "resolve_kv_bind_abi", fail_abi)
    owner = _Owner()
    with pytest.raises(RuntimeError, match="Unverified ABI"):
        kv_bind.bind_kv_cache({"qsa": torch.zeros(1)}, {"qsa": owner}, [])
    assert calls == []
    assert owner.bound == []
