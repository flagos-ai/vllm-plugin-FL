# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contract tests; only the native device memcpy launch is substituted."""

import ctypes
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.mamba import mamba_utils as state_utils
from vllm.model_executor.layers.mamba.mamba_utils import (
    get_conv_copy_spec,
    get_temporal_copy_spec,
)
from vllm.v1.kv_cache_interface import MambaAttentionBackendEnum, MambaSpec
from vllm.v1.worker import mamba_utils as native

from vllm_fl.worker import mamba_buffers as compat


class CPUBuffer:
    def __init__(self, size, *, dtype):
        self.cpu = torch.zeros(size, dtype=dtype)
        self.np = self.cpu.numpy()
        self.gpu = self.cpu

    def copy_to_gpu(self, n):
        return self.cpu[:n]


def specs():
    gdn = MambaSpec(
        block_size=4,
        shapes=((3, 5), (2, 3)),
        dtypes=(torch.bfloat16, torch.float32),
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        mamba_cache_mode="align",
        page_size_padded=128,
    )
    ple = MambaSpec(
        block_size=4,
        shapes=((4, 7),),
        dtypes=(torch.bfloat16,),
        mamba_type=MambaAttentionBackendEnum.SHORT_CONV,
        mamba_cache_mode="align",
        page_size_padded=128,
    )
    return gdn, ple


class Model:
    def get_mamba_state_copy_func(self):
        return (get_conv_copy_spec, get_temporal_copy_spec)

    def get_mamba_state_copy_funcs(self, types):
        mapping = {
            MambaAttentionBackendEnum.GDN_ATTN: self.get_mamba_state_copy_func(),
            MambaAttentionBackendEnum.SHORT_CONV: (get_conv_copy_spec,),
        }
        return {kind: mapping[kind] for kind in types}


def config(gdn=None, ple=None):
    default_gdn, default_ple = specs()
    return SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(kv_cache_spec=gdn or default_gdn, layer_names=["gdn.a"]),
            SimpleNamespace(kv_cache_spec=object(), layer_names=["attention"]),
            SimpleNamespace(kv_cache_spec=ple or default_ple, layer_names=["ple"]),
            SimpleNamespace(kv_cache_spec=gdn or default_gdn, layer_names=["gdn.b"]),
        ]
    )


def create(cfg, **overrides):
    kwargs = dict(
        model=Model(),
        max_num_reqs=8,
        kv_cache_config=cfg,
        make_buffer=CPUBuffer,
        device=torch.device("cpu"),
        with_postprocess_align=False,
    )
    kwargs.update(overrides)
    return compat.create_mamba_buffers(**kwargs)


def test_native_homogeneous_path_is_forwarded_unchanged(monkeypatch):
    gdn, _ = specs()
    cfg = config(ple=gdn)
    model = Model()
    model.get_mamba_state_copy_funcs = Mock(side_effect=AssertionError("unused"))
    result = object()
    factory = Mock(return_value=result)
    monkeypatch.setattr(native.MambaBuffers, "create", factory)
    assert create(cfg, model=model, with_postprocess_align=True) is result
    factory.assert_called_once_with(
        max_num_reqs=8,
        kv_cache_config=cfg,
        copy_funcs=model.get_mamba_state_copy_func(),
        make_buffer=CPUBuffer,
        device=torch.device("cpu"),
        with_postprocess_align=True,
    )
    model.get_mamba_state_copy_funcs.assert_not_called()


def test_native_preprocess_is_forwarded_unchanged(monkeypatch):
    args = tuple(object() for _ in range(9))
    target = Mock()
    monkeypatch.setattr(native, "preprocess_mamba", target)
    compat.preprocess_mamba(*args)
    target.assert_called_once_with(*args)


def test_group_capacity_and_original_ids():
    bufs = create(config())
    assert bufs.postprocess_align is None
    assert len(bufs.preprocess.groups) == 2
    gdn_group, ple_group = bufs.preprocess.groups
    assert gdn_group[0].mamba_group_ids == [0, 3]
    assert ple_group[0].mamba_group_ids == [2]
    assert gdn_group[0].src_ptrs.np.size == 8 * 2 * 2
    assert ple_group[0].src_ptrs.np.size == 8 * 1 * 1
    assert gdn_group[1] == (get_conv_copy_spec, get_temporal_copy_spec)
    assert ple_group[1] == (get_conv_copy_spec,)
    assert gdn_group[2] is not ple_group[2]


@pytest.mark.parametrize("bad", ["missing_type", "too_few_funcs", "too_many_funcs"])
def test_invalid_copy_contract_fails_before_allocating(bad):
    model = Model()
    mapping = model.get_mamba_state_copy_funcs(
        {
            kind
            for kind in (
                MambaAttentionBackendEnum.GDN_ATTN,
                MambaAttentionBackendEnum.SHORT_CONV,
            )
        }
    )
    if bad == "missing_type":
        del mapping[MambaAttentionBackendEnum.SHORT_CONV]
    elif bad == "too_few_funcs":
        mapping[MambaAttentionBackendEnum.GDN_ATTN] = (get_conv_copy_spec,)
    else:
        mapping[MambaAttentionBackendEnum.SHORT_CONV] = (
            get_conv_copy_spec,
            get_temporal_copy_spec,
        )
    model.get_mamba_state_copy_funcs = Mock(return_value=mapping)
    allocator = Mock(side_effect=AssertionError("must validate before allocating"))
    with pytest.raises(ValueError):
        create(config(), model=model, make_buffer=allocator)
    allocator.assert_not_called()


def test_heterogeneous_spec_decode_is_rejected_before_allocating():
    allocator = Mock(side_effect=AssertionError("must reject before allocating"))
    with pytest.raises(NotImplementedError):
        create(config(), make_buffer=allocator, with_postprocess_align=True)
    allocator.assert_not_called()


def test_missing_type_aware_copy_api_is_rejected_before_allocating():
    model = SimpleNamespace(get_mamba_state_copy_func=Model().get_mamba_state_copy_func)
    allocator = Mock(side_effect=AssertionError("must validate before allocating"))
    with pytest.raises(ValueError, match="get_mamba_state_copy_funcs"):
        create(config(), model=model, make_buffer=allocator)
    allocator.assert_not_called()


def test_spec_shape_dtype_arity_mismatch_is_rejected_before_allocating():
    gdn, _ = specs()
    cfg = config(gdn=replace(gdn, dtypes=(torch.bfloat16,)))
    allocator = Mock(side_effect=AssertionError("must validate before allocating"))
    with pytest.raises(ValueError):
        create(cfg, make_buffer=allocator)
    allocator.assert_not_called()


def make_context(cfg):
    context, storages = {}, {}
    for gid, group in enumerate(cfg.kv_cache_groups):
        if not isinstance(group.kv_cache_spec, MambaSpec):
            continue
        for layer in group.layer_names:
            states = []
            for state_id, (shape, dtype) in enumerate(
                zip(group.kv_cache_spec.shapes, group.kv_cache_spec.dtypes, strict=True)
            ):
                page = int(np.prod(shape)) + 19
                # Padding deliberately differs from the natural state size.
                storage = torch.arange(6 * page, dtype=torch.float32).to(dtype)
                storage.add_(gid * 512 + state_id * 64)
                inner_strides = tuple(
                    int(np.prod(shape[i + 1 :])) for i in range(len(shape))
                )
                state = storage.as_strided((6, *shape), (page, *inner_strides))
                states.append(state)
                storages[(layer, state_id)] = (storage, state)
            context[layer] = SimpleNamespace(kv_cache=states)
    return context, storages


@pytest.fixture
def cpu_memcpy(monkeypatch):
    calls = []
    monkeypatch.setattr(state_utils, "is_conv_state_dim_first", lambda: False)

    def memcpy(src, dst, sizes):
        copied = list(zip(src.tolist(), dst.tolist(), sizes.tolist(), strict=True))
        calls.append(copied)
        for source, destination, size in copied:
            ctypes.memmove(destination, source, size)

    monkeypatch.setattr(native, "batch_memcpy", memcpy)
    return calls


def schedule(computed=4, scheduled=1, accepted=1):
    req = SimpleNamespace(
        num_computed_tokens=computed,
        block_ids=([1, 4, 5], [], [2, 3, 0], [0, 2, 4]),
    )
    output = SimpleNamespace(
        num_scheduled_tokens={"req": scheduled},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids=set()),
    )
    batch = SimpleNamespace(
        req_ids=["req"], num_accepted_tokens_cpu=np.array([accepted], dtype=np.int32)
    )
    return output, batch, {"req": req}


def independent_expected(cfg, storages, requests, scheduled, accepted, previous=None):
    expected = {}
    req = requests["req"]
    for gid, group in enumerate(cfg.kv_cache_groups):
        spec = group.kv_cache_spec
        if not isinstance(spec, MambaSpec):
            continue
        src_idx = (req.num_computed_tokens - 1) // spec.block_size
        if previous is not None:
            src_idx = previous[gid]
        dst_idx = (req.num_computed_tokens + scheduled - 1) // spec.block_size
        for layer in group.layer_names:
            for state_id in range(len(spec.shapes)):
                storage, state = storages[(layer, state_id)]
                target_storage = storage.clone()
                target = target_storage.as_strided(state.shape, state.stride())
                if src_idx >= 0 and src_idx != dst_idx:
                    # Independent expected values, without target copy metadata/functions.
                    if state_id == 0:
                        value = state[
                            req.block_ids[gid][src_idx], accepted - 1 :
                        ].clone()
                    else:
                        value = state[
                            req.block_ids[gid][src_idx + accepted - 1]
                        ].clone()
                    target[req.block_ids[gid][dst_idx]].view(-1)[: value.numel()].copy_(
                        value.flatten()
                    )
                expected[(layer, state_id)] = target_storage
    return expected


def run(bufs, cfg, output, batch, requests, context):
    compat.preprocess_mamba(
        output,
        cfg,
        SimpleNamespace(enable_prefix_caching=True, block_size=4),
        {},
        batch,
        requests,
        context,
        Model().get_mamba_state_copy_func(),
        bufs.preprocess,
    )


@pytest.mark.parametrize(
    "computed,scheduled,accepted", [(0, 2, 1), (4, 1, 1), (5, 1, 1), (4, 5, 2)]
)
def test_native_copy_plan_matches_independent_values(
    cpu_memcpy, computed, scheduled, accepted
):
    cfg = config()
    bufs = create(cfg)
    context, storages = make_context(cfg)
    output, batch, requests = schedule(computed, scheduled, accepted)
    expected = independent_expected(cfg, storages, requests, scheduled, accepted)
    run(bufs, cfg, output, batch, requests, context)
    for key, value in expected.items():
        torch.testing.assert_close(storages[key][0], value, rtol=0, atol=0)
    if computed == 4:
        assert sum(len(call) for call in cpu_memcpy) == 5
        assert batch.num_accepted_tokens_cpu.tolist() == [1]
    else:
        assert not cpu_memcpy


def test_different_block_sizes_are_rejected_before_allocating():
    gdn, ple = specs()
    cfg = config(gdn=gdn, ple=replace(ple, block_size=8))
    allocator = Mock(side_effect=AssertionError("must validate before allocating"))
    with pytest.raises(ValueError):
        create(cfg, make_buffer=allocator)
    allocator.assert_not_called()


def test_dim_first_conv_layout_at_zero_bias(cpu_memcpy, monkeypatch):
    monkeypatch.setattr(state_utils, "is_conv_state_dim_first", lambda: True)
    gdn, ple = specs()
    cfg = config(
        gdn=replace(gdn, shapes=((5, 3), (2, 3))),
        ple=replace(ple, shapes=((7, 4),)),
    )
    bufs = create(cfg)
    context, storages = make_context(cfg)
    output, batch, requests = schedule()
    expected = independent_expected(cfg, storages, requests, 1, 1)
    run(bufs, cfg, output, batch, requests, context)
    for key, value in expected.items():
        torch.testing.assert_close(storages[key][0], value, rtol=0, atol=0)
    assert sum(len(call) for call in cpu_memcpy) == 5


@pytest.mark.parametrize("invalid", ["missing_state", "extra_state", "shape", "dtype"])
def test_invalid_later_group_state_rejected_before_any_copy(monkeypatch, invalid):
    cfg = config()
    bufs = create(cfg)
    context, storages = make_context(cfg)
    before = {key: storage.clone() for key, (storage, _) in storages.items()}
    if invalid == "missing_state":
        context["ple"].kv_cache = []
    elif invalid == "extra_state":
        context["ple"].kv_cache *= 2
    elif invalid == "shape":
        context["ple"].kv_cache[0] = context["ple"].kv_cache[0][:, :3]
    else:
        context["ple"].kv_cache[0] = context["ple"].kv_cache[0].float()
    target = Mock(side_effect=AssertionError("must validate every group before copy"))
    monkeypatch.setattr(native, "preprocess_mamba", target)
    output, batch, requests = schedule()
    with pytest.raises(ValueError):
        run(bufs, cfg, output, batch, requests, context)
    target.assert_not_called()
    for key, value in before.items():
        torch.testing.assert_close(storages[key][0], value, rtol=0, atol=0)


def test_recreated_buffers_drop_validation_and_request_state():
    cfg = config()
    old = create(cfg)
    context, _ = make_context(cfg)
    old.preprocess.validate_states(cfg, context)
    for _, _, states in old.preprocess.groups:
        states["req"] = 8
    new = create(cfg)
    assert not new.preprocess.validated
    assert all(not states for _, _, states in new.preprocess.groups)
    assert all(states == {"req": 8} for _, _, states in old.preprocess.groups)


def test_resume_removes_stale_group_state_and_finished_entries(cpu_memcpy):
    cfg = config()
    bufs = create(cfg)
    for _, _, state in bufs.preprocess.groups:
        state.update({"req": 99, "finished": 8, "preempted": 7})
    context, storages = make_context(cfg)
    output, batch, requests = schedule()
    output.finished_req_ids.add("finished")
    output.preempted_req_ids.add("preempted")
    output.scheduled_cached_reqs.resumed_req_ids.add("req")
    expected = independent_expected(cfg, storages, requests, 1, 1)
    run(bufs, cfg, output, batch, requests, context)
    for _, _, state in bufs.preprocess.groups:
        assert state == {"req": 1}
    for key, value in expected.items():
        torch.testing.assert_close(storages[key][0], value, rtol=0, atol=0)


def test_empty_batch_cleans_all_groups_without_copy(cpu_memcpy):
    cfg = config()
    bufs = create(cfg)
    context, _ = make_context(cfg)
    for _, _, state in bufs.preprocess.groups:
        state["req"] = 2
    output, batch, requests = schedule()
    output.finished_req_ids.add("req")
    output.num_scheduled_tokens = {}
    batch.req_ids = []
    batch.num_accepted_tokens_cpu = np.array([], dtype=np.int32)
    run(bufs, cfg, output, batch, {}, context)
    assert all(not state for _, _, state in bufs.preprocess.groups)
    assert not cpu_memcpy


def test_execution_error_propagates_without_fallback(monkeypatch):
    cfg = config()
    bufs = create(cfg)
    context, _ = make_context(cfg)
    output, batch, requests = schedule()
    failure = RuntimeError("copy execution failed")
    target = Mock(side_effect=failure)
    monkeypatch.setattr(native, "preprocess_mamba", target)
    with pytest.raises(RuntimeError, match="copy execution failed"):
        run(bufs, cfg, output, batch, requests, context)
    target.assert_called_once()
