# SPDX-License-Identifier: Apache-2.0
"""CPU host metadata contracts with isolated producer and CUDA scheduling spies.

These tests load the shared host facts and CUDA owner modules from their real
source paths. Producer spies only model host scheduling and replay-visible
inputs; they do not validate Triton math, CUDA graphs, or full plugin imports.
"""

import copy
import gc
import importlib.util
import sys
import weakref
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch

PLUGIN = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4"
HOST_SOURCE = PLUGIN / "metadata.py"
CUDA_SOURCE = PLUGIN / "nvidia/metadata.py"


@pytest.fixture
def host_metadata(monkeypatch):
    name = "hy4_public_metadata_cpu"
    spec = importlib.util.spec_from_file_location(name, HOST_SOURCE)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    before = {
        name
        for name in sys.modules
        if name.startswith(("flag_gems", "flaggems_vllm", "vllm"))
    }
    spec.loader.exec_module(module)
    after = {
        name
        for name in sys.modules
        if name.startswith(("flag_gems", "flaggems_vllm", "vllm"))
    }
    assert before == after
    return module


@pytest.fixture
def metadata_modules(monkeypatch):
    package = ModuleType("hy4_metadata_platform_contract")
    package.__path__ = [str(PLUGIN)]
    vendor_package = ModuleType(package.__name__ + ".nvidia")
    vendor_package.__path__ = [str(CUDA_SOURCE.parent)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, vendor_package.__name__, vendor_package)

    def load(name, source):
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    host = load(package.__name__ + ".metadata", HOST_SOURCE)
    vendor = load(vendor_package.__name__ + ".metadata", CUDA_SOURCE)
    assert vendor._cpu_sequence_proof is host._cpu_sequence_proof
    assert vendor._RequestIdMapLease is host._RequestIdMapLease
    return SimpleNamespace(host=host, vendor=vendor, package=package)


def test_input_maps_follow_reordered_request_history(host_metadata):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[-1, 1, 0]),
        input_batch=SimpleNamespace(req_ids=["new", "old-a", "old-b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"old-a": [123]})
    maps = host_metadata.build_input_maps(runner, scheduled, 3, 8, [2, 6, 8])
    assert maps.sample_dst == [4, 7]
    assert maps.sample_src == [1, 0]
    assert maps.draft_dst == [5]
    assert maps.draft_src == [3]
    assert maps.num_sampled == 2
    assert maps.copy_base is True


def test_unchanged_dense_sample_map_stays_native(host_metadata):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[0, 1]),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={})
    assert host_metadata.build_input_maps(runner, scheduled, 2, 2, [1, 2]) is None


def test_host_packet_layout_and_untouched_tail(host_metadata):
    packet = np.full(32, 99, dtype=np.int64)
    used = host_metadata.pack_host_packet(
        packet, [4, 7, 5], [1, 0, 3], [-1, 1, 0], [0, 8, 16], [0, 0, 0, 0, 0]
    )
    assert used == 17
    assert packet[:used].tolist() == [
        4,
        7,
        5,
        1,
        0,
        3,
        -1,
        1,
        0,
        0,
        8,
        16,
        0,
        0,
        0,
        0,
        0,
    ]
    assert packet[used:].tolist() == [99] * 15


def test_packet_proof_has_one_local_type_and_owns_copied_facts(metadata_modules):
    host_metadata = metadata_modules.host
    packet = np.array(
        [4, 7, 5, 1, 0, 3, -1, 1, 0, 0, 8, 16, 0, 0, 0, 0, 0], dtype=np.int64
    )
    args = [
        "owner",
        1,
        ("new", "old-a", "old-b"),
        packet,
        3,
        3,
        5,
        [2, 6, 8],
        [20, 40, 60],
        {"old-a": 1, "old-b": 0},
    ]
    proof = metadata_modules.vendor._cpu_sequence_proof(*args)
    assert type(proof) is host_metadata.CPUSequenceLengthsProof
    assert proof.query_starts == (0, 2, 6, 8)
    assert proof.computed == (0, 8, 16)
    assert proof.previous_positions == (-1, 1, 0)
    assert proof.previous_drafts == (-1, 0, 0)
    packet[:] = 999
    assert proof.computed == (0, 8, 16)
    assert host_metadata._cpu_sequence_proof(*args) is None


def test_internal_packet_conversion_failure_is_not_swallowed(host_metadata):
    with pytest.raises(ValueError):
        host_metadata._cpu_sequence_proof(
            "owner", 1, ("r",), np.array([-1, 0, 0]), 0, 1, 1, ["invalid"], [1], {}
        )


class Buffer:
    next_pointer = 10000

    def __init__(
        self, shape, dtype, device="cuda:0", values=None, storage=None, ptr=None
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = torch.device(device)
        self._storage = storage if storage is not None else SimpleNamespace()
        if ptr is None:
            Buffer.next_pointer += 1000
            ptr = Buffer.next_pointer
        self._pointer = ptr
        self.values = list(values) if values is not None else [0] * self.numel()

    def numel(self):
        return int(np.prod(self.shape))

    def dim(self):
        return len(self.shape)

    def stride(self, index=None):
        result = tuple(
            int(np.prod(self.shape[i + 1 :])) for i in range(len(self.shape))
        )
        return result if index is None else result[index]

    def data_ptr(self):
        return self._pointer

    def untyped_storage(self):
        return self._storage

    def __getitem__(self, key):
        assert isinstance(key, slice)
        begin, end, step = key.indices(self.numel())
        assert step == 1
        return Buffer(
            (end - begin,),
            self.dtype,
            self.device,
            self.values[begin:end],
            self._storage,
            self._pointer + begin * 4,
        )


class BlockTables:
    def __init__(self):
        self.block_tables = []


@pytest.fixture
def graph_host(monkeypatch, metadata_modules):
    module = metadata_modules.vendor
    host = metadata_modules.host
    package = metadata_modules.package
    state = SimpleNamespace(
        events=[], capture=None, capturing=False, fail=False, runner=None
    )

    class Graph:
        def __init__(self):
            self.operation = None

        def replay(self):
            state.events.append("replay")
            if state.fail:
                raise RuntimeError("producer failure")
            self.operation()

    @contextmanager
    def graph(value, pool):
        assert pool == "shared-pool"
        state.events.append("enter")
        state.capture = value
        state.capturing = True
        try:
            yield
        finally:
            state.capture = None
            state.capturing = False
            state.events.append("exit")

    def empty(shape, dtype, device):
        assert not state.capturing
        state.events.append("allocate")
        return Buffer(shape, dtype, device)

    module.torch = SimpleNamespace(
        int32=torch.int32,
        empty=empty,
        cuda=SimpleNamespace(
            CUDAGraph=Graph,
            graph=graph,
            is_current_stream_capturing=lambda: state.capturing,
        ),
    )
    platform = ModuleType("vllm.platforms")
    platform.current_platform = SimpleNamespace(
        get_global_graph_pool=lambda: "shared-pool"
    )
    monkeypatch.setitem(sys.modules, "vllm.platforms", platform)
    kernels = ModuleType(package.__name__ + ".metadata_kernels")

    def produce(
        table, extent, qsl, positions, lens, computed, request_ids, counts, *extra
    ):
        assert state.runner._request_id_current is None
        if state.fail:
            raise RuntimeError("producer failure")
        state.events.append("produce")

        def apply():
            request_ids.values[:] = [0] * request_ids.numel()
            for request in range(extent):
                begin, end = qsl.values[request : request + 2]
                counts.values[request] = end - begin
                request_ids.values[begin:end] = [request] * (end - begin)

        apply()
        if state.capture is not None:
            state.capture.operation = apply

    kernels._compute_step_common_with_request_ids = produce
    monkeypatch.setitem(sys.modules, kernels.__name__, kernels)
    runner = module.CommonAttentionMetadataGraphRunner()
    state.runner = runner
    table = BlockTables()
    qsl = Buffer((4,), torch.int32, values=[0, 2, 5, 6])
    positions = Buffer((8,), torch.int64)
    lens = Buffer((3,), torch.int32)
    computed = Buffer((3,), torch.int32)
    args = [table, 3, qsl, positions, lens, computed]

    def common(tokens=6):
        return SimpleNamespace(
            query_start_loc=qsl,
            query_start_loc_cpu=Buffer((4,), torch.int32, "cpu"),
            num_reqs=3,
            num_actual_tokens=tokens,
        )

    return SimpleNamespace(
        module=host,
        state=state,
        runner=runner,
        args=args,
        qsl=qsl,
        positions=positions,
        common=common,
    )


def test_fixed_eager_chain_publishes_successful_current_lease(graph_host):
    c = graph_host
    assert c.runner.run(*c.args, use_graph=False, capture=False) is False
    assert c.state.events == ["allocate", "allocate", "produce"]
    common = c.common()
    assert c.runner.attach_req_id_per_token(common)
    assert type(common._model_request_id_lease) is c.module._RequestIdMapLease
    assert c.module.request_id_per_token_from_common(common).values == [
        0,
        0,
        1,
        1,
        1,
        2,
    ]
    assert c.module.decode_lens_from_common(common, 3).values == [2, 3, 1]
    assert c.module.request_id_per_token_from_common(copy.copy(common)).values == [
        0,
        0,
        1,
        1,
        1,
        2,
    ]


def test_capture_warm_replay_and_shrinking_inputs(graph_host):
    c = graph_host
    assert c.runner.run(*c.args, use_graph=True, capture=True)
    assert c.state.events == [
        "allocate",
        "allocate",
        "produce",
        "enter",
        "produce",
        "exit",
        "replay",
    ]
    old = c.common()
    assert c.runner.attach_req_id_per_token(old)
    output = c.runner._request_id_map
    c.qsl.values[:] = [0, 1, 1, 3]
    c.state.events.clear()
    assert c.runner.run(*c.args, use_graph=True, capture=False)
    assert c.state.events == ["replay"]
    assert c.runner._request_id_map is output
    assert output.values == [0, 2, 2, 0, 0, 0, 0, 0]
    assert c.runner._request_id_counts.values == [1, 0, 2]
    assert c.module.request_id_per_token_from_common(old) is None
    new = c.common(3)
    assert c.runner.attach_req_id_per_token(new)
    assert c.module.request_id_per_token_from_common(new).values == [0, 2, 2]


@pytest.mark.parametrize(
    "change", ["input_pointer", "block_table_owner", "group_binding"]
)
def test_new_graph_binding_never_reuses_old_capture(graph_host, change):
    c = graph_host
    c.runner.run(*c.args, use_graph=True, capture=True)
    c.state.events.clear()
    if change == "input_pointer":
        c.args[2] = Buffer((4,), torch.int32, values=c.qsl.values)
    elif change == "block_table_owner":
        c.args[0] = SimpleNamespace(block_tables=[])
    else:
        g = SimpleNamespace(
            block_table=SimpleNamespace(gpu=Buffer((3, 2), torch.int32)),
            slot_mapping=SimpleNamespace(gpu=Buffer((8,), torch.int64)),
            block_size=16,
            kv_cache_block_size=16,
            blocks_per_kv_block=1,
            max_num_batched_tokens=8,
            dcp_world_size=1,
            dcp_rank=0,
            cp_kv_cache_interleave_size=1,
            slot_mapping_mode=SimpleNamespace(value=0),
        )
        c.args[0].block_tables = [g]
    assert not c.runner.run(*c.args, use_graph=True, capture=False)
    assert c.state.events == ["produce"]
    assert len(c.runner.graphs) == 1


@pytest.mark.parametrize(
    "change", ["qsl_object", "qsl_cpu_object", "num_reqs", "num_tokens"]
)
def test_lease_rejects_changed_consumer_identity_or_extent(graph_host, change):
    c = graph_host
    c.runner.run(*c.args, use_graph=False, capture=False)
    common = c.common()
    assert c.runner.attach_req_id_per_token(common)
    if change == "qsl_object":
        common.query_start_loc = c.qsl[:]
    elif change == "qsl_cpu_object":
        common.query_start_loc_cpu = common.query_start_loc_cpu[:]
    elif change == "num_reqs":
        common.num_reqs -= 1
    else:
        common.num_actual_tokens -= 1
    assert c.module.request_id_per_token_from_common(common) is None
    assert c.module.decode_lens_from_common(common, 1) is None


@pytest.mark.parametrize("captured", [False, True])
def test_failed_submission_expires_lease_and_has_no_retry(graph_host, captured):
    c = graph_host
    c.runner.run(*c.args, use_graph=captured, capture=captured)
    common = c.common()
    assert c.runner.attach_req_id_per_token(common)
    c.state.fail = True
    c.state.events.clear()
    with pytest.raises(RuntimeError, match="producer failure"):
        c.runner.run(*c.args, use_graph=captured, capture=False)
    assert c.runner._request_id_current is None
    assert c.module.request_id_per_token_from_common(common) is None
    assert c.state.events == (["replay"] if captured else [])
    c.runner.clear()
    c.state.events.clear()
    with pytest.raises(RuntimeError, match="producer failure"):
        c.runner.run(*c.args, use_graph=True, capture=True)
    assert c.runner._request_id_current is None
    assert c.runner.graphs == {}
    assert c.state.events == ["allocate", "allocate"]


def test_capture_cannot_allocate_new_addresses(graph_host):
    c = graph_host
    c.state.capturing = True
    with pytest.raises(RuntimeError, match="allocated before capture"):
        c.runner.run(*c.args, use_graph=True, capture=True)
    assert c.state.events == [] and c.runner._request_id_current is None
    c.runner._request_id_map = Buffer((8,), torch.int32)
    with pytest.raises(RuntimeError, match="request counts must be allocated"):
        c.runner.run(*c.args, use_graph=True, capture=True)
    assert c.state.events == [] and c.runner._request_id_current is None


def test_clear_drops_graph_owners_and_expires_lease(graph_host):
    c = graph_host
    c.runner.run(*c.args, use_graph=True, capture=True)
    common = c.common()
    assert c.runner.attach_req_id_per_token(common)
    owner = weakref.ref(c.args[0])
    c.args[0] = None
    gc.collect()
    assert owner() is not None
    c.runner.clear()
    gc.collect()
    assert owner() is None
    assert not c.runner.graphs and c.runner._request_id_map is None
    assert c.module.decode_lens_from_common(common, 3) is None


def test_input_event_owner_keeps_plan_until_next_begin(metadata_modules):
    vendor = metadata_modules.vendor
    s = vendor.AsyncInputStaging.__new__(vendor.AsyncInputStaging)
    s.plan = SimpleNamespace(sampled=object(), draft=object())
    s.in_step = True
    s._cpu_proof_generation = 7
    plan = s.plan
    s.end_step()
    assert s.plan is plan and not s.in_step
    s.begin_step()
    assert s.plan is None and s.in_step and s._cpu_proof_generation == 8
    with pytest.raises(RuntimeError, match="overlapping"):
        s.begin_step()


def test_decode_deltas_follow_current_order_and_prompt_boundary(host_metadata):
    requests = {
        r: SimpleNamespace(
            mrope_positions=object(),
            mrope_position_delta=d,
            prompt_token_ids=[0] * 5,
            prompt_embeds=None,
        )
        for r, d in [("a", 4), ("b", 7)]
    }
    length = lambda ids, embeds: len(ids)
    assert host_metadata.select_decode_deltas(
        ["b", "a"], requests, {"a": 2, "b": 1}, [8, 9], length, uses_mrope=True
    ) == (7, 4)
    assert (
        host_metadata.select_decode_deltas(
            ["b", "a"], requests, {"a": 2, "b": 1}, [4, 9], length, uses_mrope=True
        )
        is None
    )
