# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU graph-selection contracts, without model or CUDA execution."""

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

PLUGIN = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4"


def source_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, PLUGIN / relative)
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def graph(monkeypatch):
    package = ModuleType("hy4_graph_contract")
    package.__path__ = [str(PLUGIN / "nvidia")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    return source_module(
        "hy4_graph_contract.mtp_draft_graph", "nvidia/mtp_draft_graph.py"
    )


def common(q=4, reqs=2):
    return SimpleNamespace(
        max_query_len=q,
        num_reqs=reqs,
        num_actual_tokens=q * reqs,
        query_start_loc_cpu=torch.arange(reqs + 1) * q,
        is_prefilling=torch.zeros(reqs, dtype=torch.bool),
        causal=True,
        mm_req_doc_ranges=None,
    )


def test_draft_plan_accepts_verified_batch(graph):
    assert graph.decode_plan(
        common(), (1, 2, 4, 8, 16, 32, 64, 128, 256), verified_step=True
    ) == (4, 8, 2)


@pytest.mark.parametrize(
    "condition",
    ["unverified", "nonuniform", "prefill", "noncausal", "media", "capacity"],
)
def test_draft_plan_declines_before_running_model(graph, condition):
    c = common()
    verified = True
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    if condition == "unverified":
        verified = False
    elif condition == "nonuniform":
        c.query_start_loc_cpu[1] = 3
    elif condition == "prefill":
        c.is_prefilling[1] = True
    elif condition == "noncausal":
        c.causal = False
    elif condition == "media":
        c.mm_req_doc_ranges = {0: [(0, 2)]}
    elif condition == "capacity":
        sizes = (1, 2, 4)
    assert graph.decode_plan(c, sizes, verified_step=verified) is None


def test_q1_capacity_is_distinct_from_q4_rows(graph):
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    assert graph.decode_plan(common(1, 64), sizes, verified_step=True) == (1, 64, 64)
    assert graph.decode_plan(common(1, 65), sizes, verified_step=True) is None


@pytest.fixture
def factory(monkeypatch):
    package = ModuleType("hy4_factory_contract")
    package.__path__ = [str(PLUGIN)]
    nvidia = ModuleType("hy4_factory_contract.nvidia")
    nvidia.__path__ = [str(PLUGIN / "nvidia")]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, nvidia.__name__, nvidia)

    class Mode(Enum):
        FULL_AND_PIECEWISE = 1
        NONE = 2

    platform = SimpleNamespace(
        is_cuda=lambda: True, get_device_capability=lambda: (9, 0)
    )
    native_config = ModuleType("vllm.config")
    native_config.CUDAGraphMode = Mode
    platforms = ModuleType("vllm.platforms")
    platforms.current_platform = platform
    policy = ModuleType("hy4_factory_contract.indexer_policy")
    policy.use_int8_indexer = lambda: True
    graph = ModuleType("hy4_factory_contract.nvidia.prefill_graph")
    validations = []
    graph.supports_mixed_config = lambda c: validations.append(c) or True
    for name, module in [
        ("vllm.config", native_config),
        ("vllm.platforms", platforms),
        (policy.__name__, policy),
        (graph.__name__, graph),
    ]:
        monkeypatch.setitem(sys.modules, name, module)
    module = source_module(
        "hy4_factory_contract.nvidia.runtime_hooks", "nvidia/runtime_hooks.py"
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=16,
            enable_expert_parallel=True,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            decode_context_parallel_size=1,
            prefill_context_parallel_size=1,
            use_ubatching=False,
        ),
        speculative_config=SimpleNamespace(
            method="mtp",
            num_speculative_tokens=3,
            enforce_eager=False,
            parallel_drafting=False,
            disable_padded_drafter_batch=False,
        ),
        compilation_config=SimpleNamespace(cudagraph_mode=Mode.FULL_AND_PIECEWISE),
        lora_config=None,
    )
    return module, config, platform, policy, Mode, validations


def test_runtime_factory_creates_no_buffers(factory):
    module, config, _, _, _, validations = factory
    assert isinstance(module.create_hy4_runtime(config), module.Hy4RuntimeHooks)
    assert validations == [config]


@pytest.mark.parametrize(
    "condition",
    [
        "noncuda",
        "arch",
        "native_indexer",
        "draft_count",
        "tp",
        "lora",
        "mode",
        "unsupported_graph",
    ],
)
def test_runtime_factory_retains_native_before_allocation(factory, condition):
    module, config, platform, policy, mode, validations = factory
    if condition == "unsupported_graph":
        sys.modules[
            "hy4_factory_contract.nvidia.prefill_graph"
        ].supports_mixed_config = lambda c: False
    elif condition == "noncuda":
        platform.is_cuda = lambda: False
    elif condition == "arch":
        platform.get_device_capability = lambda: (8, 0)
    elif condition == "native_indexer":
        policy.use_int8_indexer = lambda: False
    elif condition == "draft_count":
        config.speculative_config.num_speculative_tokens = 2
    elif condition == "tp":
        config.parallel_config.tensor_parallel_size = 8
    elif condition == "lora":
        config.lora_config = object()
    elif condition == "mode":
        config.compilation_config.cudagraph_mode = mode.NONE
    assert module.create_hy4_runtime(config) is None
    assert validations == []


@pytest.mark.parametrize(
    "missing_name,optional",
    [("vllm.v1.worker.model_runtime", True), ("broken_dependency", False)],
)
def test_optional_extension_missing_vs_broken_import(
    monkeypatch, missing_name, optional
):
    import builtins

    module = source_module("hy4_optional_extension_contract", "nvidia/runtime_hooks.py")
    original_import = builtins.__import__

    def intercepted(name, *args, **kwargs):
        if name == "vllm.v1.worker.model_runtime":
            raise ModuleNotFoundError("unavailable", name=missing_name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", intercepted)
    if optional:
        assert module.register_runtime_hooks() is False
    else:
        with pytest.raises(ModuleNotFoundError) as raised:
            module.register_runtime_hooks()
        assert raised.value.name == missing_name


def test_optional_extension_does_not_hide_missing_public_api(monkeypatch):
    module = source_module("hy4_broken_api_contract", "nvidia/runtime_hooks.py")
    extension = ModuleType("vllm.v1.worker.model_runtime")
    monkeypatch.setitem(sys.modules, extension.__name__, extension)
    with pytest.raises(ImportError):
        module.register_runtime_hooks()


@pytest.fixture
def staging(monkeypatch):
    package = ModuleType("hy4_staging_contract")
    package.__path__ = [str(PLUGIN / "nvidia")]
    kernel = ModuleType("hy4_staging_contract.runtime_input_kernels")

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU metadata contract must not launch a kernel")

    kernel.async_input_correct = kernel.async_input_gather = forbidden
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setitem(sys.modules, kernel.__name__, kernel)
    return source_module(
        "hy4_staging_contract.runtime_staging", "nvidia/runtime_staging.py"
    )


def test_input_maps_follow_reordered_request_history(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[-1, 1, 0]),
        input_batch=SimpleNamespace(req_ids=["new", "old-a", "old-b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={"old-a": [123]})
    maps = staging.build_input_maps(runner, scheduled, 3, 8, [2, 6, 8])
    assert maps.sample_dst == [4, 7]
    assert maps.sample_src == [1, 0]
    assert maps.draft_dst == [5]
    assert maps.draft_src == [3]
    assert maps.num_sampled == 2
    assert maps.copy_base is True


def test_unchanged_dense_sample_map_stays_native(staging):
    runner = SimpleNamespace(
        prev_positions=SimpleNamespace(np=[0, 1]),
        input_batch=SimpleNamespace(req_ids=["a", "b"]),
        prev_num_spec_tokens=3,
    )
    scheduled = SimpleNamespace(scheduled_spec_decode_tokens={})
    assert staging.build_input_maps(runner, scheduled, 2, 2, [1, 2]) is None


def test_host_packet_layout_and_untouched_tail(staging):
    import numpy as np

    packet = np.full(32, 99, dtype=np.int64)
    used = staging.pack_host_packet(
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


def test_packet_proof_copies_only_the_transferred_facts(staging):
    from dataclasses import dataclass

    import numpy as np

    @dataclass(frozen=True)
    class Proof:
        issuer: str
        generation: int
        request_ids: tuple
        query_starts: tuple
        computed: tuple
        prompt_lengths: tuple
        previous_positions: tuple
        previous_drafts: tuple

    packet = np.array(
        [4, 7, 5, 1, 0, 3, -1, 1, 0, 0, 8, 16, 0, 0, 0, 0, 0], dtype=np.int64
    )
    ids = ("new", "old-a", "old-b")
    args = [
        Proof,
        "owner",
        1,
        ids,
        packet,
        3,
        3,
        5,
        [2, 6, 8],
        [20, 40, 60],
        {"old-a": 1, "old-b": 0},
    ]
    result = staging._cpu_sequence_proof(*args)
    assert result.query_starts == (0, 2, 6, 8)
    assert result.computed == (0, 8, 16)
    assert result.previous_positions == (-1, 1, 0)
    assert result.previous_drafts == (-1, 0, 0)
    packet[:] = 999
    assert result.computed == (0, 8, 16)
    args[2] = 0
    assert staging._cpu_sequence_proof(*args) is None


@pytest.mark.parametrize("failure", [None, "launch", "synchronize"])
def test_full_metadata_publishes_warm_capability_only_after_success(
    monkeypatch, failure
):
    from contextlib import nullcontext

    module = source_module("hy4_full_metadata_contract", "nvidia/decode_metadata.py")
    device = "cuda:0"
    counts = SimpleNamespace(
        is_cuda=True, device=device, shape=(2,), dtype="int32", stride=lambda: (1,)
    )
    seq = SimpleNamespace(
        is_cuda=True, device=device, shape=(2, 4), dtype="int32", stride=lambda: (4, 1)
    )

    class Metadata:
        num_prefills = 0
        num_decode_tokens = 8
        decode = SimpleNamespace(decode_lens=counts, seq_lens=seq)

    imported = ModuleType("vllm.v1.attention.backends.mla.indexer")
    imported.DeepseekV32IndexerMetadata = Metadata
    monkeypatch.setitem(sys.modules, imported.__name__, imported)
    stream = SimpleNamespace(wait_stream=lambda previous: None)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "Stream", lambda device: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    calls = []

    def launch(*args):
        calls.append("launch")
        assert not module.full_decode_metadata_available(counts, seq, 8, device)
        if failure == "launch":
            raise RuntimeError("launch failed")

    def synchronize(device):
        calls.append("synchronize")
        assert not module.full_decode_metadata_available(counts, seq, 8, device)
        if failure == "synchronize":
            raise RuntimeError("synchronize failed")

    monkeypatch.setattr(module, "prepare_full_decode_metadata", launch)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    assert not module.full_decode_metadata_available(counts, seq, 8, device)
    if failure is None:
        assert (
            module.warm_full_decode_metadata_before_capture({"one": Metadata()}, device)
            == 1
        )
        assert module.full_decode_metadata_available(counts, seq, 8, device)
        assert not module.full_decode_metadata_available(counts, seq, 4, device)
    else:
        with pytest.raises(RuntimeError, match="failed"):
            module.warm_full_decode_metadata_before_capture({"one": Metadata()}, device)
        assert not module.full_decode_metadata_available(counts, seq, 8, device)
    assert calls == ["launch", "synchronize"]
