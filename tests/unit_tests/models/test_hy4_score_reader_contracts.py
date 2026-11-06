# SPDX-License-Identifier: Apache-2.0
"""Isolated CPU host contracts, not a complete vLLM 0.28 or GPU qualification.

Selected definitions execute directly from production source. Only their host
interfaces are supplied here, avoiding unrelated 0.28 package/GPU imports on the
CPU executor. Native partition/decode checks also call installed stock helpers.
"""

import ast
import dataclasses
import importlib.util
import inspect
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch.nn.modules import module as nn_module

from vllm.config import CUDAGraphMode

MODEL = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4/nvidia"


def source_load(monkeypatch, name, filename, names, **bindings):
    """Load unchanged host definitions into an explicit isolated test namespace."""
    path = MODEL / filename
    tree = ast.parse(path.read_text(), filename=str(path))
    body = [
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
    ]
    found = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            selected = {node.name} & set(names)
        elif isinstance(node, ast.Assign):
            selected = {
                target.id for target in node.targets if isinstance(target, ast.Name)
            } & set(names)
        else:
            selected = set()
        if selected:
            found.update(selected)
            body.append(node)
    assert found == set(names), f"missing production definitions: {set(names) - found}"
    loaded = module(monkeypatch, name, **bindings)
    loaded.__file__ = str(path)
    loaded.__package__ = name.rpartition(".")[0]
    code = ast.fix_missing_locations(ast.Module(body=body, type_ignores=[]))
    exec(compile(code, str(path), "exec"), vars(loaded))
    return loaded


class TensorMetadata:
    def __init__(self, shape, dtype, stride=None, device="cuda:0"):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda = len(shape), device.startswith("cuda")
        if stride is None:
            stride, size = [], 1
            for dim in reversed(shape):
                stride.insert(0, size)
                size *= dim
        self.strides = tuple(stride)

    def stride(self, dim=None):
        return self.strides if dim is None else self.strides[dim]

    def item(self):
        raise AssertionError("selection must not read a device scalar")

    def cpu(self):
        raise AssertionError("selection must not copy device values")

    def tolist(self):
        raise AssertionError("selection must not read device values")


@dataclasses.dataclass(frozen=True)
class StepProof:
    generation: int
    query_starts: tuple
    computed: tuple
    prompt_lengths: tuple
    previous_positions: tuple
    previous_drafts: tuple


@dataclasses.dataclass(frozen=True)
class ChunkProof:
    step_proof: StepProof
    request_start: int
    request_stop: int
    chunk_start: int
    chunk_end: int
    seq_lens: tuple


class Metadata(SimpleNamespace):
    pass


class Chunk(SimpleNamespace):
    pass


class MixedFixture:
    def __init__(self, queries=(1796,), bounds=(4096,), decodes=63):
        starts = [0]
        for q in [4] * decodes + list(queries):
            starts.append(starts[-1] + q)
        nreq = decodes + len(queries)
        self.step = StepProof(
            7,
            tuple(starts),
            tuple([4092] * decodes + [b - q for b, q in zip(bounds, queries)]),
            tuple([4092] * decodes + [b + 64 for b in bounds]),
            (-1,) * nreq,
            (-1,) * nreq,
        )
        m, n = sum(queries), len(queries)
        self.table = TensorMetadata((n, 3200), torch.int32)
        self.chunk = Chunk(
            num_reqs=n,
            total_seq_lens=sum(bounds),
            local_total_seq_lens=sum(bounds),
            token_start=starts[decodes],
            token_end=starts[-1],
            block_table=self.table,
            local_cu_seq_lens=TensorMetadata((n + 1,), torch.int32),
            cu_seqlen_ks=TensorMetadata((m,), torch.int32),
            cu_seqlen_ke=TensorMetadata((m,), torch.int32),
        )
        self.chunk.cpu_seq_len_bound = ChunkProof(
            self.step, decodes, nreq, starts[decodes], starts[-1], tuple(bounds)
        )
        self.meta = Metadata(
            num_decodes=decodes,
            num_decode_tokens=4 * decodes,
            num_prefills=n,
            num_prefill_tokens=m,
            prefill=SimpleNamespace(chunks=[self.chunk], use_dense_mha=False),
            decode=SimpleNamespace(
                block_table=TensorMetadata((decodes, 3200), torch.int32)
            ),
            cpu_seq_lens_proof_generation=7,
        )
        self.context = SimpleNamespace(
            cudagraph_runtime_mode=CUDAGraphMode.PIECEWISE,
            ubatch_slices=None,
            dp_metadata=None,
            batch_descriptor=SimpleNamespace(
                num_tokens=starts[-1], has_lora=False, num_active_loras=0
            ),
            attn_metadata={"cache": self.meta},
        )
        self.indexer = SimpleNamespace(
            _hy4_int8_empty_tile_enabled=True,
            _hy4_int8_prefix_topk_enabled=True,
            cache=SimpleNamespace(
                prefix="cache", kv_cache=TensorMetadata((4096, 64, 132), torch.uint8)
            ),
            topk=2048,
        )
        self.args = [
            self.indexer,
            self.context,
            TensorMetadata((m, 32, 128), torch.int8),
            TensorMetadata((m, 32), torch.float32),
            self.table,
            TensorMetadata((m,), torch.int64),
            TensorMetadata((m,), torch.int32),
            TensorMetadata((m, 2048), torch.int32),
            204800,
            64,
        ]

    def select(self, selector, scoped=True):
        selector.get_current_cpu_seq_lens_proof = lambda: self.step
        with torch.inference_mode():
            if scoped:
                with selector.score_width_attention_scope(self.context):
                    return selector.select_mixed_score_capacity(*self.args)
            return selector.select_mixed_score_capacity(*self.args)


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    policy = source_load(
        monkeypatch,
        "hy4_score_policy_contract",
        "../indexer_policy.py",
        {"_score_capacity"},
    )
    return source_load(
        monkeypatch,
        "hy4_score_contract",
        "indexer.py",
        {
            "_ATTENTION_CONTEXT",
            "score_width_attention_scope",
            "select_eager_score_capacity",
            "select_mixed_score_capacity",
            "_score_context_supported",
            "_score_layout_supported",
            "HYV4Int8Indexer",
        },
        torch=torch,
        nn=torch.nn,
        ContextVar=ContextVar,
        contextmanager=contextmanager,
        _score_capacity=policy._score_capacity,
        CUDAGraphMode=CUDAGraphMode,
        DeepseekV32IndexerMetadata=Metadata,
        DeepseekV32IndexerPrefillChunkMetadata=Chunk,
        PrefillChunkSequenceBounds=ChunkProof,
        get_current_cpu_seq_lens_proof=lambda: None,
    )


@pytest.mark.parametrize(
    "bound,capacity",
    [(63, 16384), (16384, 16384), (16385, 32768), (32768, 32768), (32769, None)],
)
def test_score_capacity_edges(selector, bound, capacity):
    f = MixedFixture((16,), (bound,))
    assert f.select(selector) == capacity


@pytest.mark.parametrize(
    "queries", [(63,), (1796,), (1800,), (1956,), (25, 26), (25, 26, 27, 28, 29)]
)
def test_score_accepts_proven_query_and_request_counts(selector, queries):
    f = MixedFixture(queries, tuple(12000 + i for i in range(len(queries))))
    assert f.select(selector) == 16384


def test_score_capacity_uses_request_maximum(selector):
    f = MixedFixture((898, 898), (12000, 13000))
    assert f.chunk.total_seq_lens == 25000
    assert f.select(selector) == 16384


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f: setattr(f, "step", dataclasses.replace(f.step)),
        lambda f: setattr(f.meta, "cpu_seq_lens_proof_generation", 8),
        lambda f: setattr(f.chunk, "token_end", f.chunk.token_end - 1),
        lambda f: setattr(f.chunk, "local_total_seq_lens", 1),
        lambda f: setattr(f.chunk, "cpu_seq_len_bound", None),
        lambda f: setattr(f.args[5], "strides", (2,)),
        lambda f: setattr(f.args[7], "dtype", torch.float32),
        lambda f: setattr(f.indexer, "topk", 4096),
        lambda f: setattr(f.indexer, "_hy4_int8_empty_tile_enabled", False),
        lambda f: setattr(f.context, "cudagraph_runtime_mode", CUDAGraphMode.FULL),
    ],
)
def test_score_single_condition_rejects(selector, mutation):
    f = MixedFixture()
    assert f.select(selector) == 16384
    mutation(f)
    assert f.select(selector) is None


def test_score_scope_and_capture_reject(selector, monkeypatch):
    f = MixedFixture()
    assert f.select(selector) == 16384
    assert f.select(selector, scoped=False) is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert f.select(selector) is None


def module(monkeypatch, name, **attrs):
    value = ModuleType(name)
    vars(value).update(attrs)
    monkeypatch.setitem(sys.modules, name, value)
    return value


@pytest.fixture
def reader(monkeypatch):
    class FP8Separate:
        pass

    class FP8Kernel:
        pass

    metadata = SimpleNamespace(
        FP8SeparatePrefillDecode=FP8Separate, FP8KernelMetadata=FP8Kernel
    )
    package = module(monkeypatch, "hy4_reader_fixture")
    package.__path__ = [str(MODEL.parent)]
    module(monkeypatch, package.__name__ + ".nvidia").__path__ = [str(MODEL)]
    implementation = source_load(
        monkeypatch,
        "hy4_reader_fixture.nvidia.attention",
        "attention.py",
        {
            "HYV4FlashMLASparseImpl",
            "HYV4FlashMLASparseMetadataBuilder",
            "HYV4FlashMLASparseBackend",
        },
        torch=torch,
        FlashMLASparseBackend=type("Backend", (), {}),
        FlashMLASparseImpl=type("Impl", (), {}),
        FlashMLASparseMetadataBuilder=type("Builder", (), {}),
        FlashMLASparseMetadata=metadata,
        CPUSequenceLengthsProof=StepProof,
        get_current_cpu_seq_lens_proof=lambda: None,
        use_flaggems_op=lambda *args, **kwargs: True,
        current_platform=SimpleNamespace(is_device_capability=lambda value: True),
        logger=SimpleNamespace(warning_once=lambda message: None),
        flash_mla_sparse_fwd=None,
        flash_mla_with_kvcache=None,
        current_workspace_manager=lambda: None,
    )
    return implementation, implementation, metadata


def reader_fixture(reader, bounds=(12000, 13000), queries=(898, 898)):
    implementation, backend, metadata = reader
    f = MixedFixture(queries, bounds)
    backend.get_current_cpu_seq_lens_proof = lambda: f.step
    ndecode = f.meta.num_decodes
    common = SimpleNamespace(
        num_reqs=ndecode + len(queries),
        num_actual_tokens=f.step.query_starts[-1],
        causal=True,
        mm_req_doc_ranges=None,
        cpu_seq_lens_proof=f.step,
        query_start_loc_cpu=torch.tensor(f.step.query_starts),
        seq_lens_cpu_upper_bound=torch.tensor([4096] * ndecode + list(bounds)),
        is_prefilling=torch.tensor([False] * ndecode + [True] * len(queries)),
    )
    f.meta.fp8_use_mixed_batch = True
    f.meta.fp8_extra_metadata = metadata.FP8KernelMetadata()
    owner = object.__new__(implementation.HYV4FlashMLASparseMetadataBuilder)
    owner._fp8_bf16_prefill_supported = True
    owner._fp8_bf16_prefill_query_capacity = 2048
    owner._fp8_bf16_prefill_workspace_capacity = 25000
    owner.model_config = SimpleNamespace(max_model_len=204800)
    return f, owner, common


def test_reader_gather_uses_sum_and_real_capacity(reader):
    f, owner, common = reader_fixture(reader)
    assert owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)
    owner._fp8_bf16_prefill_workspace_capacity = 24999
    assert not owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda f, owner, common: setattr(
            owner, "_fp8_bf16_prefill_query_capacity", common.num_actual_tokens - 1
        ),
        lambda f, owner, common: setattr(
            common, "cpu_seq_lens_proof", dataclasses.replace(f.step)
        ),
        lambda f, owner, common: setattr(common, "causal", False),
        lambda f, owner, common: setattr(common, "mm_req_doc_ranges", {}),
        lambda f, owner, common: setattr(
            f.meta, "num_decode_tokens", f.meta.num_decode_tokens - 1
        ),
    ],
)
def test_reader_single_condition_rejects(reader, mutation):
    f, owner, common = reader_fixture(reader)
    assert owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)
    mutation(f, owner, common)
    assert not owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)


def test_reader_operator_policy_can_disable_after_construction(reader, monkeypatch):
    implementation, _, _ = reader
    f, owner, common = reader_fixture(reader)
    assert owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)
    monkeypatch.setattr(
        implementation, "use_flaggems_op", lambda *args, **kwargs: False
    )
    assert not owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)


@pytest.fixture
def score_consumer(monkeypatch, selector):
    policy, calls = {}, []
    context = SimpleNamespace(attn_metadata={})
    fgv = SimpleNamespace(
        int8_indexer_cache_store=lambda *args: calls.append("store"),
        int8_indexer_paged_logits=lambda *args, **kwargs: calls.append("base"),
    )
    module(
        monkeypatch,
        "flag_gems",
        topk=lambda *args, **kwargs: calls.append("native_topk"),
    )
    selector._hy4_fgv = fgv
    selector.int8_indexer_cache_store = fgv.int8_indexer_cache_store
    selector.use_flaggems_op = lambda op, default=True: policy.get(op, default)
    selector.get_forward_context = lambda: context
    selector.current_platform = SimpleNamespace(
        is_cuda=lambda: True, is_device_capability=lambda capability: capability == 90
    )
    parallel = SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1
    )
    owner = selector.HYV4Int8Indexer(
        SimpleNamespace(prefix="cache", kv_cache=torch.empty(1, 64, 132)),
        2048,
        204800,
        torch.full((2, 2048), -1, dtype=torch.int32),
        parallel,
    )
    return owner, fgv, policy, calls, selector


def test_score_consumer_forwards_capacity_and_writes_same_output(
    score_consumer, monkeypatch
):
    owner, fgv, _, calls, selector = score_consumer
    monkeypatch.setattr(selector, "select_eager_score_capacity", lambda *args: None)
    monkeypatch.setattr(selector, "select_mixed_score_capacity", lambda *args: 32768)

    def scores(*args, **kwargs):
        calls.append(("scores", args[6], kwargs))
        return torch.zeros(2, 32768)

    def prefix(logits, lengths, count, output):
        calls.append(("prefix", count))
        output.fill_(17)

    fgv.int8_indexer_paged_logits_empty_tile = scores
    owner._hy4_int8_paged_logits = scores
    owner._hy4_int8_prefix_topk_enabled = True
    owner._hy4_supports_int8_prefix_topk = lambda *args: True
    owner._hy4_int8_prefix_topk = prefix
    output = torch.full((2, 2048), -1, dtype=torch.int32)
    owner._score(
        torch.zeros(2, 32, 128, dtype=torch.int8),
        torch.zeros(2, 32),
        torch.zeros(1, 3200, dtype=torch.int32),
        torch.zeros(2, dtype=torch.int64),
        torch.ones(2, dtype=torch.int32),
        output,
    )
    assert calls == [("scores", 32768, {}), ("prefix", 2048)]
    assert torch.equal(output, torch.full_like(output, 17))


def test_score_execution_failure_does_not_retry(score_consumer, monkeypatch):
    owner, fgv, _, calls, selector = score_consumer
    monkeypatch.setattr(selector, "select_eager_score_capacity", lambda *args: None)
    monkeypatch.setattr(selector, "select_mixed_score_capacity", lambda *args: None)

    def fail(*args, **kwargs):
        calls.append("selected_launch")
        raise RuntimeError("selected execution failed")

    fgv.int8_indexer_paged_logits = fail
    owner._configure_providers()
    with pytest.raises(RuntimeError, match="selected execution failed"):
        owner._score(
            torch.zeros(2, 32, 128, dtype=torch.int8),
            torch.zeros(2, 32),
            torch.zeros(1, 3200, dtype=torch.int32),
            torch.zeros(2, dtype=torch.int64),
            torch.ones(2, dtype=torch.int32),
            owner.indices,
        )
    assert calls == ["selected_launch"]


def test_missing_selected_api_is_rejected_before_cache_write(score_consumer):
    owner, _, policy, calls, _ = score_consumer
    context = owner.forward.__globals__["get_forward_context"]()
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    context.attn_metadata = {
        "cache": SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=2,
            slot_mapping=torch.tensor([18, 19]),
            decode=SimpleNamespace(
                decode_lens=torch.tensor([2]),
                seq_lens=torch.tensor([20]),
                block_table=torch.zeros(1, 3200, dtype=torch.int32),
            ),
        )
    }
    owner._score = lambda *args: calls.append("score")
    args = (
        torch.zeros(2, 16),
        torch.zeros(2, 32, 128, dtype=torch.int8),
        torch.zeros(2, 128),
        torch.zeros(2, 32),
    )
    owner.forward(*args)
    assert calls == ["store", "score"]
    calls.clear()
    owner.indices.fill_(99)
    policy["int8_indexer_paged_logits_empty_tile"] = True
    with pytest.raises(AttributeError, match="int8_indexer_paged_logits_empty_tile"):
        owner.forward(*args)
    assert calls == []
    assert torch.equal(owner.indices, torch.full_like(owner.indices, 99))


def test_provider_policy_enable_and_disable_use_public_exports(score_consumer):
    owner, fgv, policy, calls, _ = score_consumer
    fgv.int8_indexer_paged_logits_empty_tile = lambda *args: None
    fgv.indexer_prefix_topk = lambda *args: None
    fgv.supports_indexer_prefix_topk = lambda *args: True
    assert not owner._hy4_int8_empty_tile_enabled
    assert not owner._hy4_int8_prefix_topk_enabled
    policy.update(int8_indexer_paged_logits_empty_tile=True, indexer_prefix_topk=True)
    owner._configure_providers()
    assert owner._hy4_int8_empty_tile_enabled and owner._hy4_int8_prefix_topk_enabled
    assert owner._hy4_int8_paged_logits is fgv.int8_indexer_paged_logits_empty_tile
    assert owner._hy4_int8_prefix_topk is fgv.indexer_prefix_topk
    assert owner._hy4_supports_int8_prefix_topk is fgv.supports_indexer_prefix_topk
    policy["int8_indexer_paged_logits_empty_tile"] = False
    owner._configure_providers()
    assert not owner._hy4_int8_empty_tile_enabled
    assert owner._hy4_int8_paged_logits is fgv.int8_indexer_paged_logits
    assert calls == []


def test_non_nvidia_policy_does_not_select_optional_provider(
    score_consumer, monkeypatch
):
    owner, fgv, policy, calls, implementation = score_consumer
    policy.update(int8_indexer_paged_logits_empty_tile=True, indexer_prefix_topk=True)
    monkeypatch.setattr(
        implementation,
        "current_platform",
        SimpleNamespace(
            is_cuda=lambda: False, is_device_capability=lambda capability: True
        ),
    )
    owner._configure_providers()
    assert not owner._hy4_int8_empty_tile_enabled
    assert not owner._hy4_int8_prefix_topk_enabled
    assert owner._hy4_int8_paged_logits is fgv.int8_indexer_paged_logits
    assert calls == []


@pytest.fixture
def gate_overlap(monkeypatch):
    import contextlib

    events, policy = [], {"mul_scaled_int8_quant": True}

    class Stream:
        def __init__(self, name):
            self.name = name

        def wait_stream(self, other):
            events.append((self.name, "wait", other.name))

    class Tensor:
        is_cuda = True
        dtype = torch.bfloat16
        device = "cuda:0"

        def record_stream(self, stream):
            events.append(("tensor", "record", stream.name))

    parent, auxiliary = Stream("parent"), Stream("auxiliary")
    context = SimpleNamespace(cudagraph_runtime_mode=CUDAGraphMode.FULL)
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: auxiliary)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: parent)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch, "sigmoid", lambda value: value)
    implementation = source_load(
        monkeypatch,
        "hy4_gate_overlap_contract",
        "fusions.py",
        {"_has_module_hooks", "GateOverlap", "GateTicket"},
        torch=torch,
        nn_module=nn_module,
        CUDAGraphMode=CUDAGraphMode,
        BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: None),
        get_forward_context=lambda: context,
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )

    class Gate(torch.nn.Module):
        gather_output = False

        def __init__(self):
            super().__init__()
            self.weight = Tensor()

        def forward(self, hidden):
            events.append("projection")
            return (Tensor(),)

    owner = SimpleNamespace(
        linear_gate=Gate(),
        gated_mla=True,
        use_hpc_gated_mla=False,
        config=SimpleNamespace(gating_type="elementwise"),
    )
    return implementation, owner, Tensor(), events, policy, parent, auxiliary


def test_gate_overlap_has_current_stream_fork_and_join(gate_overlap):
    implementation, owner, hidden, events, _, _, _ = gate_overlap
    ticket = implementation.GateOverlap().begin(owner, hidden)
    assert events[0] == ("auxiliary", "wait", "parent")
    assert events.count("projection") == 1
    assert ("parent", "wait", "auxiliary") not in events
    factor = ticket.finish()
    assert factor is ticket.factor
    assert events[-2:] == [
        ("parent", "wait", "auxiliary"),
        ("tensor", "record", "parent"),
    ]
    with pytest.raises(RuntimeError, match="already joined"):
        ticket.finish()


def test_gate_overlap_disabled_policy_keeps_serial_path(gate_overlap):
    implementation, owner, hidden, events, policy, _, _ = gate_overlap
    overlap = implementation.GateOverlap()
    ticket = overlap.begin(owner, hidden)
    assert ticket is not None
    ticket.finish()
    events.clear()
    policy["mul_scaled_int8_quant"] = False
    assert overlap.begin(owner, hidden) is None
    assert events == []
    policy["mul_scaled_int8_quant"] = True
    gate = owner.linear_gate
    for register in (
        gate.register_forward_pre_hook,
        gate.register_forward_hook,
        gate.register_full_backward_pre_hook,
        gate.register_full_backward_hook,
    ):
        handle = register(lambda *args: None)
        try:
            assert overlap.begin(owner, hidden) is None
            assert events == []
        finally:
            handle.remove()
    observed = []
    handle = gate.register_forward_hook(lambda *args: observed.append("hook"))
    try:
        assert overlap.begin(owner, hidden) is None
        gate(hidden)
        assert observed == ["hook"] and events == ["projection"]
    finally:
        handle.remove()


def test_gate_overlap_projection_failure_joins_and_propagates(gate_overlap):
    implementation, owner, hidden, events, _, _, _ = gate_overlap

    class BrokenGate(type(owner.linear_gate)):
        def forward(self, hidden):
            events.append("failed_projection")
            raise RuntimeError("gate failed")

    owner.linear_gate = BrokenGate()
    with pytest.raises(RuntimeError, match="gate failed"):
        implementation.GateOverlap().begin(owner, hidden)
    assert events.count("failed_projection") == 1
    assert events[-1] == ("parent", "wait", "auxiliary")


@pytest.fixture
def rms_handoff(monkeypatch):
    calls, policy = [], {"rms_scaled_int8_quant": True, "rms_norm": True}

    class Kernel:
        def __init__(self, config, layer_param_names):
            self.config, self.layer_param_names = config, layer_param_names

        def _get_layer_params(self, linear):
            return linear.parameters

        def apply_weights(self, linear, x, bias=None):
            calls.append("native")
            return "native_output"

    class Tensor(TensorMetadata):
        layout = torch.strided
        requires_grad = False

        def storage_offset(self):
            return 0

        def is_conj(self):
            return False

        def is_neg(self):
            return False

    class QuantizedActivation:
        def __init__(self, data, scale, orig_dtype, orig_shape, quant_key):
            self.data, self.scale = data, scale
            self.orig_dtype, self.orig_shape, self.quant_key = (
                orig_dtype,
                orig_shape,
                quant_key,
            )

    class Norm:
        def forward_cuda(self):
            pass

        forward_oot = forward_cuda

    class Linear:
        pass

    class Scheme:
        def apply_weights(self):
            pass

    class Method:
        pass

    class FlagGemsBackend:
        def rms_norm(self):
            pass

    backend = FlagGemsBackend()
    manager = SimpleNamespace(policy_epoch=7, resolve=lambda op: backend.rms_norm)
    module(
        monkeypatch,
        "vllm.model_executor.kernels.linear.scaled_mm.cutlass",
        CutlassInt8ScaledMMLinearKernel=Kernel,
    )
    module(
        monkeypatch, "vllm.model_executor.layers.linear", ColumnParallelLinear=Linear
    )
    module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors",
        CompressedTensorsLinearMethod=Method,
    )
    module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_w8a8_int8",
        CompressedTensorsW8A8Int8=Scheme,
    )
    module(
        monkeypatch,
        "vllm.model_executor.layers.fusion.quant_activation",
        QuantizedActivation=QuantizedActivation,
        as_quantized_activation=lambda x, key: x
        if isinstance(x, QuantizedActivation)
        else None,
        expose_input_quant_key=lambda *args: None,
    )
    module(
        monkeypatch,
        "vllm.model_executor.layers.quantization.utils.quant_utils",
        kInt8DynamicTokenSym="dynamic_symmetric_int8",
    )
    module(
        monkeypatch,
        "vllm.platforms",
        PlatformEnum=SimpleNamespace(CUDA="CUDA"),
        current_platform=SimpleNamespace(
            _enum="CUDA", get_device_capability=lambda: (9, 0)
        ),
    )
    module(
        monkeypatch,
        "vllm_fl.utils",
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )
    module(
        monkeypatch,
        "vllm_fl.dispatch",
        get_default_manager=lambda: manager,
        get_policy_epoch=lambda: 3,
        is_dump_enabled=lambda: False,
    )
    module(
        monkeypatch,
        "vllm_fl.dispatch.backends.flaggems.flaggems",
        FlagGemsBackend=FlagGemsBackend,
    )
    module(monkeypatch, "vllm_fl.ops.layernorm", RMSNormFL=Norm)
    module(
        monkeypatch,
        "flag_gems.modules",
        normalization=SimpleNamespace(use_c_extension=False),
    )
    ops = SimpleNamespace(
        cutlass_scaled_mm=lambda *args, **kwargs: calls.append(("gemm", args, kwargs))
    )
    native = module(monkeypatch, "vllm", _custom_ops=ops)
    native.__path__ = []
    package = module(monkeypatch, "hy4_quant_handoff_fixture")
    quant_dir = MODEL.parents[2] / "quantization/w8a8"
    package.__path__ = [str(quant_dir)]

    def load_quant(name, filename):
        spec = importlib.util.spec_from_file_location(name, quant_dir / filename)
        loaded = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(loaded)
        monkeypatch.setitem(sys.modules, name, loaded)
        return loaded

    consumer = load_quant(
        "hy4_quant_handoff_fixture.prequantized_int8", "prequantized_int8.py"
    )
    producer = load_quant("hy4_quant_handoff_fixture.q_a_rms_quant", "q_a_rms_quant.py")
    torch_metadata = SimpleNamespace(
        Tensor=Tensor,
        bfloat16=torch.bfloat16,
        int8=torch.int8,
        float32=torch.float32,
        strided=torch.strided,
        compiler=SimpleNamespace(is_compiling=lambda: False),
    )
    consumer.torch = torch_metadata
    producer.torch = torch_metadata
    norm, linear = Norm(), Linear()
    for obj in (norm, linear):
        obj._forward_hooks, obj._forward_pre_hooks = {}, {}
        obj._backward_hooks, obj._backward_pre_hooks = {}, {}
    norm.variance_size_override = None
    norm.has_weight = norm.pass_weight = True
    norm.hidden_size, norm.variance_epsilon = 1024, 1e-5
    norm.weight = Tensor((1024,), torch.bfloat16)
    norm._forward_method = norm.forward_cuda
    linear.quant_method, linear.scheme = Method(), Scheme()
    linear.bias, linear.return_bias = None, True
    linear.input_size_per_partition, linear.params_dtype = 1024, torch.bfloat16
    config = SimpleNamespace(
        is_channelwise=True, is_static_input_scheme=False, input_symmetric=True
    )
    linear.scheme.kernel = Kernel(config, ["weight", "scale"])
    linear.parameters = (
        Tensor((1024, 6144), torch.int8, (1, 1024)),
        Tensor((6144, 1), torch.float32, (1, 1)),
        None,
        None,
        None,
    )
    x = Tensor((2, 1024), torch.bfloat16)
    qc, q, scale = (
        Tensor((2, 1024), torch.bfloat16),
        Tensor((2, 1024), torch.int8),
        Tensor((2, 1), torch.float32),
    )

    def fused(*args):
        calls.append("rms_quant")
        return qc, q, scale

    provider = module(
        monkeypatch,
        "flaggems_vllm",
        rms_scaled_int8_quant=fused,
        supports_rms_scaled_int8_quant=lambda *args: True,
    )
    return (
        producer,
        consumer,
        norm,
        linear,
        x,
        calls,
        policy,
        manager,
        provider,
        (qc, q, scale),
        ops,
    )


def test_rms_handoff_uses_fused_output_and_native_gemm(rms_handoff):
    producer, _, norm, linear, x, calls, _, _, _, expected, _ = rms_handoff
    selected = producer.configure_q_a_rms_quant(norm, linear)
    qc, activation = producer.maybe_q_a_rms_quant(selected, norm, linear, x)
    assert qc is expected[0]
    assert activation.data is expected[1]
    assert activation.scale is expected[2]
    linear.scheme.kernel.apply_weights(linear, activation)
    assert calls[0] == "rms_quant"
    assert calls[1][0] == "gemm"
    assert calls[1][1][0] is expected[1]
    assert calls[1][2]["scale_a"] is expected[2]
    assert "native" not in calls


@pytest.mark.parametrize("change", ["manager_epoch", "policy", "dtype"])
def test_rms_handoff_rejects_changed_contract_before_launch(rms_handoff, change):
    producer, _, norm, linear, x, calls, policy, manager, _, _, _ = rms_handoff
    selected = producer.configure_q_a_rms_quant(norm, linear)
    assert selected.enabled
    if change == "manager_epoch":
        manager.policy_epoch += 1
    elif change == "policy":
        policy["rms_scaled_int8_quant"] = False
    else:
        linear.parameters[0].dtype = torch.float32
    assert producer.maybe_q_a_rms_quant(selected, norm, linear, x) is None
    assert calls == []


def test_rms_selected_execution_error_does_not_retry(rms_handoff):
    producer, _, norm, linear, x, calls, _, _, provider, _, _ = rms_handoff
    selected = producer.configure_q_a_rms_quant(norm, linear)

    def fail(*args):
        calls.append("failed_rms_quant")
        raise RuntimeError("RMS quantization failed")

    provider.rms_scaled_int8_quant = fail
    with pytest.raises(RuntimeError, match="RMS quantization failed"):
        producer.maybe_q_a_rms_quant(selected, norm, linear, x)
    assert calls == ["failed_rms_quant"]


def test_reader_forwards_slot_endpoint_and_native_return_contract(reader, monkeypatch):
    implementation, _, _ = reader
    calls, endpoints = [], []

    def pad(q, sinks, indices, q_out, sink_out, slot_out):
        calls.append("pad")
        q_out.zero_()
        sink_out.fill_(float("-inf"))
        slot_out.fill_(7)
        endpoints.append(slot_out)

    def native(q, cache, indices, scale, **kwargs):
        calls.append("reader")
        assert kwargs["topk_length"] is endpoints[0]
        assert kwargs["attn_sink"].shape == (64,)
        assert tuple(indices.shape) == (2, 1, 2048)
        return (
            torch.full((2, 64, 512), 3, dtype=torch.bfloat16),
            None,
            torch.zeros(2, 64),
        )

    module(monkeypatch, "flaggems_vllm", pad_mla_q_sink_slot_bound=pad)
    monkeypatch.setattr(implementation, "flash_mla_sparse_fwd", native)
    owner = object.__new__(implementation.HYV4FlashMLASparseImpl)
    owner.kv_cache_dtype, owner.num_heads, owner.prefill_padding = "fp8_ds_mla", 4, 64
    owner.softmax_scale = 1 / 16
    owner.prefill_bf16_workspace = torch.empty(5, 576, dtype=torch.bfloat16)
    owner.sinks = torch.zeros(4)
    q = torch.zeros(2, 4, 576, dtype=torch.bfloat16)
    indices = torch.full((2, 2048), -1, dtype=torch.int32)
    indices[:, 6] = 4
    actual = owner._bf16_flash_mla_kernel(q, owner.prefill_bf16_workspace, indices)
    assert calls == ["pad", "reader"]
    assert torch.equal(actual, torch.full((2, 4, 512), 3, dtype=torch.bfloat16))
    assert indices[0, 6].item() == 4


def test_pair_policy_support_and_failure_are_resolved_before_retry(monkeypatch):
    policy, calls = {}, []
    implementation = source_load(
        monkeypatch,
        "hy4_ihc_pair_contract",
        "../fusions.py",
        {"maybe_ihc_pair_pre"},
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )
    owner = SimpleNamespace(
        hc_fn=SimpleNamespace(weight="weight"),
        hc_scale="scale",
        hc_base="base",
        magnitude=2.0,
        hc_eps=1e-6,
        layernorm_epsilon=1e-5,
    )
    assert implementation.maybe_ihc_pair_pre(owner, "input") is None

    def execute(*args):
        calls.append(args)
        raise RuntimeError("pair execution failed")

    provider = module(
        monkeypatch,
        "flaggems_vllm",
        can_ihc_pair=lambda *args, **kwargs: True,
        ihc_pre_pair=execute,
    )
    policy["ihc_pre_pair"] = True
    with pytest.raises(RuntimeError, match="pair execution failed"):
        implementation.maybe_ihc_pair_pre(owner, "input")
    assert calls == [("input", "weight", "scale", "base", 2.0, 1e-6, 1e-5)]
    calls.clear()
    provider.can_ihc_pair = lambda *args, **kwargs: False
    assert implementation.maybe_ihc_pair_pre(owner, "input") is None
    assert calls == []


def test_reader_unbound_common_keeps_original_path(reader):
    f, owner, common = reader_fixture(reader)
    assert owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)
    common.cpu_seq_lens_proof = None
    assert not owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)


@pytest.mark.parametrize("available", [False, True])
def test_full_decode_selects_only_prewarmed_metadata(
    score_consumer, monkeypatch, available
):
    owner, _, _, calls, _ = score_consumer
    context = owner.forward.__globals__["get_forward_context"]()
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    metadata = SimpleNamespace(
        num_prefills=0,
        num_decode_tokens=4,
        slot_mapping=torch.arange(4),
        decode=SimpleNamespace(
            decode_lens=torch.tensor([4]),
            seq_lens=torch.tensor([123]),
            block_table=torch.zeros(1, 3200, dtype=torch.int32),
        ),
    )
    context.attn_metadata = {"cache": metadata}
    owner.indices = torch.full((4, 2048), -1, dtype=torch.int32)

    def prepare(*args):
        calls.append("prepare")
        return torch.zeros(4, dtype=torch.int64), torch.arange(
            120, 124, dtype=torch.int32
        )

    monkeypatch.setitem(
        owner.forward.__globals__,
        "full_decode_metadata_available",
        lambda *args: available,
    )
    monkeypatch.setitem(
        owner.forward.__globals__, "prepare_full_decode_metadata", prepare
    )
    observed = []
    monkeypatch.setattr(
        owner,
        "_score",
        lambda q, weights, table, rows, lengths, output: observed.append(
            (rows.clone(), lengths.clone())
        ),
    )
    owner.forward(
        torch.zeros(4, 16),
        torch.zeros(4, 32, 128, dtype=torch.int8),
        torch.zeros(4, 128),
        torch.zeros(4, 32),
    )
    assert calls == (["store", "prepare"] if available else ["store"])
    assert len(observed) == 1
    assert torch.equal(observed[0][0], torch.zeros(4, dtype=torch.int64))
    assert torch.equal(
        observed[0][1], torch.arange(120, 124, dtype=observed[0][1].dtype)
    )


def test_full_decode_metadata_execution_error_does_not_retry(
    score_consumer, monkeypatch
):
    owner, _, _, calls, _ = score_consumer
    context = owner.forward.__globals__["get_forward_context"]()
    context.cudagraph_runtime_mode = CUDAGraphMode.FULL
    context.attn_metadata = {
        "cache": SimpleNamespace(
            num_prefills=0,
            num_decode_tokens=4,
            slot_mapping=torch.arange(4),
            decode=SimpleNamespace(
                decode_lens=torch.tensor([4]), seq_lens=torch.tensor([123])
            ),
        )
    }
    owner.indices = torch.full((4, 2048), -1, dtype=torch.int32)

    def fail(*args):
        calls.append("failed_metadata")
        raise RuntimeError("metadata launch failed")

    monkeypatch.setitem(
        owner.forward.__globals__, "full_decode_metadata_available", lambda *args: True
    )
    monkeypatch.setitem(owner.forward.__globals__, "prepare_full_decode_metadata", fail)
    monkeypatch.setattr(owner, "_score", lambda *args: calls.append("score"))
    with pytest.raises(RuntimeError, match="metadata launch failed"):
        owner.forward(
            torch.zeros(4, 16),
            torch.zeros(4, 32, 128, dtype=torch.int8),
            torch.zeros(4, 128),
            torch.zeros(4, 32),
        )
    assert calls == ["store", "failed_metadata"]


@pytest.fixture
def stock_builder_contract(monkeypatch):
    """Exercise plugin orchestration against the stock API surface, on CPU."""
    monkeypatch.setenv("VLLM_PLUGINS", "")
    from vllm.v1.attention.backends.mla.indexer import (
        DeepseekV32IndexerMetadataBuilder as InstalledStockBuilder,
    )
    from vllm.v1.attention.backends.utils import (
        split_decodes_and_prefills as installed_split,
    )

    events = []
    package = module(monkeypatch, "hy4_stock_builder_fixture")
    package.__path__ = [str(MODEL.parent)]
    module(monkeypatch, package.__name__ + ".nvidia").__path__ = [str(MODEL)]
    spec = importlib.util.spec_from_file_location(
        "hy4_stock_builder_fixture.metadata", MODEL.parent / "metadata.py"
    )
    proof_api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, proof_api)
    spec.loader.exec_module(proof_api)

    @dataclasses.dataclass
    class StockIndexerMetadata:
        seq_lens: object
        max_seq_len: int
        slot_mapping: object
        num_decodes: int
        num_decode_tokens: int
        num_prefills: int
        num_prefill_tokens: int
        decode: object = None
        prefill: object = None

    @dataclasses.dataclass
    class StockPrefill:
        chunks: list

    @dataclasses.dataclass
    class StockDecode:
        block_table: object
        seq_lens: object
        decode_lens: object
        requires_padding: bool
        schedule_metadata: object
        global_seq_lens: object = None
        indices: object = None

    class StockBuilder:
        _prepare_decode_tensors = InstalledStockBuilder._prepare_decode_tensors
        _split = staticmethod(installed_split)

        def __init__(self, *args, **kwargs):
            events.append("native_init")

        def build(self, *args, **kwargs):
            raise AssertionError("the INT8 adapter must not request the FP8 scheduler")

    starts = torch.tensor([0, 4, 13, 23], dtype=torch.int32)
    upper = torch.tensor([23, 12000, 13000], dtype=torch.int32)
    proof = proof_api.CPUSequenceLengthsProof(
        "test",
        9,
        ("decode", "prefill-a", "prefill-b"),
        tuple(starts.tolist()),
        (19, 11991, 12990),
        (19, 12100, 13100),
        (-1, -1, -1),
        (-1, -1, -1),
    )
    common = SimpleNamespace(
        num_reqs=3,
        num_actual_tokens=23,
        query_start_loc=starts.clone(),
        query_start_loc_cpu=starts,
        seq_lens=upper.clone(),
        max_seq_len=13000,
        seq_lens_cpu_upper_bound=upper,
        block_table_tensor=torch.zeros(3, 3200, dtype=torch.int32),
        slot_mapping=torch.arange(23),
        is_prefilling=torch.tensor([False, True, True]),
    )
    active = [proof]

    def bind(current):
        events.append("bind")
        current.cpu_seq_lens_proof = active[0]
        proof_api.set_current_cpu_seq_lens_proof(active[0])

    module(
        monkeypatch,
        "hy4_stock_builder_fixture.runtime",
        bind_common_runtime_metadata=bind,
    )

    def split(common, **kwargs):
        events.append(("split", kwargs))
        return 1, 2, 4, 19

    def specs(lengths, queries, max_buffer, logits_bytes, **kwargs):
        events.append(("chunks", lengths.tolist(), queries.tolist(), kwargs))
        return [(slice(1, 3), slice(0, 19))]

    chunk = Chunk(
        num_reqs=2,
        total_seq_lens=25000,
        local_total_seq_lens=25000,
        token_start=4,
        token_end=23,
        block_table=common.block_table_tensor[1:3],
    )

    def make_chunk(*args, **kwargs):
        events.append(("make_chunk", args, kwargs))
        return chunk

    def forbidden_schedule(*args, **kwargs):
        events.append("FP8_scheduler")
        raise AssertionError("INT8 has no FP8 scheduler consumer")

    native = SimpleNamespace(
        DeepseekV32IndexerMetadata=StockIndexerMetadata,
        DeepseekV32IndexerPrefillMetadata=StockPrefill,
        DeepSeekV32IndexerDecodeMetadata=StockDecode,
        split_indexer_prefill_chunks=specs,
        build_prefill_chunk_metadata=make_chunk,
        _supports_native_decode=lambda count: count in (1, 2, 4),
        has_deep_gemm=lambda: True,
        get_paged_mqa_logits_metadata=forbidden_schedule,
    )
    implementation = source_load(
        monkeypatch,
        "hy4_stock_builder_fixture.nvidia.indexer",
        "indexer.py",
        {
            "HYV4Int8IndexerMetadataBuilder",
            "HYV4Int8IndexerBackend",
            "HYV4Int8IndexerCache",
        },
        torch=torch,
        native_indexer=native,
        DeepseekV32IndexerBackend=type("StockBackend", (), {}),
        DeepseekV32IndexerCache=type("StockCache", (), {}),
        DeepseekV32IndexerMetadataBuilder=StockBuilder,
        CPUSequenceLengthsProof=proof_api.CPUSequenceLengthsProof,
        get_current_cpu_seq_lens_proof=proof_api.get_current_cpu_seq_lens_proof,
        validated_prefill_chunk_bound=proof_api.validated_prefill_chunk_bound,
        split_decodes_and_prefills=split,
        envs=SimpleNamespace(VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=1024),
    )
    owner = object.__new__(implementation.HYV4Int8IndexerMetadataBuilder)
    owner.decode_threshold = 4
    owner.use_flattening = owner.supports_varlen = False
    owner.max_prefill_buffer_size = 8192000
    owner.cp_kv_cache_interleave_size = 1
    owner.decode_lens_buffer = torch.zeros(32, dtype=torch.int32)
    owner.num_speculative_tokens = 3
    owner.scheduler_metadata_buffer = torch.empty(1, 2, dtype=torch.int32)
    decode_seq = torch.tensor([[20, 21, 22, 23]], dtype=torch.int32)

    def prepare(**kwargs):
        events.append(("decode", kwargs))
        return decode_seq, kwargs["block_table"], kwargs["decode_lens"], 1, False

    owner._prepare_decode_tensors = prepare
    return (
        implementation,
        owner,
        common,
        active,
        proof_api,
        native,
        events,
        chunk,
        decode_seq,
    )


def test_stock_int8_builder_binds_exact_chunks_without_fp8_scheduler(
    stock_builder_contract,
):
    _, owner, common, active, _, native, events, chunk, decode_seq = (
        stock_builder_contract
    )
    original_schedule = native.get_paged_mqa_logits_metadata
    actual = owner.build(0, common, fast_build=True)
    assert events[0] == "bind"
    assert actual.seq_lens is common.seq_lens
    assert actual.slot_mapping is common.slot_mapping
    assert (
        actual.num_decodes,
        actual.num_decode_tokens,
        actual.num_prefills,
        actual.num_prefill_tokens,
    ) == (1, 4, 2, 19)
    assert actual.cpu_seq_lens_proof_generation == 9
    assert actual.prefill.chunks == [chunk]
    assert chunk.cpu_seq_len_bound.step_proof is active[0]
    assert chunk.cpu_seq_len_bound.seq_lens == (12000, 13000)
    assert (
        chunk.cpu_seq_len_bound.request_start,
        chunk.cpu_seq_len_bound.request_stop,
    ) == (1, 3)
    assert actual.decode.seq_lens is decode_seq
    assert actual.decode.schedule_metadata is owner.scheduler_metadata_buffer
    assert "FP8_scheduler" not in events
    assert native.get_paged_mqa_logits_metadata is original_schedule
    built = next(
        event
        for event in events
        if isinstance(event, tuple) and event[0] == "make_chunk"
    )
    assert built[1][0:2] == (1, 3)
    assert built[1][2] is common.query_start_loc
    assert built[1][4] is common.seq_lens and built[1][5] is common.seq_lens
    assert built[2]["query_slice"] == slice(0, 19)


def test_stock_int8_builder_does_not_reuse_a_previous_step_proof(
    stock_builder_contract,
):
    _, owner, common, active, _, _, _, chunk, _ = stock_builder_contract
    first = owner.build(0, common)
    assert first.cpu_seq_lens_proof_generation == 9
    assert chunk.cpu_seq_len_bound is not None
    active[0] = None
    second = owner.build(0, common)
    assert second.cpu_seq_lens_proof_generation is None
    assert chunk.cpu_seq_len_bound is None


def test_stock_int8_builder_native_helper_error_is_not_retried(stock_builder_contract):
    _, owner, common, _, _, native, events, _, _ = stock_builder_contract

    def fail(*args, **kwargs):
        events.append("failed_chunk")
        raise RuntimeError("native metadata helper failed")

    native.build_prefill_chunk_metadata = fail
    with pytest.raises(RuntimeError, match="native metadata helper failed"):
        owner.build(0, common)
    assert events.count("failed_chunk") == 1
    assert "FP8_scheduler" not in events


def test_stock_int8_builder_declines_context_parallelism(stock_builder_contract):
    implementation, _, _, _, _, _, events, _, _ = stock_builder_contract
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=2, prefill_context_parallel_size=1
        )
    )
    with pytest.raises(NotImplementedError, match="CP1"):
        implementation.HYV4Int8IndexerMetadataBuilder(
            SimpleNamespace(compress_ratio=1),
            [],
            config,
            torch.device("cpu"),
            block_table_width=3200,
        )
    assert "native_init" not in events


def test_mla_builder_binds_common_before_stock_builder(reader, monkeypatch):
    implementation, _, _ = reader
    calls = []
    base = implementation.HYV4FlashMLASparseMetadataBuilder.__bases__[0]
    common = SimpleNamespace()
    result = SimpleNamespace()

    def native_build(self, prefix, current, fast):
        assert current is common and current.bound
        calls.append(("native", prefix, fast))
        return result

    def bind(current):
        current.bound = True
        calls.append("bind")

    module(
        monkeypatch,
        "hy4_reader_fixture.runtime",
        bind_common_runtime_metadata=bind,
    )
    monkeypatch.setattr(base, "build", native_build, raising=False)
    owner = object.__new__(implementation.HYV4FlashMLASparseMetadataBuilder)
    owner._use_fp8_bf16_prefill = lambda *args: False
    assert owner.build(7, common, fast_build=True) is result
    assert calls == ["bind", ("native", 7, True)]


@pytest.mark.parametrize(
    "counts,seq_lens,padded,use_native",
    [
        ((4, 4), (20, 31), 8, True),
        ((4, 1), (20, 7), 5, False),
        ((4, 1), (20, 7), 8, False),
        ((4, 0), (20, 0), 8, True),
        ((0, 0), (0, 0), 8, False),
    ],
)
def test_decode_layout_matches_stock_helper_and_token_oracle(
    stock_builder_contract,
    score_consumer,
    monkeypatch,
    counts,
    seq_lens,
    padded,
    use_native,
):
    implementation, builder, common, active, _, _, events, _, _ = stock_builder_contract
    consumer, _, _, consumer_calls, _ = score_consumer
    base = implementation.HYV4Int8IndexerMetadataBuilder.__bases__[0]
    split_calls = []

    def split(current, **kwargs):
        split_calls.append(kwargs)
        # Older installed vLLM lacks this optional 0.28 keyword; its CPU
        # partition semantics for these cases are otherwise identical.
        supported = inspect.signature(base._split).parameters
        return base._split(
            current, **{k: v for k, v in kwargs.items() if k in supported}
        )

    monkeypatch.setattr(implementation, "split_decodes_and_prefills", split)
    builder.use_flattening = counts == (4, 1)
    active[0] = None
    starts = [0, counts[0], sum(counts)]
    common.num_reqs, common.num_actual_tokens = 2, padded
    common.max_query_len = max(counts)
    common.query_start_loc_cpu = torch.tensor(starts, dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
    common.block_table_tensor = torch.tensor([[5, 6], [8, 9]], dtype=torch.int32)
    expected_lengths, expected_tables, slots = [], [], []
    for request, (count, length) in enumerate(zip(counts, seq_lens)):
        for step in range(count):
            position = length - count + step
            expected_lengths.append(position + 1)
            expected_tables.append(common.block_table_tensor[request].clone())
            slots.append(
                int(common.block_table_tensor[request, position // 64]) * 64
                + position % 64
            )
    slots.extend([-1] * (padded - len(slots)))
    expected_lengths.extend([0] * (padded - len(expected_lengths)))
    common.slot_mapping = torch.tensor(slots)
    original_slots, original_seq_lens = (
        common.slot_mapping.clone(),
        common.seq_lens.clone(),
    )
    builder.decode_seq_lens_buffer = torch.full((32,), -999, dtype=torch.int32)
    builder.expanded_block_table_buffer = torch.full((32, 2), -999, dtype=torch.int32)
    builder.arange_buffer = torch.arange(32, dtype=torch.int32)
    builder.offsets_buffer = torch.arange(4, dtype=torch.int32)
    actual_stock_helper = implementation.HYV4Int8IndexerMetadataBuilder.__bases__[
        0
    ]._prepare_decode_tensors
    helper_calls = []

    def prepare(**kwargs):
        helper_calls.append(kwargs)
        return actual_stock_helper(builder, **kwargs)

    builder._prepare_decode_tensors = prepare
    metadata = builder.build(0, common)
    assert len(split_calls) == len(helper_calls) == 1
    assert split_calls[0]["require_uniform"] is not builder.use_flattening
    assert (metadata.num_decodes, metadata.num_prefills) == (2, 0)
    assert helper_calls[0]["use_native"] is use_native
    assert metadata.decode.requires_padding is (
        use_native and min(counts) != max(counts)
    )
    if use_native:
        assert metadata.decode.seq_lens.shape == (2, 4)
        assert (
            metadata.decode.seq_lens.data_ptr()
            == builder.decode_seq_lens_buffer.data_ptr()
        )
        assert (
            metadata.decode.block_table.data_ptr()
            == common.block_table_tensor.data_ptr()
        )
    elif max(counts):
        assert metadata.decode.seq_lens.shape == (padded, 1)
        assert metadata.decode.decode_lens.tolist() == [1] * padded
    else:
        assert not torch.any(metadata.decode.seq_lens)
        # Both the older stock empty-request layout and the target 0.28
        # flattened layout must produce a fully masked fixed-size token map.
        assert int(metadata.decode.decode_lens.sum()) in (0, padded)
    assert metadata.slot_mapping is common.slot_mapping
    assert torch.equal(common.seq_lens, original_seq_lens)
    assert torch.equal(metadata.slot_mapping, original_slots)
    context = consumer.forward.__globals__["get_forward_context"]()
    context.attn_metadata = {"cache": metadata}
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    stored_slots, score_inputs = [], []

    def cache_store(key, cache, slot_mapping, page_size):
        consumer_calls.append("store")
        stored_slots.append(slot_mapping.clone())

    def score(q, weights, table, rows, lengths, output):
        score_inputs.append((table, rows.clone(), lengths.clone()))

    monkeypatch.setitem(
        consumer.forward.__globals__, "int8_indexer_cache_store", cache_store
    )
    monkeypatch.setattr(consumer, "_score", score)
    consumer.indices = torch.full((padded, 2048), 99, dtype=torch.int32)

    def forbid_device_read(*args, **kwargs):
        raise AssertionError(
            "decode token mapping must not synchronize a device scalar"
        )

    with monkeypatch.context() as reads:
        reads.setattr(torch.Tensor, "item", forbid_device_read)
        reads.setattr(torch.Tensor, "cpu", forbid_device_read)
        consumer.forward(
            torch.zeros(padded, 16),
            torch.zeros(padded, 32, 128, dtype=torch.int8),
            torch.zeros(padded, 128),
            torch.zeros(padded, 32),
        )
    assert consumer_calls == ["store"]
    assert torch.equal(stored_slots[0], original_slots)
    assert len(score_inputs) == 1
    table, rows, lengths = score_inputs[0]
    assert lengths.tolist() == expected_lengths
    expected_rows = (
        [request for request, count in enumerate(counts) for _ in range(count)]
        + [-1] * (padded - sum(counts))
        if use_native
        else (
            list(range(padded))
            if metadata.decode.decode_lens.shape[0] == padded
            else [-1] * padded
        )
    )
    assert rows.tolist() == expected_rows
    for token, expected_table in enumerate(expected_tables):
        assert torch.equal(table[rows[token]], expected_table)
    for token in range(sum(counts), padded):
        if int(rows[token]) < 0:
            assert int(rows[token]) == -1
        else:
            assert int(table[rows[token], 0]) == 0
        assert int(lengths[token]) == 0 and int(stored_slots[0][token]) == -1
    assert torch.equal(consumer.indices, torch.full_like(consumer.indices, -1))
    assert "FP8_scheduler" not in events


@pytest.mark.parametrize(
    "starts,tokens,error",
    [
        ((0, 4, 3), 3, "decode lengths"),
        ((0, 4, 8), 7, "decode lengths"),
        ((1, 5, 9), 8, "begin at zero"),
    ],
)
def test_invalid_decode_counts_fail_before_metadata_buffer_writes(
    stock_builder_contract, monkeypatch, starts, tokens, error
):
    implementation, owner, common, _, _, _, events, _, _ = stock_builder_contract
    split_calls = []

    def split(*args, **kwargs):
        split_calls.append((args, kwargs))
        raise AssertionError(
            "invalid CPU counts must be rejected before native splitting"
        )

    monkeypatch.setattr(implementation, "split_decodes_and_prefills", split)
    common.num_reqs, common.num_actual_tokens = 2, tokens
    common.query_start_loc_cpu = torch.tensor(starts, dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    owner.decode_lens_buffer.fill_(17)
    before = owner.decode_lens_buffer.clone()
    with pytest.raises(ValueError, match=error):
        owner.build(0, common)
    assert not split_calls
    assert torch.equal(owner.decode_lens_buffer, before)
    assert not any(
        isinstance(event, tuple) and event[0] in ("decode", "make_chunk")
        for event in events
    )


@pytest.mark.parametrize(
    "counts,lengths", [((4, 1), (20, 7)), ((4, 9, 10), (23, 30, 40))]
)
def test_stock_partition_keeps_uniform_decode_and_routes_other_rows_to_prefill(
    stock_builder_contract, score_consumer, monkeypatch, counts, lengths
):
    implementation, builder, common, active, _, native, events, _, _ = (
        stock_builder_contract
    )
    consumer, _, _, consumer_calls, _ = score_consumer
    active[0] = None
    base = implementation.HYV4Int8IndexerMetadataBuilder.__bases__[0]
    split_calls, chunk_calls, helper_calls = [], [], []

    def split(current, **kwargs):
        split_calls.append(kwargs)
        supported = inspect.signature(base._split).parameters
        return base._split(
            current, **{k: v for k, v in kwargs.items() if k in supported}
        )

    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count)
    common.num_reqs = len(counts)
    common.num_actual_tokens = starts[-1]
    common.max_query_len = max(counts)
    common.query_start_loc_cpu = torch.tensor(starts, dtype=torch.int32)
    common.query_start_loc = common.query_start_loc_cpu.clone()
    common.seq_lens = torch.tensor(lengths, dtype=torch.int32)
    common.seq_lens_cpu_upper_bound = common.seq_lens.clone()
    common.block_table_tensor = torch.tensor(
        [[5 + request] for request in range(len(counts))], dtype=torch.int32
    )
    slots, oracle = [], []
    for request, (count, length) in enumerate(zip(counts, lengths)):
        for step in range(count):
            oracle.append((request, length - count + step + 1))
            slots.append((5 + request) * 64 + length - count + step)
    common.slot_mapping = torch.tensor(slots)
    builder.decode_seq_lens_buffer = torch.zeros(32, dtype=torch.int32)
    builder.offsets_buffer = torch.arange(4, dtype=torch.int32)
    builder._prepare_decode_tensors = lambda **kwargs: (
        helper_calls.append(kwargs) or base._prepare_decode_tensors(builder, **kwargs)
    )

    def chunk_specs(upper, queries, *args, **kwargs):
        chunk_calls.append((upper.clone(), queries.clone(), kwargs))
        return [(slice(1, len(counts)), slice(0, sum(counts[1:])))]

    def make_chunk(start, stop, *args, **kwargs):
        local_starts, keys, ends = [0], [], []
        for request in range(start, stop):
            for step in range(counts[request]):
                keys.append(local_starts[-1])
                ends.append(
                    local_starts[-1] + lengths[request] - counts[request] + step + 1
                )
            local_starts.append(local_starts[-1] + lengths[request])
        return Chunk(
            token_start=starts[start],
            token_end=starts[stop],
            local_cu_seq_lens=torch.tensor(local_starts, dtype=torch.int32),
            cu_seqlen_ks=torch.tensor(keys, dtype=torch.int32),
            cu_seqlen_ke=torch.tensor(ends, dtype=torch.int32),
            block_table=common.block_table_tensor[start:stop],
        )

    monkeypatch.setattr(implementation, "split_decodes_and_prefills", split)
    monkeypatch.setattr(native, "split_indexer_prefill_chunks", chunk_specs)
    monkeypatch.setattr(native, "build_prefill_chunk_metadata", make_chunk)
    result = builder.build(0, common)
    assert (result.num_decodes, result.num_prefills) == (1, len(counts) - 1)
    assert (result.num_decode_tokens, result.num_prefill_tokens) == (4, sum(counts[1:]))
    assert len(split_calls) == len(chunk_calls) == len(helper_calls) == 1
    assert split_calls[0]["require_uniform"] is True
    assert helper_calls[0]["use_native"] is True
    assert helper_calls[0]["num_decodes"] == 1
    assert chunk_calls[0][2]["request_offset"] == 1
    assert result.decode.seq_lens.shape == (1, 4)
    assert result.decode.seq_lens[0].tolist() == [
        lengths[0] - 3 + step for step in range(4)
    ]
    context = consumer.forward.__globals__["get_forward_context"]()
    context.attn_metadata = {"cache": result}
    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
    scored = []

    def score(q, weights, table, rows, seq_lens, output):
        scored.append(
            [
                (int(table[row, 0]) - 5, int(length))
                for row, length in zip(rows, seq_lens)
            ]
        )

    monkeypatch.setattr(consumer, "_score", score)
    consumer.indices = torch.full((sum(counts), 2048), 99, dtype=torch.int32)
    consumer.forward(
        torch.zeros(sum(counts), 16),
        torch.zeros(sum(counts), 32, 128, dtype=torch.int8),
        torch.zeros(sum(counts), 128),
        torch.zeros(sum(counts), 32),
    )
    assert scored == [oracle[4:], oracle[:4]]
    assert consumer_calls == ["store"]
    assert result.slot_mapping.tolist() == slots
    assert torch.equal(consumer.indices, torch.full_like(consumer.indices, -1))
    assert "FP8_scheduler" not in events
