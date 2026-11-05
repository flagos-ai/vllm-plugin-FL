# SPDX-License-Identifier: Apache-2.0
"""CPU metadata and dispatch contracts; these tests do not qualify GPU math."""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

MODEL = Path(__file__).resolve().parents[3] / "vllm_fl/models/hy_v4/nvidia"


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, MODEL / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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
            cudagraph_runtime_mode=SimpleNamespace(name="PIECEWISE"),
            ubatch_slices=None,
            dp_metadata=None,
            batch_descriptor=SimpleNamespace(
                num_tokens=starts[-1], has_lora=False, num_active_loras=0
            ),
            attn_metadata={"cache": self.meta},
        )
        self.indexer = SimpleNamespace(
            _hy4_int8_bounded_score_enabled=True,
            _hy4_int8_empty_tile_enabled=True,
            _hy4_int8_prefix_topk_enabled=True,
            _hy4_int8_paged_logits_symbol="int8_indexer_paged_logits_empty_tile",
            _hy4_int8_bounded_score_metadata_types=(Metadata, Chunk),
            _hy4_int8_cpu_bound_proof_type=ChunkProof,
            _hy4_int8_current_cpu_proof=lambda: self.step,
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
        with torch.inference_mode():
            if scoped:
                with selector.score_width_attention_scope(self.context):
                    return selector.select_mixed_score_capacity(*self.args)
            return selector.select_mixed_score_capacity(*self.args)


@pytest.fixture
def selector(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return load("hy4_score_contract", "score_width_dispatch.py")


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
        lambda f: setattr(f.context.cudagraph_runtime_mode, "name", "FULL"),
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
    backend = module(
        monkeypatch, "vllm.v1.attention.backend", CPUSequenceLengthsProof=StepProof
    )
    module(monkeypatch, "vllm.v1.attention", backend=backend)
    module(monkeypatch, "vllm_fl.utils", use_flaggems_op=lambda *args, **kwargs: True)
    module(
        monkeypatch,
        "vllm.logger",
        init_logger=lambda name: SimpleNamespace(warning_once=lambda message: None),
    )
    module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(is_device_capability=lambda value: True),
    )
    module(
        monkeypatch,
        "vllm.v1.attention.backends.mla.flashmla_sparse",
        FlashMLASparseBackend=type("Backend", (), {}),
        FlashMLASparseImpl=type("Impl", (), {}),
        FlashMLASparseMetadata=metadata,
        FlashMLASparseMetadataBuilder=type("Builder", (), {}),
    )
    module(
        monkeypatch,
        "vllm.v1.attention.ops.flashmla",
        flash_mla_sparse_fwd=None,
        flash_mla_with_kvcache=None,
    )
    module(
        monkeypatch, "vllm.v1.worker.workspace", current_workspace_manager=lambda: None
    )
    implementation = load("hy4_reader_contract", "flashmla_sparse.py")
    return implementation, backend, metadata


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
    fgv = module(
        monkeypatch,
        "flaggems_vllm",
        int8_indexer_cache_store=lambda *args: calls.append("store"),
        int8_indexer_paged_logits=lambda *args, **kwargs: calls.append("base"),
    )
    module(
        monkeypatch,
        "flag_gems",
        topk=lambda *args, **kwargs: calls.append("native_topk"),
    )
    module(
        monkeypatch,
        "vllm_fl.utils",
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )
    module(monkeypatch, "vllm.forward_context", get_forward_context=lambda: context)
    module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability=lambda capability: capability == 90,
        ),
    )
    package = module(monkeypatch, "hy4_score_consumer_fixture")
    package.__path__ = [str(MODEL)]
    monkeypatch.setitem(
        sys.modules, "hy4_score_consumer_fixture.score_width_dispatch", selector
    )
    consumer = load("hy4_score_consumer_fixture.int8_indexer", "int8_indexer.py")
    parallel = SimpleNamespace(
        decode_context_parallel_size=1, prefill_context_parallel_size=1
    )
    owner = consumer.HYV4Int8Indexer(
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
    owner._hy4_int8_paged_logits_symbol = "int8_indexer_paged_logits_empty_tile"
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
    policy["int8_indexer_paged_logits_empty_tile"] = True
    with pytest.raises(AttributeError, match="int8_indexer_paged_logits_empty_tile"):
        owner._configure_providers()
    assert calls == []


def test_provider_policy_enable_and_disable_use_public_exports(
    score_consumer, monkeypatch
):
    owner, fgv, policy, calls, _ = score_consumer
    fgv.int8_indexer_paged_logits_empty_tile = lambda *args: None
    fgv.indexer_prefix_topk = lambda *args: None
    fgv.supports_indexer_prefix_topk = lambda *args: True
    module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_cuda=lambda: True,
            is_device_capability=lambda capability: capability == 90,
        ),
    )
    backend = module(
        monkeypatch,
        "vllm.v1.attention.backend",
        PrefillChunkSequenceBounds=ChunkProof,
        get_current_cpu_seq_lens_proof=lambda: None,
    )
    module(monkeypatch, "vllm.v1.attention", backend=backend)
    module(
        monkeypatch,
        "vllm.v1.attention.backends.mla.indexer",
        DeepseekV32IndexerMetadata=Metadata,
        DeepseekV32IndexerPrefillChunkMetadata=Chunk,
    )
    assert not owner._hy4_int8_empty_tile_enabled
    assert not owner._hy4_int8_prefix_topk_enabled
    policy.update(int8_indexer_paged_logits_empty_tile=True, indexer_prefix_topk=True)
    owner._configure_providers()
    assert owner._hy4_int8_bounded_score_enabled
    assert owner._hy4_int8_prefix_topk is fgv.indexer_prefix_topk
    assert owner._hy4_int8_cpu_bound_proof_type is ChunkProof
    policy["int8_indexer_paged_logits_empty_tile"] = False
    owner._configure_providers()
    assert not owner._hy4_int8_bounded_score_enabled
    assert owner._hy4_int8_paged_logits_symbol == "int8_indexer_paged_logits"
    assert calls == []


def test_non_nvidia_policy_does_not_select_optional_provider(
    score_consumer, monkeypatch
):
    owner, _, policy, calls, _ = score_consumer
    policy.update(int8_indexer_paged_logits_empty_tile=True, indexer_prefix_topk=True)
    module(
        monkeypatch,
        "vllm.platforms",
        current_platform=SimpleNamespace(
            is_cuda=lambda: False, is_device_capability=lambda capability: True
        ),
    )
    owner._configure_providers()
    assert not owner._hy4_int8_empty_tile_enabled
    assert not owner._hy4_int8_prefix_topk_enabled
    assert not owner._hy4_int8_bounded_score_enabled
    assert owner._hy4_int8_paged_logits_symbol == "int8_indexer_paged_logits"
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
    mode = SimpleNamespace(NONE="NONE", FULL="FULL")
    context = SimpleNamespace(cudagraph_runtime_mode=mode.FULL)
    module(
        monkeypatch,
        "vllm.compilation.breakable_cudagraph",
        BreakableCUDAGraphCapture=SimpleNamespace(current=lambda: None),
    )
    module(monkeypatch, "vllm.config", CUDAGraphMode=mode)
    module(
        monkeypatch,
        "vllm.forward_context",
        get_forward_context=lambda: context,
        is_forward_context_available=lambda: True,
    )
    module(
        monkeypatch,
        "vllm_fl.utils",
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )
    monkeypatch.setattr(torch.cuda, "Stream", lambda **kwargs: auxiliary)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: parent)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(torch, "sigmoid", lambda value: value)
    implementation = load("hy4_gate_overlap_contract", "gate_overlap.py")

    class Gate:
        gather_output, bias = False, None
        weight = Tensor()
        _forward_hooks, _forward_pre_hooks = {}, {}

        def __call__(self, hidden):
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
    policy["mul_scaled_int8_quant"] = False
    assert implementation.GateOverlap().begin(owner, hidden) is None
    assert events == []


def test_gate_overlap_projection_failure_joins_and_propagates(gate_overlap):
    implementation, owner, hidden, events, _, _, _ = gate_overlap

    class BrokenGate(type(owner.linear_gate)):
        def __call__(self, hidden):
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
    module(
        monkeypatch,
        "vllm_fl.utils",
        use_flaggems_op=lambda op, default=True: policy.get(op, default),
    )
    implementation = load("hy4_ihc_pair_contract", "ihc_pair.py")
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


def test_reader_missing_native_proof_capability_keeps_original_path(reader):
    _, backend, _ = reader
    f, owner, common = reader_fixture(reader)
    assert owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)
    del backend.CPUSequenceLengthsProof
    assert not owner._use_fp8_bf16_mixed_prefill(0, common, f.meta)


@pytest.mark.parametrize("available", [False, True])
def test_full_decode_selects_only_prewarmed_metadata(
    score_consumer, monkeypatch, available
):
    owner, _, _, calls, _ = score_consumer
    context = owner.forward.__globals__["get_forward_context"]()
    context.cudagraph_runtime_mode = "FULL"
    module(monkeypatch, "vllm.config", CUDAGraphMode=SimpleNamespace(FULL="FULL"))
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

    module(
        monkeypatch,
        "hy4_score_consumer_fixture.decode_metadata",
        full_decode_metadata_available=lambda *args: available,
        prepare_full_decode_metadata=prepare,
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
    context.cudagraph_runtime_mode = "FULL"
    module(monkeypatch, "vllm.config", CUDAGraphMode=SimpleNamespace(FULL="FULL"))
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

    module(
        monkeypatch,
        "hy4_score_consumer_fixture.decode_metadata",
        full_decode_metadata_available=lambda *args: True,
        prepare_full_decode_metadata=fail,
    )
    monkeypatch.setattr(owner, "_score", lambda *args: calls.append("score"))
    with pytest.raises(RuntimeError, match="metadata launch failed"):
        owner.forward(
            torch.zeros(4, 16),
            torch.zeros(4, 32, 128, dtype=torch.int8),
            torch.zeros(4, 128),
            torch.zeros(4, 32),
        )
    assert calls == ["store", "failed_metadata"]
