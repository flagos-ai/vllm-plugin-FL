# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Coverage tests for the Enflame GCU attention adapter.

Target (read-only, commit 030b452f)::

    vllm_fl/dispatch/backends/vendor/gcu/impl/attention.py

The module is a pure Python adapter around the Enflame ``flash_attn``
kernel plus vLLM attention plumbing.  It never needs a real GCU for the
logic exercised here: ``flash_attn.vllm_flash_attn`` and the GCU
``reshape_and_cache`` helper are replaced with recorders, and the vLLM
symbols are either the real ones (when the host has vLLM) or small
in-memory placeholders installed only for the modules that cannot be
imported on a CPU host.

Everything runs on CPU tensors.  No kernel is launched, no device is
touched, nothing spawns or hangs.

Set ``FL_GCU_ATTENTION_PY`` to override the path to the module under
test if it lives outside the default Enflame source tree.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")


# ---------------------------------------------------------------------------
# Locate the module under test
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_REL = os.path.join(
    "vllm_fl", "dispatch", "backends", "vendor", "gcu", "impl", "attention.py"
)


def _candidate_paths():
    env = os.environ.get("FL_GCU_ATTENTION_PY")
    if env:
        yield env
    yield os.path.normpath(os.path.join(_HERE, os.pardir, "enflame-src-030b", _REL))
    yield os.path.normpath(os.path.join(_HERE, _REL))
    yield os.path.join(_HERE, "enflame-src-030b", _REL)
    yield (
        r"C:\Users\21649\AppData\Local\Temp\opencode\enflame-src-030b"
        "\\" + _REL
    )


ATTENTION_PY = next((p for p in _candidate_paths() if os.path.isfile(p)), None)
if ATTENTION_PY is None:
    pytest.skip(
        "GCU attention.py not found; set FL_GCU_ATTENTION_PY",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Import scaffolding
# ---------------------------------------------------------------------------


def _raising_assertion(*_a, **_k):
    raise AssertionError("DCP is not initialised in tests")


def _not_implemented(name):
    def _fn(*_a, **_k):
        raise NotImplementedError(f"{name} is not available in tests")
    return _fn


def _fake_is_quantized_kv_cache(dtype):
    return isinstance(dtype, str) and dtype.startswith("fp8")


class _FakeLogger:
    def __init__(self):
        self.warnings = []

    def warning_once(self, *a, **k):
        self.warnings.append((a, k))

    def warning(self, *a, **k):
        self.warnings.append((a, k))

    def info(self, *a, **k):
        pass

    def debug(self, *a, **k):
        pass


class _FakeAttentionType:
    DECODER = "DECODER"
    ENCODER = "ENCODER"
    ENCODER_ONLY = "ENCODER_ONLY"
    ENCODER_DECODER = "ENCODER_DECODER"


class _FakeAttentionCGSupport:
    ALWAYS = "ALWAYS"
    NEVER = "NEVER"
    NEVER_IN_CUDA_GRAPH = "NEVER_IN_CUDA_GRAPH"
    UNIFORM_BATCH = "UNIFORM_BATCH"


class _FakeMultipleOf:
    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        return getattr(other, "value", other) == self.value

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return f"MultipleOf({self.value})"


class _FakeAttentionBackend:
    accept_output_buffer = False


class _FakeAttentionImpl:
    pass


class _FakeAttentionMetadataBuilder:
    _cudagraph_support = _FakeAttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec=None, layer_names=None, vllm_config=None,
                 device=None):
        self.kv_cache_spec = kv_cache_spec
        self.layer_names = layer_names
        self.vllm_config = vllm_config
        self.device = device

    def __class_getitem__(cls, item):
        return cls


def _fake_backend_attrs():
    return {
        "AttentionBackend": _FakeAttentionBackend,
        "AttentionCGSupport": _FakeAttentionCGSupport,
        "AttentionImpl": _FakeAttentionImpl,
        "AttentionMetadataBuilder": _FakeAttentionMetadataBuilder,
        "AttentionType": _FakeAttentionType,
        "MultipleOf": _FakeMultipleOf,
    }


_PRE_MODULES = set(sys.modules)
_MISSING = object()
_PARENT_ATTR_BACKUP = {}


def _link(parent_name, child, obj):
    parent = sys.modules.get(parent_name)
    if parent is None:
        return
    if parent_name in _PRE_MODULES:
        key = (parent_name, child)
        if key not in _PARENT_ATTR_BACKUP:
            _PARENT_ATTR_BACKUP[key] = getattr(parent, child, _MISSING)
    setattr(parent, child, obj)


def _install_fake(name, attrs=None):
    attrs = dict(attrs or {})
    parts = name.split(".")
    for i in range(1, len(parts)):
        pkg_name = ".".join(parts[:i])
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = []
            sys.modules[pkg_name] = pkg
            if i > 1:
                _link(".".join(parts[: i - 1]), parts[i - 1], pkg)
    mod = types.ModuleType(name)
    mod.__path__ = []
    for key, value in attrs.items():
        setattr(mod, key, value)
    sys.modules[name] = mod
    if len(parts) > 1:
        _link(".".join(parts[:-1]), parts[-1], mod)
    return mod


def _ensure_module(name, attrs=None):
    attrs = dict(attrs or {})
    try:
        mod = importlib.import_module(name)
    except Exception:
        mod = _install_fake(name, attrs)
    else:
        for key, value in attrs.items():
            if not hasattr(mod, key):
                setattr(mod, key, value)
    return mod


def _force_fake(name, attrs=None):
    return _install_fake(name, attrs)


# vLLM surface used by the target.  Real modules win when importable; only
# missing modules/attributes are backfilled so the target can be imported on
# a host without a GCU stack.
_VLLM_SPECS = [
    ("vllm.config", {
        "VllmConfig": object,
        "get_layers_from_vllm_config": lambda *_a, **_k: {},
    }),
    ("vllm.config.cache", {"CacheDType": str}),
    ("vllm.distributed.parallel_state", {"get_dcp_group": _raising_assertion}),
    ("vllm.logger", {"init_logger": lambda _name: _FakeLogger()}),
    ("vllm.model_executor.determinism.batch_invariant",
     {"_batch_invariant_MODE": False}),
    ("vllm.model_executor.layers.attention.attention",
     {"Attention": type("Attention", (), {})}),
    ("vllm.platforms.interface", {"DeviceCapability": object}),
    ("vllm.utils.math_utils",
     {"cdiv": lambda a, b: (a + b - 1) // b}),
    ("vllm.utils.torch_utils",
     {"is_quantized_kv_cache": _fake_is_quantized_kv_cache}),
    ("vllm.v1.attention.backend", _fake_backend_attrs()),
    ("vllm.v1.attention.backends.utils", {
        "CommonAttentionMetadata": type("CommonAttentionMetadata", (), {}),
        "get_dcp_local_seq_lens": lambda seq_lens, *_a, **_k: seq_lens,
    }),
    ("vllm.v1.attention.ops.dcp",
     {"cp_lse_ag_out_rs": _not_implemented("cp_lse_ag_out_rs")}),
    ("vllm.v1.attention.ops.merge_attn_states",
     {"merge_attn_states": _not_implemented("merge_attn_states")}),
    ("vllm.v1.kv_cache_interface",
     {"AttentionSpec": type("AttentionSpec", (), {})}),
    ("vllm.v1.attention.backends.fa_utils",
     {"get_scheduler_metadata": None}),
    ("vllm.config.vllm",
     {"get_current_vllm_config_or_none": lambda: None}),
]

for _name, _attrs in _VLLM_SPECS:
    _ensure_module(_name, _attrs)

# Anything imported while backfilling vLLM counts as pre-existing from the
# perspective of the vendor-package cleanup below.
_PRE_MODULES = set(sys.modules)

# The Enflame wheel and the GCU reshape helper never exist on a CPU host, so
# their submodules are always replaced with recorders.
_force_fake(
    "flash_attn.vllm_flash_attn",
    {"flash_attn_varlen_func": lambda **_k: None},
)
_force_fake(
    "vllm_fl.dispatch.backends.vendor.gcu.impl.reshape_and_cache",
    {"reshape_and_cache_flash": lambda *_a, **_k: None},
)


def _cleanup_fake_roots(roots):
    """Drop fake trees created here so other test files see the real packages.

    Modules that existed before this file ran are left untouched; only the
    placeholder entries injected for the target's import are removed (the
    loaded module already holds direct references to them).
    """
    for name in list(sys.modules):
        if name in _PRE_MODULES:
            continue
        if any(name == r or name.startswith(r + ".") for r in roots):
            del sys.modules[name]
    for key, old in list(_PARENT_ATTR_BACKUP.items()):
        parent_name, child = key
        if not any(parent_name == r or parent_name.startswith(r + ".")
                   for r in roots):
            continue
        parent = sys.modules.get(parent_name)
        if parent is None:
            del _PARENT_ATTR_BACKUP[key]
            continue
        if old is _MISSING:
            if hasattr(parent, child):
                delattr(parent, child)
        else:
            setattr(parent, child, old)
        del _PARENT_ATTR_BACKUP[key]


def _load_under_test(path):
    spec = importlib.util.spec_from_file_location("covfl_gcu_attention", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


try:
    MOD = _load_under_test(ATTENTION_PY)
except Exception as exc:  # pragma: no cover - host dependency guard
    pytest.skip(f"cannot import GCU attention module: {exc!r}",
                allow_module_level=True)
finally:
    _cleanup_fake_roots(("vllm_fl", "flash_attn"))


AT = MOD.AttentionType
Metadata = MOD.AttentionGCUMetadata
Impl = MOD.AttentionGCUImpl
Builder = MOD.AttentionGCUMetadataBuilder
Backend = MOD.AttentionGCUBackend


def test_import_without_scheduler_helper_falls_back_to_none(monkeypatch):
    """Reload the module with ``fa_utils`` lacking the helper.

    The first import binds whatever ``get_scheduler_metadata`` the host
    provides; this reload forces the ``ImportError`` fallback so the module's
    optional-dependency branch is exercised as well.
    """
    chain = [
        "flash_attn",
        "flash_attn.vllm_flash_attn",
        "vllm_fl",
        "vllm_fl.dispatch",
        "vllm_fl.dispatch.backends",
        "vllm_fl.dispatch.backends.vendor",
        "vllm_fl.dispatch.backends.vendor.gcu",
        "vllm_fl.dispatch.backends.vendor.gcu.impl",
        "vllm_fl.dispatch.backends.vendor.gcu.impl.reshape_and_cache",
        "vllm.v1.attention.backends.fa_utils",
    ]
    for name in chain:
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    for name in chain:
        parent, _, leaf = name.rpartition(".")
        if parent in sys.modules:
            setattr(sys.modules[parent], leaf, sys.modules[name])
    sys.modules["flash_attn.vllm_flash_attn"].flash_attn_varlen_func = (
        lambda **kwargs: None
    )
    sys.modules[
        "vllm_fl.dispatch.backends.vendor.gcu.impl.reshape_and_cache"
    ].reshape_and_cache_flash = lambda *a, **k: None

    spec = importlib.util.spec_from_file_location(
        "covfl_gcu_attention_no_fa_utils", ATTENTION_PY
    )
    reloaded = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, reloaded)
    spec.loader.exec_module(reloaded)

    assert reloaded.get_scheduler_metadata is None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class Recorder:
    def __init__(self, returns=None):
        self.calls = []
        self.returns = returns

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        if isinstance(self.returns, list):
            return self.returns.pop(0) if self.returns else None
        return self.returns

    @property
    def count(self):
        return len(self.calls)

    def last(self):
        assert self.calls, "recorder was not called"
        return self.calls[-1]

    def last_kwargs(self):
        return self.last()[1]


def _mul_value(value):
    if hasattr(value, "value"):
        return value.value
    try:
        return int(value)
    except Exception:
        return value


@pytest.fixture(autouse=True)
def _no_dcp(monkeypatch):
    monkeypatch.setattr(MOD, "get_dcp_group", _raising_assertion)
    yield


def make_vllm_config(*, full=False, max_cg=64, num_heads=8, num_kv=2,
                     head=64, max_num_seqs=4, splits=3, interleave=1,
                     is_hybrid=False, cache_dtype="auto"):
    model_config = SimpleNamespace(
        get_num_attention_heads=lambda parallel: num_heads,
        get_num_kv_heads=lambda parallel: num_kv,
        get_head_size=lambda: head,
        is_hybrid=is_hybrid,
    )
    parallel_config = SimpleNamespace(cp_kv_cache_interleave_size=interleave)
    compilation_config = SimpleNamespace(
        cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: full),
        max_cudagraph_capture_size=max_cg,
    )
    cache_config = SimpleNamespace(
        cache_dtype=cache_dtype,
        mamba_ssm_cache_dtype="auto",
        mamba_cache_dtype="auto",
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        cache_config=cache_config,
        compilation_config=compilation_config,
        scheduler_config=SimpleNamespace(max_num_seqs=max_num_seqs),
        attention_config=SimpleNamespace(
            flash_attn_max_num_splits_for_cuda_graph=splits
        ),
    )


def make_builder(*, full=False, max_cg=64, interleave=1, block_size=16,
                 num_heads=8, num_kv=2, head=64, splits=3, max_num_seqs=4,
                 cache_dtype="auto"):
    spec = SimpleNamespace(dtype=torch.float16, block_size=block_size)
    cfg = make_vllm_config(
        full=full,
        max_cg=max_cg,
        num_heads=num_heads,
        num_kv=num_kv,
        head=head,
        max_num_seqs=max_num_seqs,
        splits=splits,
        interleave=interleave,
        cache_dtype=cache_dtype,
    )
    return Builder(spec, ["layer0"], cfg, torch.device("cpu"))


def make_common(*, query_lens=(2, 3), seq_lens_values=(7, 10), causal=True):
    starts = [0]
    for length in query_lens:
        starts.append(starts[-1] + length)
    num_reqs = len(query_lens)
    return SimpleNamespace(
        num_reqs=num_reqs,
        num_actual_tokens=int(starts[-1]),
        max_query_len=max(query_lens),
        max_seq_len=max(seq_lens_values),
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens_values, dtype=torch.int32),
        block_table_tensor=torch.zeros((num_reqs, 4), dtype=torch.int32),
        slot_mapping=torch.arange(starts[-1], dtype=torch.int64),
        causal=causal,
    )


def make_impl(**over):
    kwargs = dict(
        num_heads=8,
        head_size=64,
        scale=0.125,
        num_kv_heads=2,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AT.DECODER,
        kv_sharing_target_layer_name=None,
        sinks=None,
    )
    kwargs.update(over)
    return Impl(**kwargs)


def make_metadata(*, num_tokens=5, max_query_len=3, qsl=(0, 2, 5),
                  max_seq_len=10, seq_lens=(7, 10), block_rows=2,
                  use_cascade=False, common_prefix_len=0, causal=True,
                  scheduler_metadata=None, max_num_splits=0,
                  cu_prefix_query_lens=None, prefix_kv_lens=None,
                  suffix_kv_lens=None, dcp_context_kv_lens=None,
                  max_dcp_context_kv_len=None):
    return Metadata(
        num_actual_tokens=num_tokens,
        max_query_len=max_query_len,
        query_start_loc=torch.tensor(qsl, dtype=torch.int32),
        max_seq_len=max_seq_len,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        block_table=torch.zeros((block_rows, 4), dtype=torch.int32),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64),
        use_cascade=use_cascade,
        common_prefix_len=common_prefix_len,
        cu_prefix_query_lens=cu_prefix_query_lens,
        prefix_kv_lens=prefix_kv_lens,
        suffix_kv_lens=suffix_kv_lens,
        max_dcp_context_kv_len=max_dcp_context_kv_len,
        dcp_context_kv_lens=dcp_context_kv_lens,
        scheduler_metadata=scheduler_metadata,
        max_num_splits=max_num_splits,
        causal=causal,
    )


def make_forward_inputs(*, num_tokens=5, num_heads=8, num_kv=2, head=64,
                        block=4, num_blocks=8, attn_type=AT.DECODER):
    impl = make_impl(
        num_heads=num_heads,
        head_size=head,
        num_kv_heads=num_kv,
        attn_type=attn_type,
    )
    query = torch.zeros(num_tokens, num_heads, head)
    key = torch.zeros(num_tokens, num_kv, head)
    value = torch.zeros(num_tokens, num_kv, head)
    kv_cache = torch.zeros(num_blocks, num_kv, block, 2 * head)
    output = torch.zeros(num_tokens, num_heads * head)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1),
        _k_scale=torch.ones(1),
        _v_scale=torch.ones(1),
    )
    return impl, layer, query, key, value, kv_cache, output


def _patch_scheduler(monkeypatch, recorder):
    monkeypatch.setattr(MOD, "get_scheduler_metadata", recorder, raising=False)


# ---------------------------------------------------------------------------
# AttentionGCUBackend: static/class dispatch metadata
# ---------------------------------------------------------------------------


def test_backend_basic_metadata():
    assert Backend.accept_output_buffer is True
    assert torch.float16 in Backend.supported_dtypes
    assert torch.bfloat16 in Backend.supported_dtypes
    assert Backend.get_name() == "FLASH_ATTN"
    assert Backend.get_impl_cls() is Impl
    assert Backend.get_builder_cls() is Builder
    assert Backend.supports_sink() is True
    assert Impl.can_return_lse_for_decode is True


@pytest.mark.parametrize(
    "attn_type,expected",
    [
        (AT.DECODER, True),
        (AT.ENCODER, True),
        (AT.ENCODER_ONLY, True),
        (AT.ENCODER_DECODER, True),
        ("UNKNOWN", False),
    ],
)
def test_backend_supports_attn_type(attn_type, expected):
    assert Backend.supports_attn_type(attn_type) is expected


@pytest.mark.parametrize(
    "dtype,expected",
    [
        (None, True),
        ("auto", True),
        ("fp8", True),
        ("fp8_e4m3", True),
        ("int8", False),
        ("fp16", False),
    ],
)
def test_backend_supports_kv_cache_dtype(dtype, expected):
    assert Backend.supports_kv_cache_dtype(dtype) is expected


@pytest.mark.parametrize(
    "head_size,expected",
    [(8, True), (64, True), (128, True), (256, True),
     (264, False), (12, False), (257, False)],
)
def test_backend_supports_head_size(head_size, expected):
    assert Backend.supports_head_size(head_size) is expected


def test_backend_supports_combination_returns_none():
    assert Backend.supports_combination(
        head_size=128,
        dtype=torch.float16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        device_capability=None,
    ) is None


@pytest.mark.parametrize(
    "cfg,expected",
    [
        (None, [16]),
        (SimpleNamespace(model_config=None, cache_config=SimpleNamespace()),
         [16]),
        (
            SimpleNamespace(
                model_config=SimpleNamespace(is_hybrid=True),
                cache_config=SimpleNamespace(
                    mamba_ssm_cache_dtype="float32",
                    mamba_cache_dtype="auto",
                ),
            ),
            [16, 32, 64],
        ),
        (
            SimpleNamespace(
                model_config=SimpleNamespace(is_hybrid=True),
                cache_config=SimpleNamespace(
                    mamba_ssm_cache_dtype="auto",
                    mamba_cache_dtype="float32",
                ),
            ),
            [16, 32, 64],
        ),
        (
            SimpleNamespace(
                model_config=SimpleNamespace(is_hybrid=True),
                cache_config=SimpleNamespace(
                    mamba_ssm_cache_dtype="bfloat16",
                    mamba_cache_dtype="auto",
                ),
            ),
            [16],
        ),
        (
            SimpleNamespace(
                model_config=SimpleNamespace(is_hybrid=False),
                cache_config=SimpleNamespace(
                    mamba_ssm_cache_dtype="float32",
                    mamba_cache_dtype="float32",
                ),
            ),
            [16],
        ),
    ],
)
def test_backend_get_supported_kernel_block_sizes(monkeypatch, cfg, expected):
    monkeypatch.setattr(
        sys.modules["vllm.config.vllm"],
        "get_current_vllm_config_or_none",
        lambda: cfg,
    )
    result = Backend.get_supported_kernel_block_sizes()
    assert [_mul_value(x) for x in result] == expected


# ---------------------------------------------------------------------------
# _get_sliding_window_configs
# ---------------------------------------------------------------------------


def test_get_sliding_window_configs_collects_impl_windows(monkeypatch):
    impl_a = make_impl(sliding_window=4)
    impl_b = make_impl(sliding_window=None)
    monkeypatch.setattr(
        MOD,
        "get_layers_from_vllm_config",
        lambda cfg, attention_cls: {
            "a": SimpleNamespace(impl=impl_a),
            "b": SimpleNamespace(impl=impl_b),
        },
    )
    windows = MOD._get_sliding_window_configs(object())
    assert windows == {(3, 0), (-1, -1)}


# ---------------------------------------------------------------------------
# Metadata builder construction and build()
# ---------------------------------------------------------------------------


def test_builder_init_defaults_and_dcp_fallback():
    builder = make_builder()
    assert builder.dcp_world_size == 1
    assert builder.dcp_rank == 0
    assert builder.aot_schedule is True
    assert builder.aot_sliding_window is None
    assert builder.use_full_cuda_graph is False
    assert builder.max_cudagraph_size == 64
    assert builder.num_heads_q == 8
    assert builder.num_heads_kv == 2
    assert builder.headdim == 64
    assert builder.block_size == 16
    assert builder.kv_cache_dtype is torch.float16
    assert not hasattr(builder, "max_num_splits")


def test_builder_init_full_cudagraph_allocates_scheduler_buffer():
    builder = make_builder(full=True, max_cg=64, max_num_seqs=4, splits=5)
    expected = 1024 + (64 + 1) * 16
    assert tuple(builder.scheduler_metadata.shape) == (expected,)
    assert builder.scheduler_metadata.dtype == torch.int32
    assert builder.max_num_splits == 5


def test_builder_init_uses_dcp_group_when_available(monkeypatch):
    monkeypatch.setattr(
        MOD,
        "get_dcp_group",
        lambda: SimpleNamespace(world_size=4, rank_in_group=2),
    )
    builder = make_builder(interleave=8)
    assert builder.dcp_world_size == 4
    assert builder.dcp_rank == 2
    assert builder.cp_kv_cache_interleave_size == 8


def test_build_plain_metadata_without_prefix(monkeypatch):
    builder = make_builder()
    builder.aot_sliding_window = (-1, -1)
    monkeypatch.setattr(MOD, "get_scheduler_metadata", None)
    common = make_common()

    md = builder.build(0, common, fast_build=False)

    assert isinstance(md, Metadata)
    assert md.num_actual_tokens == common.num_actual_tokens
    assert md.max_query_len == common.max_query_len
    assert md.query_start_loc is common.query_start_loc
    assert md.seq_lens is common.seq_lens
    assert md.block_table is common.block_table_tensor
    assert md.slot_mapping is common.slot_mapping
    assert md.use_cascade is False
    assert md.common_prefix_len == 0
    assert md.cu_prefix_query_lens is None
    assert md.prefix_kv_lens is None
    assert md.suffix_kv_lens is None
    assert md.scheduler_metadata is None
    assert md.prefix_scheduler_metadata is None
    assert md.dcp_context_kv_lens is None
    assert md.max_dcp_context_kv_len == 0
    assert md.max_num_splits == 0
    assert md.causal is True


def test_build_fast_build_skips_aot_probe(monkeypatch):
    builder = make_builder()

    def _fail(_cfg):
        raise AssertionError("fast_build must not probe sliding windows")

    monkeypatch.setattr(MOD, "_get_sliding_window_configs", _fail)
    monkeypatch.setattr(MOD, "get_scheduler_metadata", None)

    md = builder.build(0, make_common(causal=False), fast_build=True)

    assert builder.aot_sliding_window == (-1, -1)
    assert md.scheduler_metadata is None
    assert md.causal is False


def test_build_aot_single_sliding_window_records_scheduler_args(monkeypatch):
    builder = make_builder()
    sched = Recorder(torch.arange(7, dtype=torch.int32))
    _patch_scheduler(monkeypatch, sched)
    monkeypatch.setattr(MOD, "is_quantized_kv_cache", lambda _d: False)
    monkeypatch.setattr(
        MOD, "_get_sliding_window_configs", lambda _cfg: {(4, 4)}
    )

    md = builder.build(0, make_common(), fast_build=False)

    assert builder.aot_sliding_window == (4, 4)
    kwargs = sched.last_kwargs()
    assert kwargs["batch_size"] == 2
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["max_seqlen_k"] == 10
    assert kwargs["num_heads_q"] == 8
    assert kwargs["num_heads_kv"] == 2
    assert kwargs["headdim"] == 64
    assert kwargs["page_size"] == 16
    assert kwargs["causal"] is True
    assert kwargs["window_size"] == (4, 4)
    assert kwargs["num_splits"] == 0
    assert kwargs["qkv_dtype"] is torch.float16
    assert md.scheduler_metadata is not None


def test_build_aot_multiple_sliding_windows_disables_aot(monkeypatch):
    builder = make_builder()
    builder.aot_sliding_window = None
    sched = Recorder(torch.zeros(3, dtype=torch.int32))
    _patch_scheduler(monkeypatch, sched)
    monkeypatch.setattr(
        MOD, "_get_sliding_window_configs", lambda _cfg: {(4, 4), (8, 0)}
    )

    md = builder.build(0, make_common(), fast_build=False)

    assert builder.aot_schedule is False
    assert builder.aot_sliding_window == (-1, -1)
    assert sched.count == 0
    assert md.scheduler_metadata is None


def test_build_aot_single_none_window_keeps_full_window(monkeypatch):
    builder = make_builder()
    builder.aot_sliding_window = None
    sched = Recorder(torch.zeros(3, dtype=torch.int32))
    _patch_scheduler(monkeypatch, sched)
    monkeypatch.setattr(
        MOD, "_get_sliding_window_configs", lambda _cfg: {None}
    )

    builder.build(0, make_common(), fast_build=False)

    assert builder.aot_sliding_window == (-1, -1)
    assert sched.count == 1
    assert sched.last_kwargs()["window_size"] == (-1, -1)


@pytest.mark.parametrize("cache_dtype,quant,expected", [
    ("fp8", True, "fp8"),
    ("auto", False, torch.float16),
])
def test_build_scheduler_cache_dtype_branch(monkeypatch, cache_dtype, quant,
                                            expected):
    builder = make_builder(cache_dtype=cache_dtype)
    builder.aot_sliding_window = (-1, -1)
    sched = Recorder(torch.zeros(3, dtype=torch.int32))
    _patch_scheduler(monkeypatch, sched)
    monkeypatch.setattr(MOD, "is_quantized_kv_cache", lambda _d: quant)

    builder.build(0, make_common(), fast_build=False)

    assert sched.last_kwargs()["qkv_dtype"] == expected


def test_build_dcp_partitions_context(monkeypatch):
    builder = make_builder(interleave=2)
    builder.aot_sliding_window = (-1, -1)
    builder.dcp_world_size = 2
    builder.dcp_rank = 1
    local = Recorder()
    local.returns = None

    def _local(seq_lens, world_size, rank, interleave):
        local(seq_lens, world_size, rank, interleave)
        return seq_lens // 2

    monkeypatch.setattr(MOD, "get_dcp_local_seq_lens", _local)
    sched = Recorder(torch.zeros(3, dtype=torch.int32))
    _patch_scheduler(monkeypatch, sched)

    common = make_common(query_lens=(2, 3), seq_lens_values=(7, 10))
    md = builder.build(4, common, fast_build=False)

    expected_context = common.seq_lens - (common.query_start_loc[1:]
                                          - common.query_start_loc[:-1])
    args = local.last()[0]
    assert torch.equal(args[0], expected_context)
    assert args[1:] == (2, 1, 2)
    assert md.use_cascade is True
    assert torch.equal(md.dcp_context_kv_lens, expected_context // 2)
    assert md.max_dcp_context_kv_len == 6
    assert md.cu_prefix_query_lens is None
    assert md.prefix_kv_lens is None
    assert md.suffix_kv_lens is None
    kwargs = sched.last_kwargs()
    assert kwargs["causal"] is False
    assert kwargs["max_seqlen_k"] == 6
    assert torch.equal(kwargs["cache_seqlens"], expected_context // 2)
    assert kwargs["cu_seqlens_q"] is common.query_start_loc


def test_build_cascade_prefix_and_suffix(monkeypatch):
    builder = make_builder()
    builder.aot_sliding_window = (-1, -1)
    prefix_ret = torch.tensor([11], dtype=torch.int32)
    suffix_ret = torch.tensor([22], dtype=torch.int32)
    sched = Recorder()
    sched.returns = [prefix_ret, suffix_ret]
    _patch_scheduler(monkeypatch, sched)

    common = make_common()
    md = builder.build(4, common, fast_build=False)

    assert md.use_cascade is True
    assert md.common_prefix_len == 4
    assert md.cu_prefix_query_lens.dtype == torch.int32
    assert torch.equal(
        md.cu_prefix_query_lens,
        torch.tensor([0, common.num_actual_tokens], dtype=torch.int32),
    )
    assert torch.equal(md.prefix_kv_lens, torch.tensor([4], dtype=torch.int32))
    assert torch.equal(
        md.suffix_kv_lens, common.seq_lens[: common.num_reqs] - 4
    )
    assert md.prefix_scheduler_metadata is prefix_ret
    assert md.scheduler_metadata is suffix_ret

    assert sched.count == 2
    first = sched.calls[0][1]
    second = sched.calls[1][1]
    assert first["batch_size"] == 1
    assert first["max_seqlen_q"] == common.num_actual_tokens
    assert first["max_seqlen_k"] == 4
    assert torch.equal(first["cache_seqlens"], md.prefix_kv_lens)
    assert first["causal"] is False
    assert second["batch_size"] == 2
    assert second["max_seqlen_q"] == common.max_query_len
    assert second["max_seqlen_k"] == 6
    assert torch.equal(second["cache_seqlens"], md.suffix_kv_lens)
    assert second["causal"] is True


def test_build_full_cudagraph_copies_into_buffer(monkeypatch):
    builder = make_builder(full=True, max_cg=64, splits=3)
    builder.aot_sliding_window = (-1, -1)
    produced = torch.full((5,), 7, dtype=torch.int32)
    sched = Recorder(produced)
    _patch_scheduler(monkeypatch, sched)

    md = builder.build(0, make_common(), fast_build=False)

    assert tuple(md.scheduler_metadata.shape) == (5,)
    assert torch.equal(md.scheduler_metadata, produced)
    assert torch.count_nonzero(builder.scheduler_metadata[5:]) == 0
    assert md.max_num_splits == 3


def test_build_max_num_splits_zero_when_over_capture_size(monkeypatch):
    builder = make_builder(full=True, max_cg=4, splits=3)
    builder.aot_sliding_window = (-1, -1)
    monkeypatch.setattr(MOD, "get_scheduler_metadata", None)

    md = builder.build(0, make_common(), fast_build=False)

    assert make_common().num_actual_tokens == 5
    assert md.max_num_splits == 0


def test_builder_use_cascade_attention_delegates(monkeypatch):
    builder = make_builder()
    rec = Recorder(returns=True)
    monkeypatch.setattr(MOD, "use_cascade_attention", rec)

    assert builder.use_cascade_attention(1, 2, 3) is True
    assert rec.last()[0] == (1, 2, 3)


# ---------------------------------------------------------------------------
# AttentionGCUImpl construction
# ---------------------------------------------------------------------------


def test_impl_init_defaults_and_dcp_fallback():
    impl = make_impl()
    assert impl.num_heads == 8
    assert impl.num_kv_heads == 2
    assert impl.head_size == 64
    assert impl.scale == 0.125
    assert impl.alibi_slopes is None
    assert impl.sliding_window == (-1, -1)
    assert impl.logits_soft_cap == 0
    assert impl.num_queries_per_kv == 4
    assert impl.attn_type is AT.DECODER
    assert impl.vllm_flash_attn_version == 3
    assert impl.supports_quant_query_input is False
    assert impl.dcp_world_size == 1
    assert impl.sinks is None
    assert impl.kv_sharing_target_layer_name is None
    assert impl.batch_invariant_enabled is MOD._bi_mode


def test_impl_init_alibi_and_encoder_only_window():
    impl = make_impl(
        alibi_slopes=[0.1, 0.2, 0.3],
        sliding_window=8,
        attn_type=AT.ENCODER_ONLY,
        logits_soft_cap=30.0,
        kv_sharing_target_layer_name="target",
        sinks=torch.zeros(2),
    )
    assert isinstance(impl.alibi_slopes, torch.Tensor)
    assert impl.alibi_slopes.dtype == torch.float32
    assert impl.sliding_window == (7, 7)
    assert impl.logits_soft_cap == 30.0
    assert impl.kv_sharing_target_layer_name == "target"
    assert impl.sinks.shape == (2,)


def test_impl_init_decoder_sliding_window():
    impl = make_impl(sliding_window=8)
    assert impl.sliding_window == (7, 0)


def test_impl_init_quantized_kv_warns(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(MOD.logger, "warning_once", rec, raising=False)
    monkeypatch.setattr(MOD, "is_quantized_kv_cache", lambda _d: True)

    impl = make_impl(kv_cache_dtype="fp8")

    assert impl.kv_cache_dtype == "fp8"
    assert rec.count == 1


def test_impl_init_uses_dcp_group(monkeypatch):
    monkeypatch.setattr(
        MOD, "get_dcp_group", lambda: SimpleNamespace(world_size=2)
    )
    impl = make_impl()
    assert impl.dcp_world_size == 2


# ---------------------------------------------------------------------------
# AttentionGCUImpl.forward
# ---------------------------------------------------------------------------


def test_forward_requires_output():
    impl, layer, query, key, value, kv_cache, _ = make_forward_inputs()
    with pytest.raises(AssertionError):
        impl.forward(layer, query, key, value, kv_cache, None, output=None)


@pytest.mark.parametrize("which", ["output_scale", "output_block_scale"])
def test_forward_rejects_output_quantization(which):
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs()
    extra = {which: torch.ones(1)}
    with pytest.raises(NotImplementedError):
        impl.forward(layer, query, key, value, kv_cache, None,
                     output=output, **extra)


def test_forward_none_metadata_fills_zero():
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs()
    output.fill_(3)
    result = impl.forward(layer, query, key, value, kv_cache, None,
                          output=output)
    assert result is output
    assert torch.count_nonzero(result) == 0


@pytest.mark.parametrize("attn_type", [AT.ENCODER, AT.ENCODER_ONLY])
def test_forward_encoder_dispatches_without_cache(monkeypatch, attn_type):
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs(
        attn_type=attn_type
    )
    rec = Recorder(returns="ENCODER")
    monkeypatch.setattr(impl, "_forward_encoder_attention", rec)
    md = make_metadata()

    result = impl.forward(layer, query, key, value, kv_cache, md,
                          output=output)

    assert result == "ENCODER"
    args = rec.last()[0]
    assert args[0].shape == (5, 8, 64)
    assert args[1].shape == (5, 2, 64)
    assert args[2].shape == (5, 2, 64)
    assert args[3].shape == (5, 512)
    assert args[4] is md
    assert args[5] is layer


@pytest.mark.parametrize("bad_cache", ["ndim", "last_dim"])
def test_forward_rejects_malformed_kv_cache(bad_cache):
    impl, layer, query, key, value, _, output = make_forward_inputs()
    md = make_metadata()
    if bad_cache == "ndim":
        kv_cache = torch.zeros(8, 2, 128)
    else:
        kv_cache = torch.zeros(8, 2, 4, 64)
    with pytest.raises(ValueError, match="GCU attention requires"):
        impl.forward(layer, query, key, value, kv_cache, md, output=output)


def test_forward_decoder_no_cascade_records_kernel_and_cache(monkeypatch):
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs()
    flash = Recorder()
    reshape = Recorder()
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(MOD, "reshape_and_cache_flash", reshape)
    md = make_metadata()

    result = impl.forward(layer, query, key, value, kv_cache, md,
                          output=output)

    assert result is output
    assert reshape.count == 1
    r_args = reshape.last()[0]
    assert r_args[0] is key
    assert r_args[1] is value
    assert r_args[4] is md.slot_mapping
    assert r_args[5] == "auto"
    assert r_args[6] is layer._k_scale
    assert r_args[7] is layer._v_scale

    assert flash.count == 1
    kwargs = flash.last_kwargs()
    assert torch.equal(kwargs["q"], query[:5])
    assert kwargs["k"].shape == (8, 4, 2, 64)
    assert kwargs["v"].shape == (8, 4, 2, 64)
    assert torch.equal(kwargs["out"], output[:5])
    assert kwargs["cu_seqlens_q"] is md.query_start_loc
    assert kwargs["max_seqlen_q"] == 3
    assert kwargs["seqused_k"] is md.seq_lens
    assert kwargs["max_seqlen_k"] == 10
    assert kwargs["softmax_scale"] == 0.125
    assert kwargs["causal"] is True
    assert kwargs["alibi_slopes"] is None
    assert kwargs["window_size"] == (-1, -1)
    assert kwargs["block_table"] is md.block_table
    assert kwargs["softcap"] == 0
    assert kwargs["scheduler_metadata"] is None
    assert kwargs["fa_version"] == 3
    assert kwargs["num_splits"] == 0
    assert kwargs["s_aux"] is None
    assert kwargs["q_descale"].shape == (2, 2)
    assert kwargs["k_descale"].shape == (2, 2)
    assert kwargs["v_descale"].shape == (2, 2)


def test_forward_decoder_skips_cache_write_when_kv_shared(monkeypatch):
    impl = make_impl(kv_sharing_target_layer_name="shared")
    query = torch.zeros(5, 8, 64)
    key = torch.zeros(5, 2, 64)
    value = torch.zeros(5, 2, 64)
    kv_cache = torch.zeros(8, 2, 4, 128)
    output = torch.zeros(5, 512)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    flash = Recorder()
    reshape = Recorder()
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(MOD, "reshape_and_cache_flash", reshape)

    impl.forward(layer, query, key, value, kv_cache, make_metadata(),
                 output=output)

    assert reshape.count == 0
    assert flash.count == 1


def test_forward_decoder_skips_cache_write_without_key_value(monkeypatch):
    impl, layer, query, _, _, kv_cache, output = make_forward_inputs()
    flash = Recorder()
    reshape = Recorder()
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(MOD, "reshape_and_cache_flash", reshape)

    impl.forward(layer, query, None, None, kv_cache, make_metadata(),
                 output=output)

    assert reshape.count == 0
    assert flash.count == 1


def test_forward_decoder_dcp_path(monkeypatch):
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs()
    impl.dcp_world_size = 2
    rec = Recorder(returns="DCP")
    monkeypatch.setattr(impl, "_forward_with_dcp", rec)
    flash = Recorder()
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)

    md = make_metadata(dcp_context_kv_lens=torch.tensor([5, 7],
                                                       dtype=torch.int32),
                       max_dcp_context_kv_len=6)
    result = impl.forward(layer, query, key, value, kv_cache, md,
                          output=output)

    assert result is output
    assert rec.count == 1
    assert flash.count == 0
    kwargs = rec.last()[1]
    assert kwargs["q_descale"].shape == (2, 2)
    assert kwargs["k_descale"].shape == (2, 2)
    assert kwargs["v_descale"].shape == (2, 2)


def test_forward_decoder_cascade_path(monkeypatch):
    impl, layer, query, key, value, kv_cache, output = make_forward_inputs()
    flash = Recorder()
    reshape = Recorder()
    cascade = Recorder(returns="CASCADE")
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)
    monkeypatch.setattr(MOD, "reshape_and_cache_flash", reshape)
    monkeypatch.setattr(MOD, "cascade_attention", cascade)

    md = make_metadata(
        use_cascade=True,
        common_prefix_len=4,
        cu_prefix_query_lens=torch.tensor([0, 5], dtype=torch.int32),
        prefix_kv_lens=torch.tensor([4], dtype=torch.int32),
        suffix_kv_lens=torch.tensor([3, 6], dtype=torch.int32),
        scheduler_metadata=torch.zeros(3, dtype=torch.int32),
        max_num_splits=2,
    )
    result = impl.forward(layer, query, key, value, kv_cache, md,
                          output=output)

    assert result is output
    assert flash.count == 0
    assert reshape.count == 1
    assert cascade.count == 1
    c_args, kwargs = cascade.last()
    assert torch.equal(c_args[0], output[:5])
    assert torch.equal(c_args[1], query[:5])
    assert kwargs["common_prefix_len"] == 4
    assert kwargs["max_kv_len"] == md.max_seq_len
    assert kwargs["softmax_scale"] == 0.125
    assert kwargs["sliding_window"] == (-1, -1)
    assert kwargs["logits_soft_cap"] == 0
    assert kwargs["fa_version"] == 3
    assert kwargs["q_descale"] is layer._q_scale
    assert kwargs["k_descale"] is layer._k_scale
    assert kwargs["v_descale"] is layer._v_scale
    assert kwargs["s_aux"] is None


# ---------------------------------------------------------------------------
# DCP forward, encoder forward
# ---------------------------------------------------------------------------


def test_forward_with_dcp_merges_context_and_query(monkeypatch):
    impl = make_impl()
    gather = Recorder()
    group = SimpleNamespace(
        all_gather=lambda tensor, dim=0: (gather(tensor, dim=dim) or tensor),
    )
    monkeypatch.setattr(MOD, "get_dcp_group", lambda: group)

    query = torch.zeros(5, 8, 64)
    key = torch.zeros(5, 2, 64)
    value = torch.zeros(5, 2, 64)
    key_cache = torch.zeros(8, 4, 2, 64)
    value_cache = torch.zeros(8, 4, 2, 64)
    output = torch.zeros(5, 8, 64)
    md = make_metadata(
        dcp_context_kv_lens=torch.tensor([5, 7], dtype=torch.int32),
        max_dcp_context_kv_len=6,
    )

    flash_calls = []

    def _flash(**kwargs):
        flash_calls.append(kwargs)
        if len(flash_calls) == 1:
            return torch.zeros(5, 8, 64), torch.zeros(8, 5)
        return torch.zeros(5, 8, 64), torch.zeros(5, 8)

    monkeypatch.setattr(MOD, "flash_attn_varlen_func", _flash)

    cp_seen = {}

    def _cp(out, lse, dcp_group, return_lse=False):
        cp_seen["lse_shape"] = tuple(lse.shape)
        cp_seen["group"] = dcp_group
        cp_seen["return_lse"] = return_lse
        return torch.zeros(5, 8, 64), torch.zeros(8, 5)

    monkeypatch.setattr(MOD, "cp_lse_ag_out_rs", _cp)
    merge = Recorder()
    monkeypatch.setattr(MOD, "merge_attn_states", merge)

    q_descale = torch.ones(1, 1)
    k_descale = torch.ones(1, 1)
    v_descale = torch.ones(1, 1)
    result = impl._forward_with_dcp(
        query, key, value, key_cache, value_cache, output, md,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )

    assert result is None
    assert len(flash_calls) == 2
    context_call, query_call = flash_calls
    assert context_call["causal"] is False
    assert context_call["out"] is None
    assert context_call["seqused_k"] is md.dcp_context_kv_lens
    assert context_call["max_seqlen_k"] == 6
    assert context_call["return_softmax_lse"] is True
    assert context_call["q_descale"] is q_descale
    assert query_call["causal"] is md.causal
    assert query_call["cu_seqlens_k"] is md.query_start_loc
    assert query_call["max_seqlen_k"] == md.max_query_len
    assert query_call["k_descale"] is k_descale
    assert query_call["v_descale"] is v_descale
    assert cp_seen["lse_shape"] == (5, 8)
    assert cp_seen["group"] is group
    assert cp_seen["return_lse"] is True
    assert merge.count == 1
    assert merge.last()[0][0] is output


def test_forward_encoder_rejects_quantized_kv(monkeypatch):
    monkeypatch.setattr(MOD, "is_quantized_kv_cache", lambda _d: False)
    impl = make_impl(kv_cache_dtype="fp8", attn_type=AT.ENCODER)
    query = torch.zeros(5, 8, 64)
    key = torch.zeros(5, 2, 64)
    value = torch.zeros(5, 2, 64)
    output = torch.zeros(5, 512)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    with pytest.raises(NotImplementedError):
        impl._forward_encoder_attention(
            query, key, value, output, make_metadata(), layer
        )


def test_forward_encoder_records_kernel(monkeypatch):
    impl = make_impl(attn_type=AT.ENCODER)
    query = torch.zeros(5, 8, 64)
    key = torch.zeros(5, 2, 64)
    value = torch.zeros(5, 2, 64)
    output = torch.zeros(5, 512)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    flash = Recorder()
    monkeypatch.setattr(MOD, "flash_attn_varlen_func", flash)
    md = make_metadata()

    result = impl._forward_encoder_attention(
        query, key, value, output, md, layer
    )

    assert result is output
    assert flash.count == 1
    kwargs = flash.last_kwargs()
    assert kwargs["q"] is query
    assert kwargs["k"] is key
    assert kwargs["v"] is value
    assert kwargs["out"] is output
    assert kwargs["cu_seqlens_q"] is md.query_start_loc
    assert kwargs["cu_seqlens_k"] is md.query_start_loc
    assert kwargs["max_seqlen_q"] == md.max_query_len
    assert kwargs["max_seqlen_k"] == md.max_query_len
    assert kwargs["causal"] is False
    assert kwargs["fa_version"] == 3
    assert kwargs["q_descale"].shape == (2, 2)
    assert kwargs["k_descale"].shape == (2, 2)
    assert kwargs["v_descale"].shape == (2, 2)


# ---------------------------------------------------------------------------
# use_cascade_attention heuristics
# ---------------------------------------------------------------------------


def test_use_cascade_short_prefix_false():
    assert MOD.use_cascade_attention(
        common_prefix_len=255,
        query_lens=np.ones(16, dtype=np.int32),
        num_query_heads=32,
        num_kv_heads=4,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=1,
    ) is False


@pytest.mark.parametrize(
    "flag_key",
    ["use_alibi", "use_sliding_window", "use_local_attention"],
)
def test_use_cascade_feature_flags_false(flag_key):
    kwargs = dict(
        common_prefix_len=512,
        query_lens=np.ones(16, dtype=np.int32),
        num_query_heads=32,
        num_kv_heads=4,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=1,
    )
    kwargs[flag_key] = True
    assert MOD.use_cascade_attention(**kwargs) is False


def test_use_cascade_too_few_requests_false():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.ones(4, dtype=np.int32),
        num_query_heads=32,
        num_kv_heads=4,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=1,
    ) is False


def test_use_cascade_dcp_false():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.ones(16, dtype=np.int32),
        num_query_heads=32,
        num_kv_heads=4,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=2,
    ) is False


def test_use_cascade_no_flash_decoding_true():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.ones(16, dtype=np.int32),
        num_query_heads=8,
        num_kv_heads=8,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=1,
    ) is True


def test_use_cascade_flash_decoding_loses():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.ones(8, dtype=np.int32),
        num_query_heads=8,
        num_kv_heads=2,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=10 ** 9,
        dcp_world_size=1,
    ) is False


def test_use_cascade_flash_decoding_wins():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.ones(8, dtype=np.int32),
        num_query_heads=8,
        num_kv_heads=2,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=1,
        dcp_world_size=1,
    ) is True


def test_use_cascade_mixed_query_lens_short_circuit():
    assert MOD.use_cascade_attention(
        common_prefix_len=512,
        query_lens=np.array([1, 1, 2, 1, 1, 1, 1, 1], dtype=np.int32),
        num_query_heads=8,
        num_kv_heads=2,
        use_alibi=False,
        use_sliding_window=False,
        use_local_attention=False,
        num_sms=120,
        dcp_world_size=1,
    ) is True


# ---------------------------------------------------------------------------
# cascade_attention
# ---------------------------------------------------------------------------


def _cascade_inputs(*, common_prefix_len=32, block_size=16):
    num_tokens = 5
    output = torch.zeros(num_tokens, 8, 64)
    query = torch.zeros(num_tokens, 8, 64)
    key_cache = torch.zeros(8, block_size, 2, 64)
    value_cache = torch.zeros(8, block_size, 2, 64)
    cu_query_lens = torch.tensor([0, 2, 5], dtype=torch.int32)
    cu_prefix_query_lens = torch.tensor([0, num_tokens], dtype=torch.int32)
    prefix_kv_lens = torch.tensor([common_prefix_len], dtype=torch.int32)
    suffix_kv_lens = torch.tensor([3, 6], dtype=torch.int32)
    block_table = torch.zeros((2, 8), dtype=torch.int32)
    return dict(
        output=output,
        query=query,
        key_cache=key_cache,
        value_cache=value_cache,
        cu_query_lens=cu_query_lens,
        max_query_len=3,
        cu_prefix_query_lens=cu_prefix_query_lens,
        prefix_kv_lens=prefix_kv_lens,
        suffix_kv_lens=suffix_kv_lens,
        max_kv_len=64,
        softmax_scale=0.125,
        alibi_slopes=None,
        sliding_window=(-1, -1),
        logits_soft_cap=0,
        block_table=block_table,
        common_prefix_len=common_prefix_len,
        fa_version=3,
    )


def test_cascade_attention_rejects_alibi():
    kwargs = _cascade_inputs()
    kwargs["alibi_slopes"] = torch.ones(2)
    with pytest.raises(AssertionError):
        MOD.cascade_attention(**kwargs)


def test_cascade_attention_rejects_sliding_window():
    kwargs = _cascade_inputs()
    kwargs["sliding_window"] = (4, 0)
    with pytest.raises(AssertionError):
        MOD.cascade_attention(**kwargs)


def test_cascade_attention_rejects_misaligned_prefix():
    kwargs = _cascade_inputs(common_prefix_len=33)
    with pytest.raises(AssertionError):
        MOD.cascade_attention(**kwargs)


def test_cascade_attention_rejects_zero_blocks():
    kwargs = _cascade_inputs(common_prefix_len=0)
    with pytest.raises(AssertionError):
        MOD.cascade_attention(**kwargs)


def test_cascade_attention_runs_prefix_suffix_and_merge(monkeypatch):
    kwargs = _cascade_inputs()
    prefix_out = torch.zeros(5, 8, 64)
    prefix_lse = torch.zeros(2, 5)
    suffix_out = torch.zeros(5, 8, 64)
    suffix_lse = torch.zeros(2, 5)
    calls = []

    def _flash(**kw):
        calls.append(kw)
        if len(calls) == 1:
            return prefix_out, prefix_lse
        return suffix_out, suffix_lse

    monkeypatch.setattr(MOD, "flash_attn_varlen_func", _flash)
    merge = Recorder()
    monkeypatch.setattr(MOD, "merge_attn_states", merge)

    result = MOD.cascade_attention(**kwargs)

    assert result is None
    assert len(calls) == 2
    prefix_call, suffix_call = calls
    assert prefix_call["causal"] is False
    assert prefix_call["max_seqlen_q"] == 5
    assert prefix_call["max_seqlen_k"] == 32
    assert prefix_call["block_table"].shape[0] == 1
    assert prefix_call["q_descale"] is None
    assert prefix_call["s_aux"] is None
    assert suffix_call["causal"] is True
    assert suffix_call["block_table"].shape == (2, 6)
    assert suffix_call["max_seqlen_k"] == 64 - 32
    assert merge.count == 1
    merge_args = merge.last()[0]
    assert merge_args[0] is kwargs["output"]
    assert merge_args[1] is prefix_out
    assert merge_args[3] is suffix_out


def test_cascade_attention_expands_descaled_tensors(monkeypatch):
    kwargs = _cascade_inputs()
    q_descale = torch.ones(1, 1)
    k_descale = torch.ones(1, 1)
    v_descale = torch.ones(1, 1)
    kwargs.update(q_descale=q_descale, k_descale=k_descale,
                  v_descale=v_descale, s_aux=torch.ones(2))
    calls = []

    def _flash(**kw):
        calls.append(kw)
        return torch.zeros(5, 8, 64), torch.zeros(2, 5)

    monkeypatch.setattr(MOD, "flash_attn_varlen_func", _flash)
    monkeypatch.setattr(MOD, "merge_attn_states", Recorder())

    MOD.cascade_attention(**kwargs)

    prefix_call, suffix_call = calls
    assert prefix_call["q_descale"].shape == (1, 2)
    assert prefix_call["k_descale"].shape == (1, 2)
    assert prefix_call["v_descale"].shape == (1, 2)
    assert suffix_call["q_descale"].shape == (2, 2)
    assert suffix_call["k_descale"].shape == (2, 2)
    assert suffix_call["v_descale"].shape == (2, 2)
    assert prefix_call["s_aux"] is kwargs["s_aux"]
    assert suffix_call["s_aux"] is kwargs["s_aux"]
