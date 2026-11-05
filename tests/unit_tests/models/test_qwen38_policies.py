# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8 integration and contract regressions."""

import __future__

import ast
import importlib.util
import sys
from bisect import bisect_right
from pathlib import Path
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from _qwen38_cpu import ROOT, declarations, sampling_metadata

# Hotspot contracts.

# SPDX-License-Identifier: Apache-2.0


hotspot_VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
hotspot_moe = declarations(hotspot_VENDOR / "moe_launch.py", "qwen38_hotspot_moe")
hotspot_limits_spec = importlib.util.spec_from_file_location(
    "qwen38_hotspot_limits", hotspot_VENDOR / "shape_limits.py"
)
hotspot_limits = importlib.util.module_from_spec(hotspot_limits_spec)
hotspot_limits_spec.loader.exec_module(hotspot_limits)
hotspot_linear = declarations(
    hotspot_VENDOR / "linear_policy.py",
    "qwen38_hotspot_linear",
    {"max_prefill_rows": hotspot_limits.max_prefill_rows},
)
hotspot_qsa = declarations(
    hotspot_VENDOR / "qsa_policy.py",
    "qwen38_hotspot_split",
    names={"qsa_sparse_split_count"},
)
hotspot_workspace = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/ops/qsa.py",
    "qwen38_hotspot_workspace",
    names={"qsa_prepare_split_workspace"},
)


def hotspot__moe_args():
    return dict(
        num_tokens=64,
        w1_shape=(512, 160, 2560),
        w2_shape=(512, 2560, 80),
        topk=10,
        dtype=torch.bfloat16,
        vendor="metax",
        device_name="MetaX C550",
    )


@pytest.mark.parametrize(
    "rows,eligible",
    [
        (1, False),
        (7, False),
        (8, True),
        (64, True),
        (1024, True),
        (1025, False),
        (2048, False),
    ],
)
def test_moe_scope_preserves_assignment_and_base(rows, eligible):
    base = {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "sentinel": 123}
    args = hotspot__moe_args()
    args["num_tokens"] = rows
    g1, g2, selected = hotspot_moe.select_stage_configs(base, **args)
    assert selected is eligible
    assert base == {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "sentinel": 123}
    if eligible:
        # Both GEMMs must consume the same BM16 expert assignment.
        assert g1["BLOCK_SIZE_M"] == g2["BLOCK_SIZE_M"] == 16
        assert (g1["BLOCK_SIZE_N"], g1["BLOCK_SIZE_K"]) == (32, 64)
        assert (g2["BLOCK_SIZE_N"], g2["BLOCK_SIZE_K"]) == (256, 32)
        assert g1["sentinel"] == g2["sentinel"] == 123
    else:
        assert g1 == g2 == base
    g1["sentinel"] = -1
    assert base["sentinel"] == g2["sentinel"] == 123


@pytest.mark.parametrize(
    "override",
    [
        {"vendor": "cuda"},
        {"device_name": "other"},
        {"dtype": torch.float16},
        {"topk": 8},
        {"quantized": True},
        {"has_lora": True},
        {"has_bias": True},
        {"activation": "relu"},
        {"w1_shape": (512, 128, 2560)},
    ],
)
def test_moe_unsupported_contract_retains_complete_config(override):
    base = {"BLOCK_SIZE_M": 32, "num_stages": 5, "SPLIT_K": 2}
    args = hotspot__moe_args()
    args.update(override)
    g1, g2, selected = hotspot_moe.select_stage_configs(base, **args)
    assert not selected
    assert g1 == g2 == base


def hotspot__tensor(shape, *, device="cuda:0", dtype=torch.bfloat16, contiguous=True):
    return NS(
        shape=shape,
        ndim=len(shape),
        device=torch.device(device),
        dtype=dtype,
        is_contiguous=lambda: contiguous,
    )


@pytest.mark.parametrize(
    "rows,eligible",
    [
        (0, False),
        (1, True),
        (64, True),
        (65, False),
        (1023, False),
        (1024, True),
        (2048, True),
        (2049, True),
        (4096, True),
        (4097, True),
        (8192, True),
        (8193, False),
    ],
)
def test_gr_row_boundaries(rows, eligible):
    assert (
        hotspot_linear.supports_native_grdown(
            hotspot__tensor((rows, 10240)), hotspot__tensor((320, 10240))
        )
        is eligible
    )


@pytest.mark.parametrize(
    "override",
    [
        {"device": "cpu"},
        {"dtype": torch.float16},
        {"contiguous": False},
    ],
)
def test_gr_unsupported_input_requires_caller_fallback(override):
    x = hotspot__tensor((64, 10240), **override)
    weight = hotspot__tensor((320, 10240))
    assert not hotspot_linear.supports_native_grdown(x, weight)
    with pytest.raises(ValueError, match="caller"):
        hotspot_linear.native_grdown(x, weight)


def test_qsa_workspace_is_owned_by_layer_and_graph_bucket(monkeypatch):
    query = torch.empty(1, 3, 256, dtype=torch.bfloat16)
    cache = torch.empty(1, 64, 1, 256, dtype=torch.bfloat16)
    first, second = {}, {}
    prepare = hotspot_workspace.qsa_prepare_split_workspace
    _, a = prepare(query, cache, 2051, first, num_splits=16)
    _, reused = prepare(query, cache, 2051, first, num_splits=16)
    _, other = prepare(query, cache, 2051, second, num_splits=16)
    _, bucket = prepare(query, cache, 2051, first, num_splits=8)
    assert a is reused
    assert a[0].data_ptr() != other[0].data_ptr() != bucket[0].data_ptr()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    # Cache hit is reused without allocation; capture-time miss returns None.
    fake_query = hotspot__tensor((1, 3, 256))
    assert prepare(fake_query, cache, 2051, first, num_splits=16)[1] is a
    assert prepare(fake_query, cache, 2051, {}, num_splits=16) == (16, None)


# Optimized contracts.

# SPDX-License-Identifier: Apache-2.0


optimized_VENDOR = (
    ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"
)


def optimized_load_vendor_policy(path, name):
    package = "qwen38_test_vendor_policies"
    if package not in sys.modules:
        module_package = ModuleType(package)
        module_package.__path__ = [str(optimized_VENDOR)]
        sys.modules[package] = module_package
    spec = importlib.util.spec_from_file_location(package + "." + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


optimized_moe = optimized_load_vendor_policy(
    optimized_VENDOR / "mctlass_moe.py", "qwen_review_moe"
)
optimized_rows = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py",
    "qwen_review_packing",
    names={"qsa_pack_row_groups"},
)


def optimized_tensor(shape, dtype=torch.bfloat16):
    return NS(
        shape=shape,
        ndim=len(shape),
        dtype=dtype,
        device=torch.device("cuda:0"),
        is_contiguous=lambda: True,
    )


@pytest.mark.parametrize(
    "m,expected",
    [
        (1, False),
        (15, False),
        (16, True),
        (17, False),
        (30, False),
        (31, True),
        (64, True),
        (65, True),
        (1023, True),
        (1024, True),
        (2048, True),
        (2049, True),
        (4096, True),
        (4097, True),
        (8192, True),
        (8193, False),
    ],
)
def test_closed_moe_bounds_and_quantized_fallback(m, expected):
    args = (
        optimized_tensor((m, 2560)),
        optimized_tensor((512, 160, 2560)),
        optimized_tensor((512, 2560, 80)),
        10,
        "MetaX C550",
        "silu",
    )
    assert (
        optimized_moe.moe_eligible(*args, False, False, False, None, 512, False)
        is expected
    )
    assert not optimized_moe.moe_eligible(*args, True, False, False, None, 512, False)
    assert not optimized_moe.moe_eligible(*args, False, True, False, None, 512, False)


@pytest.mark.parametrize(
    "m,stages", [(16, 2), (32, 1), (64, 1), (1024, 2), (4096, 2), (8192, 2)]
)
def test_closed_moe_preserves_small_m_g2_fallback(monkeypatch, m, stages):
    calls = []
    provider = ModuleType("flaggems_vllm.runtime.backend._metax.ops.mctlass_bf16")
    provider.get_plan = lambda *args, **kwargs: (
        calls.append((args, kwargs)) or NS(block_m=32)
    )
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    x, w1, w2, c1, act, c2 = (
        optimized_tensor((m, 2560)),
        object(),
        object(),
        object(),
        object(),
        object(),
    )
    a, b = optimized_moe.prepare_moe_stages(x, w1, w2, c1, act, c2, eligible=True)
    assert a is not None and (b is not None) == (stages == 2)
    assert len(calls) == stages and calls[0][1] == {"topk": 10}
    if stages == 2:
        assert calls[1][0] == (act, w2, c2) and calls[1][1] == {
            "topk": 1,
            "weighted": True,
        }


def test_linear_preserves_native_fallback_and_bias(monkeypatch):
    calls = []

    class Native:
        def apply(self, layer, x, bias=None):
            calls.append((layer, x, bias))
            return "native"

    module = declarations(
        optimized_VENDOR / "mctlass_linear.py",
        "qwen_review_linear",
        {"UnquantizedLinearMethod": Native},
    )
    obj = module.Qwen38MctlassLinearMethod()
    layer = NS(weight=object())
    x = object()
    monkeypatch.setattr(module, "linear_or_none", lambda *args: "closed")
    assert obj.apply(layer, x) == "closed" and not calls
    assert obj.apply(layer, x, bias=7) == "native" and calls[-1] == (layer, x, 7)
    monkeypatch.setattr(module, "linear_or_none", lambda *args: None)
    assert obj.apply(layer, x) == "native"


def test_sampling_contract_tracks_batch_metadata_and_is_idempotent():
    metadata = NS(top_k=torch.tensor([20, 20]))
    batch = NS(
        num_reqs=2,
        top_k_cpu_tensor=torch.tensor([20, 20]),
        sampling_metadata=metadata,
        _make_sampling_metadata=lambda: metadata,
    )
    sampling_metadata.install_sampling_contract(batch)
    installed = batch._make_sampling_metadata
    sampling_metadata.install_sampling_contract(batch)
    assert batch._make_sampling_metadata is installed
    assert batch._make_sampling_metadata().top_k._fl_all_topk20
    batch.top_k_cpu_tensor[1] = 7
    assert not batch._make_sampling_metadata().top_k._fl_all_topk20
    batch.num_reqs = 1
    assert batch._make_sampling_metadata().top_k._fl_all_topk20


def test_packing_covers_rows_without_crossing_request_or_kv_group():
    starts = [0, 5, 6, 10]
    lengths = [8, 12, 21]
    groups = optimized_rows.qsa_pack_row_groups(starts, lengths, 12)
    assert sorted(x for group in groups for x in group if x >= 0) == list(range(12))
    for group in groups:
        real = [x for x in group if 0 <= x < starts[-1]]
        owners = [bisect_right(starts, x) - 1 for x in real]
        positions = [
            lengths[o] - (starts[o + 1] - starts[o]) + x - starts[o]
            for x, o in zip(real, owners)
        ]
        assert len(set(owners)) <= 1
        assert len({p // 4 for p in positions}) <= 1


def test_native_rng_context_restores_after_an_exception():
    native = NS()
    seen = []
    module = None

    def sample(*args):
        seen.append(module._POSITIVE_NATIVE_NOISE.get())
        raise RuntimeError("native failure")

    native.random_sample = sample
    native.sample_with_exponential_noise = lambda *args: "native_noise"
    native.apply_top_k_top_p_pytorch = lambda *args: "native_filter"
    module = declarations(
        ROOT / "vllm_fl/dispatch/backends/vendor/metax/patches/topk_topp_sampler.py",
        "qwen_review_sampling",
        {"topk_topp_sampler": native, "fl_envs": NS(VLLM_FL_USE_FLAGGEMS_VLLM=True)},
    )
    with pytest.raises(RuntimeError, match="native failure"):
        module._random_sample(optimized_tensor((64, 248320), torch.float32), {})
    assert seen == [True] and not module._POSITIVE_NATIVE_NOISE.get()
    assert module._sample_with_exponential_noise(object(), object()) == "native_noise"


def test_packing_metadata_cache_is_scoped_and_cleared_on_error(monkeypatch):
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_packing.py"
    spec = importlib.util.spec_from_file_location("qwen_review_pack_cache", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    class Rows:
        def to(self, *args, **kwargs):
            return self

    def make(*args, **kwargs):
        calls.append(1)
        return Rows()

    # Only the metadata upload is substituted; the real grouping and ContextVar run.
    monkeypatch.setattr(module, "torch", NS(tensor=make, int32=torch.int32))
    starts = torch.tensor([0, 3])
    lengths = torch.tensor([5])
    with (
        pytest.raises(RuntimeError, match="scope"),
        module.qsa_metadata_build_scope(),
    ):
        a = module._cached_pack_rows(starts, lengths, 3, "cpu")
        assert module._cached_pack_rows(starts, lengths, 3, "cpu") is a
        raise RuntimeError("scope")
    assert module._QSA_PACK_ROWS_CACHE.get() is None
    with module.qsa_metadata_build_scope():
        assert module._cached_pack_rows(starts, lengths, 3, "cpu") is not a
    assert len(calls) == 2


def test_runner_import_resolves_real_ple_state_without_gpu_initialization():
    package = ModuleType("qwen_review_worker_package")
    package.__path__ = [str(ROOT / "vllm_fl/worker")]
    sys.modules[package.__name__] = package
    from importlib import import_module

    module = import_module(package.__name__ + ".qwen3_8_flash_next")
    assert module._mrv1_prepare.__module__.endswith("qwen3_8_flash_next_ple")
    assert module.install_sampling_contract.__module__.endswith(
        "qwen3_8_flash_next_sampling"
    )
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("rows,columns", [(64, 2048), (8192, 17600)])
def test_packed_attention_forwards_the_complete_call(monkeypatch, rows, columns):
    module = optimized_load_vendor_policy(
        optimized_VENDOR / "qsa_packing.py", "qwen_forwarding"
    )
    seen = []
    provider = ModuleType("flaggems_vllm.runtime.backend._metax.ops.qsa_packed")

    class Candidate:
        def __init__(self, query, *args, **kwargs):
            assert query is q
            assert kwargs["collect_metadata"] is False

        def __call__(self, *args, **kwargs):
            seen.append((args, kwargs))

    provider.Candidate = Candidate
    monkeypatch.setitem(sys.modules, provider.__name__, provider)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    q = optimized_tensor((rows, 3, 256))
    q.stride = lambda dim: 1 if dim == 2 else 1536
    k = optimized_tensor((1, 400, 1, 256))
    indices, out, gate = (
        optimized_tensor((rows, 2051)),
        object(),
        optimized_tensor(q.shape),
    )
    assert module.packed_attention(
        q, k, k, indices, None, None, 0.0625, out, gate, {}, score_columns=columns
    )
    assert seen == [((q, k, k, indices, None, None, out, gate), {})]


@pytest.mark.parametrize(
    "m,columns", [(8193, 2048), (4096, 17601), (8192, 17601), (4096, None)]
)
def test_extended_qsa_retains_unvalidated_length_and_row_fallback(m, columns):
    module = optimized_load_vendor_policy(
        optimized_VENDOR / "qsa_packing.py", "qwen_mbt_qsa_fallback"
    )
    q = optimized_tensor((m, 3, 256))
    q.stride = lambda dim: 1 if dim == 2 else 1536
    k = optimized_tensor((1, 400, 1, 256))
    assert not module.packed_attention(
        q,
        k,
        k,
        optimized_tensor((m, 2051)),
        None,
        None,
        0.0625,
        object(),
        optimized_tensor(q.shape),
        {},
        score_columns=columns,
    )


# Buffers contracts.

# SPDX-License-Identifier: Apache-2.0


buffers_VLLM = Path(importlib.util.find_spec("vllm").origin).parent


def buffers_declaration(path, name, namespace):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    ns = dict(namespace)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        ns,
    )
    return ns[name]


def buffers_config():
    parallel = SimpleNamespace(
        tp_size=8,
        pcp_size=1,
        dp_size=1,
        ep_size=1,
        sp_size=1,
        use_all2all_kernels=False,
    )
    return SimpleNamespace(
        moe_parallel_config=parallel,
        in_dtype=torch.bfloat16,
        hidden_dim=2560,
        intermediate_size_per_partition=80,
        num_experts=512,
        num_local_experts=512,
        experts_per_token=10,
        activation=SimpleNamespace(value="silu"),
        has_bias=False,
        is_lora_enabled=False,
    )


def buffers_supported():
    return buffers_declaration(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/mctlass_moe.py",
        "direct_outputs_supported",
        {"torch": torch},
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_dim", 128),
        ("intermediate_size_per_partition", 160),
        ("in_dtype", torch.float16),
        ("num_experts", 256),
        ("num_local_experts", 64),
        ("experts_per_token", 8),
        ("has_bias", True),
        ("is_lora_enabled", True),
        ("tp_size", 4),
        ("pcp_size", 2),
        ("dp_size", 2),
        ("ep_size", 8),
        ("sp_size", 2),
        ("use_all2all_kernels", True),
    ],
)
def test_direct_buffers_reject_unvalidated_config(field, value):
    cfg = buffers_config()
    quant = SimpleNamespace(quant_dtype=None, gemm1_clamp_limit=None)
    assert buffers_supported()(cfg, quant, "MetaX C550")
    setattr(
        cfg.moe_parallel_config if hasattr(cfg.moe_parallel_config, field) else cfg,
        field,
        value,
    )
    assert not buffers_supported()(cfg, quant, "MetaX C550")


@pytest.mark.parametrize(
    "device,quant,clamp",
    [
        ("other", None, None),
        ("C550", torch.int8, None),
        ("C550", None, 7.0),
    ],
)
def test_direct_buffers_reject_device_quant_clamp(device, quant, clamp):
    assert not buffers_supported()(
        buffers_config(),
        SimpleNamespace(quant_dtype=quant, gemm1_clamp_limit=clamp),
        device,
    )


@pytest.mark.parametrize("strided", [False, True])
def test_silu_out_passes_original_views_and_supplied_destination(monkeypatch, strided):
    name = "flaggems_vllm.ops.silu_and_mul"
    module = ModuleType(name)
    calls = []

    def out(a, b, destination):
        calls.append((a, b, destination))
        destination.copy_((torch.nn.functional.silu(a.float()) * b.float()).to(a.dtype))
        return destination

    module.silu_and_mul_out = out
    monkeypatch.setitem(sys.modules, name, module)
    fn = buffers_declaration(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/mctlass_moe.py",
        "silu_activation_out",
        {},
    )
    x = torch.randn(7, 320 if strided else 160, dtype=torch.bfloat16)
    if strided:
        x = x[:, ::2]
    y = torch.empty(7, 80, dtype=x.dtype)
    assert fn(x, y) is y and calls[0][2] is y
    assert calls[0][0].untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(
        y,
        (torch.nn.functional.silu(x[:, :80].float()) * x[:, 80:].float()).to(x.dtype),
        atol=0,
        rtol=0,
    )


# Projection contracts.

# SPDX-License-Identifier: Apache-2.0


def projection__leaf(monkeypatch, name, filename):
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_residual_default_and_injected_projection_match(monkeypatch):
    residual = projection__leaf(
        monkeypatch, "_qwen38_residual_projection_test", "common/hyperconnection.py"
    )
    config = residual.HyperConnectionConfig(
        hidden_size=2,
        hc_count=4,
        hc_lowrank=2,
        params_dtype=torch.bfloat16,
        use_fused_ops=False,
    )
    plain = residual.GatedResidualSimple(config, use_mix=False)
    calls = []

    def projection(x, weight):
        value = F.linear(x, weight)
        calls.append((x, weight, value.dtype))
        return value

    injected = residual.GatedResidualSimple(
        config, use_mix=False, small_projection=projection
    )
    injected.block_inject_weight.weight = plain.block_inject_weight.weight
    hyper = torch.randn(3, 8, dtype=torch.bfloat16)
    normalized = torch.randn_like(hyper)
    block = torch.randn(3, 2, dtype=torch.bfloat16)
    expected = plain.combine(block, (hyper, normalized, None))
    actual = injected.combine(block, (hyper, normalized, None))
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(calls) == 1
    assert calls[0][0] is normalized
    assert calls[0][1] is plain.block_inject_weight.weight
    assert calls[0][2] == torch.bfloat16


def test_gate_helper_preserves_weight_loader_and_shared_alias(monkeypatch):
    class Unquantized:
        def create_weights(self):
            return "original loader"

        def process_weights_after_loading(self):
            return "original post-load"

    linear_module = ModuleType("vllm.model_executor.layers.linear")
    linear_module.UnquantizedLinearMethod = Unquantized
    dispatch_module = ModuleType("vllm_fl.dispatch")
    dispatch_module.resolve_op = lambda name: F.linear
    monkeypatch.setitem(sys.modules, "vllm.model_executor.layers.linear", linear_module)
    monkeypatch.setitem(sys.modules, "vllm_fl.dispatch", dispatch_module)
    helpers = projection__leaf(
        monkeypatch, "_qwen38_projection_helper_test", "projections.py"
    )
    gate = torch.nn.Module()
    gate.weight = torch.nn.Parameter(torch.randn(1, 8, dtype=torch.bfloat16))
    gate.weight.weight_loader = object()
    gate.bias = None
    gate.quant_method = Unquantized()
    shared = SimpleNamespace(gate=gate)
    weight, loader = gate.weight, gate.weight.weight_loader

    returned = helpers.use_qwen38_shared_expert_projection(gate, F.linear)
    assert returned is gate and shared.gate is gate
    assert gate.weight is weight and gate.weight.weight_loader is loader
    assert gate.quant_method.create_weights() == "original loader"
    assert gate.quant_method.process_weights_after_loading() == "original post-load"
    assert (
        helpers.Qwen38SmallProjectionMethod.create_weights is Unquantized.create_weights
    )
    assert helpers.use_qwen38_shared_expert_projection(gate) is gate
    x = torch.randn(3, 8, dtype=torch.bfloat16)
    torch.testing.assert_close(
        gate.quant_method.apply(gate, x), F.linear(x, weight), atol=0, rtol=0
    )


# Chains contracts.

# SPDX-License-Identifier: Apache-2.0


def chains_fn(path, name, ns):
    node = next(
        n
        for n in ast.walk(ast.parse(path.read_text()))
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    scope = dict(ns)
    exec(
        compile(
            ast.Module(body=[node], type_ignores=[]),
            str(path),
            "exec",
            flags=__future__.annotations.compiler_flag,
        ),
        scope,
    )
    return scope[name]


@pytest.mark.parametrize(
    "active,padded,capability",
    [(3, 3, True), (3, 5, True), (0, 5, True), (3, 5, False)],
)
@pytest.mark.parametrize("native", [False, True])
def test_qsa_zero_elision_requires_provider_and_preserves_padding(
    monkeypatch, active, padded, capability, native
):
    module = ModuleType("vllm_fl.models.qwen3_8_flash_next.gpu.ops.qsa")
    module.qsa_prepare_split_workspace = lambda *a, **k: (1, None)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    cache = torch.empty(1, 16, 1, 256, dtype=torch.bfloat16)
    indices = torch.zeros(active, 2051, dtype=torch.int32)

    class Indexer:
        raw_key_cache = SimpleNamespace(prefix="raw")
        compressed_key_cache = SimpleNamespace(prefix="compressed")
        output_width = 2051

        def __call__(self, *args):
            return indices

    main = SimpleNamespace(
        num_actual_tokens=active,
        slot_mapping=torch.full((active,), -1),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
    )
    raw = SimpleNamespace(
        num_actual_tokens=active, token_to_req=torch.zeros(active, dtype=torch.int32)
    )
    context = SimpleNamespace(
        attn_metadata={
            "main": main,
            "raw": raw,
            "compressed": SimpleNamespace(score_columns=2048),
        }
    )
    calls = []

    def attention(q, k, v, idx, table, mapping, scale, out, **kw):
        assert torch.isnan(out).all() if capability else torch.all(out == 0)
        out.fill_(3)
        assert kw["prefer_native"] is native
        calls.append(active)

    if capability:
        attention._qwen38_full_output = True
    obj = SimpleNamespace(
        layer_name="main",
        kv_cache=torch.empty(1),
        indexer=Indexer(),
        _metax=True,
        num_kv_heads=1,
        head_dim=256,
        _qsa_store_cache_rows=lambda *a: None,
        _qsa_split_workspaces={},
        _qsa_sparse_split_count=lambda *a: 1,
        _qsa_sparse_paged_attention=attention,
        _qsa_packed_workspaces={},
        _qsa_long_context=True,
        _qsa_native_attention=native,
        topk_indices_buffer=indices,
        scaling=0.0625,
    )
    method = chains_fn(
        ROOT / "vllm_fl/models/qwen3_8_flash_next/gpu/qsa.py",
        "_run_qsa",
        {
            "torch": torch,
            "cast": lambda t, x: x,
            "QSAForwardMetadata": object,
            "get_forward_context": lambda: context,
            "_unpack_qsa_kv_cache": lambda *a: (cache, cache),
            "__package__": "vllm_fl.models.qwen3_8_flash_next.gpu",
        },
    )
    q = torch.ones(padded, 3, 256, dtype=torch.bfloat16)
    kv = torch.ones(padded, 1, 256, dtype=torch.bfloat16)
    out = torch.full_like(q, float("nan"))
    method(obj, torch.empty(padded, 2560), torch.arange(padded), q, kv, kv, q, out)
    assert torch.all(out[:active] == 3) and torch.all(out[active:] == 0)
    assert calls == ([active] if active else [])


@pytest.mark.parametrize("packed", [False, True])
def test_vendor_fallback_initializes_output_before_call(monkeypatch, packed):
    package = "vllm_fl.dispatch.backends.vendor.metax.impl.qwen3_8_flash_next"
    module = ModuleType(package + ".qsa_packing")

    def packed_call(*args, **kw):
        if packed:
            args[7].fill_(5)
        return packed

    module.packed_attention = packed_call
    monkeypatch.setitem(sys.modules, module.__name__, module)

    def portable_call(*args, **kw):
        assert torch.all(args[7] == 0)
        args[7].fill_(7)

    method = chains_fn(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/qsa_policy.py",
        "qsa_sparse_paged_attention",
        {
            "torch": torch,
            "portable": SimpleNamespace(qsa_sparse_paged_attention=portable_call),
            "__package__": package,
        },
    )
    x = torch.ones(2, 3, 256)
    out = torch.full_like(x, float("nan"))
    method(x, x, x, None, None, None, 0.0625, out, packed_workspace_cache={})
    assert torch.all(out == (5 if packed else 7))


def test_hc_callback_and_none_fallback_preserve_checkpoint_and_values():
    path = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/hyperconnection.py"
    hc = declarations(
        path,
        "_hc_small_callback_cpu",
        {"F": torch.nn.functional, "can_use_hc_combine_norm_triton": None},
        names={
            "HyperConnectionConfig",
            "GroupedGemmaRMSNorm",
            "HyperConnectionBase",
            "GatedResidualSimple",
        },
    )
    cfg = hc.HyperConnectionConfig(
        hc_count=4,
        hidden_size=8,
        hc_lowrank=3,
        hc_per_branch_norm=True,
        use_fused_ops=False,
        params_dtype=torch.float32,
    )
    calls = []

    def activation(x, groups):
        calls.append((x.shape, groups))
        return torch.nn.functional.silu(x / groups)

    base = hc.GatedResidualSimple(cfg)
    new = hc.GatedResidualSimple(cfg, down_activation=activation)
    new.load_state_dict(base.state_dict())
    assert list(base.state_dict()) == list(new.state_dict())
    x = torch.randn(5, 32)
    block = torch.randn(5, 8)

    def run(mod):
        y, state = mod.mix(x)
        return y, mod.combine(block, state)

    with torch.no_grad():
        expected = run(base)
        actual = run(new)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
        new._down_activation = lambda *args: None
        fallback = run(new)
        assert all(torch.equal(a, b) for a, b in zip(expected, fallback))
    assert calls == [(torch.Size([5, 3]), 4)]


def test_tuning_readers_do_not_share_mutable_configuration():
    from _qwen38_cpu import _limits

    first = _limits.load_tuning_config()
    second = _limits.load_tuning_config()
    first["prefill_max_rows"] = -1
    first["hc_small_ops"]["norm"].clear()
    assert second["prefill_max_rows"] == _limits.max_prefill_rows() == 8192
    assert second["hc_small_ops"]["norm"]
