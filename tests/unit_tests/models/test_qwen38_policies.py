# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026 FlagOS Contributors
"MetaX policy guards, QSA ownership and adapter contracts."

import ast
import importlib.util
import sys
import weakref
from bisect import bisect_right
from contextlib import contextmanager
from functools import lru_cache
from inspect import signature
from types import ModuleType, SimpleNamespace, SimpleNamespace as NS

import pytest
import torch
import torch.nn.functional as F
from _qwen38_cpu import ROOT, declarations, sampling_metadata

hotspot_VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"


hotspot_limits_spec = importlib.util.spec_from_file_location(
    "qwen38_hotspot_limits", hotspot_VENDOR / "shape_limits.py"
)


hotspot_limits = importlib.util.module_from_spec(hotspot_limits_spec)


hotspot_limits_spec.loader.exec_module(hotspot_limits)


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
    assert optimized_moe.direct_outputs_supported(cfg, quant, "MetaX C550")
    setattr(
        cfg.moe_parallel_config if hasattr(cfg.moe_parallel_config, field) else cfg,
        field,
        value,
    )
    assert not optimized_moe.direct_outputs_supported(cfg, quant, "MetaX C550")


@pytest.mark.parametrize(
    "device,quant,clamp",
    [
        ("other", None, None),
        ("C550", torch.int8, None),
        ("C550", None, 7.0),
    ],
)
def test_direct_buffers_reject_device_quant_clamp(device, quant, clamp):
    assert not optimized_moe.direct_outputs_supported(
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
    fn = optimized_moe.silu_activation_out
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

    method = declarations(
        ROOT
        / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next/qsa_policy.py",
        "qwen_fallback_contract",
        namespace={
            "torch": torch,
            "portable": SimpleNamespace(qsa_sparse_paged_attention=portable_call),
            "__package__": package,
        },
        names={"qsa_sparse_paged_attention"},
    ).qsa_sparse_paged_attention
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


native_VENDOR = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/qwen3_8_flash_next"


native_COMMON = ROOT / "vllm_fl/models/qwen3_8_flash_next/common"


native_packing = declarations(
    native_COMMON / "qsa_packing.py",
    "qsa_contract_packing",
    namespace={"contextmanager": contextmanager},
)


native_workspace = declarations(
    native_COMMON / "qsa_workspace.py",
    "qsa_contract_workspace",
    namespace={
        "arange": torch.arange,
        "_QSA_PACK_ROWS_CACHE": native_packing._QSA_PACK_ROWS_CACHE,
    },
)


def native_load_capabilities(importer, get_device_name):
    return declarations(
        native_VENDOR / "qsa_native.py",
        "qsa_contract_native",
        namespace={
            "lru_cache": lru_cache,
            "signature": signature,
            "import_module": importer,
            "torch": NS(cuda=NS(get_device_name=get_device_name)),
        },
    )


def native_paged_api(q=None, *, block_table=None, softmax_scale=None, causal=False):
    pass


def test_native_capability_checks_actual_extension_and_device():
    calls = []
    modules = {
        "flash_attn": NS(flash_attn_varlen_func=native_paged_api),
        "flash_attn_2_cuda": NS(varlen_fwd=lambda: None),
    }

    def importer(name):
        calls.append(name)
        return modules[name]

    cap = native_load_capabilities(
        importer, lambda d: "C550" if d.index == 0 else "other"
    )
    assert cap.native_available(torch.device("cuda:0"))
    assert not cap.native_available(torch.device("cuda:1"))
    assert not cap.native_available(torch.device("cpu"))
    assert calls == ["flash_attn", "flash_attn_2_cuda"]


@pytest.mark.parametrize(
    "kind",
    ["package_missing", "symbol_missing", "wrong_signature", "extension_missing"],
)
def test_missing_native_capability_is_detected_before_execution(kind):
    def importer(name):
        if kind == "package_missing":
            raise ImportError("missing optional dependency")
        if name == "flash_attn":
            if kind == "symbol_missing":
                return NS()
            return NS(
                flash_attn_varlen_func=(lambda q: None)
                if kind == "wrong_signature"
                else native_paged_api
            )
        return NS() if kind == "extension_missing" else NS(varlen_fwd=lambda: None)

    cap = native_load_capabilities(importer, lambda device: "C550")
    assert not cap.native_available(torch.device("cuda:0"))


def test_workspace_shared_only_in_same_metadata_scope():
    with native_packing.qsa_metadata_build_scope():
        one = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        two = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        other_shape = native_workspace.prepare_native_qsa_workspace(
            18, torch.device("cpu")
        )
        assert one is two
        assert one is not other_shape
        assert one["table"].shape == (17, 513)
        assert one["query"].shape == (17, 3, 256)
        assert one["query"].dtype == torch.bfloat16
        assert one["cuq"].tolist() == list(range(18))
    with native_packing.qsa_metadata_build_scope():
        fresh = native_workspace.prepare_native_qsa_workspace(17, torch.device("cpu"))
        assert fresh["query"].data_ptr() != one["query"].data_ptr()
    with (
        pytest.raises(RuntimeError, match="scope failure"),
        native_packing.qsa_metadata_build_scope(),
    ):
        native_packing._QSA_PACK_ROWS_CACHE.set({"probe": object()})
        raise RuntimeError("scope failure")
    assert native_packing._QSA_PACK_ROWS_CACHE.get() is None


def test_workspace_rejects_first_allocation_inside_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        native_workspace,
        "allocate_qsa_workspace",
        lambda *args: pytest.fail("must reject before allocating"),
    )
    with pytest.raises(RuntimeError, match="before graph capture"):
        native_workspace.prepare_native_qsa_workspace(8, torch.device("cuda:0"))


def test_native_guard_preserves_scale_gate_and_output_contract():
    guard = declarations(
        native_VENDOR / "qsa_native.py",
        "qsa_native_eligibility",
        names={"native_eligible"},
    ).native_eligible
    device = torch.device("cuda:0")

    def tensor(shape, strides, dtype=torch.bfloat16):
        return NS(
            shape=shape,
            dtype=dtype,
            device=device,
            stride=lambda axis=None: strides if axis is None else strides[axis],
        )

    q = tensor((8, 3, 256), (768, 256, 1))
    k = tensor((10, 400, 1, 256), (102400, 256, 256, 1))
    idx = tensor((8, 2051), (2051, 1), torch.int32)
    good = guard(q, k, k, idx, 0.0625, q, q, {})
    assert good
    assert not guard(q, k, k, idx, 0.125, q, q, {})
    assert not guard(q, k, k, idx, 0.0625, q, None, {})
    fp32 = tensor(q.shape, q.stride(), torch.float32)
    assert not guard(q, k, k, idx, 0.0625, q, fp32, {})
    assert not guard(q, k, k, idx, 0.0625, fp32, q, {})
    assert not guard(q, k, k, idx, 0.0625, q, q, None)


def test_capture_workspace_lifetime_is_owned_by_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: NS(cuda_stream=13))
    owner = {}
    query = NS(shape=(8, 3, 256), device=torch.device("cuda:0"), dtype=torch.bfloat16)
    first, second = {"buffer": object()}, {"buffer": object()}
    assert (
        native_workspace.retain_native_graph_workspace(owner, first, query, 2048)
        is first
    )
    assert (
        native_workspace.retain_native_graph_workspace(owner, second, query, 2048)
        is first
    )
    assert (
        native_workspace.retain_native_graph_workspace(owner, second, query, 8192)
        is second
    )
    assert len(owner) == 2
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    eager = {}
    assert (
        native_workspace.retain_native_graph_workspace(owner, eager, query, 2048)
        is eager
    )
    assert len(owner) == 2


groups_packing = native_packing


def test_indexer_groups_preserve_request_boundaries_and_step_lifetime():
    starts = torch.tensor([0, 3, 3, 20], dtype=torch.int32)
    with groups_packing.qsa_metadata_build_scope():
        rows = groups_packing._cached_indexer_pack_rows(starts, 20, torch.device("cpu"))
        assert rows is groups_packing._cached_indexer_pack_rows(
            starts.clone(), 20, torch.device("cpu")
        )
        active = rows[rows >= 0]
        assert active.tolist() == list(range(20))
        assert rows.shape == (3, 16)
        for group in rows.tolist():
            requests = {0 if x < 3 else 2 for x in group if x >= 0}
            assert len(requests) <= 1
    with groups_packing.qsa_metadata_build_scope():
        fresh = groups_packing._cached_indexer_pack_rows(
            starts, 20, torch.device("cpu")
        )
        assert fresh is not rows
        assert torch.equal(fresh, rows)


def test_indexer_groups_include_padded_rows_without_gpu_readback():
    rows = groups_packing._cached_indexer_pack_rows(
        torch.tensor([0, 3, 3, 20]), 24, torch.device("cpu")
    )
    assert rows[rows >= 0].tolist() == list(range(24))
    with pytest.raises(ValueError, match="already be on CPU"):
        groups_packing._cached_indexer_pack_rows(
            torch.empty(3, device="meta"), 24, torch.device("cpu")
        )


@pytest.mark.parametrize(
    "starts,tokens,pack",
    [([], 0, 16), ([1, 2], 2, 16), ([0, 4, 3], 4, 16), ([0, 5], 4, 16), ([0, 4], 4, 4)],
)
def test_indexer_rejects_invalid_request_boundaries(starts, tokens, pack):
    with pytest.raises(ValueError):
        groups_packing._cached_indexer_pack_rows(
            torch.tensor(starts, dtype=torch.int32), tokens, torch.device("cpu"), pack
        )


def test_indexer_empty_batch_is_valid():
    assert (
        groups_packing._cached_indexer_pack_rows(
            torch.tensor([0]), 0, torch.device("cpu")
        )
        is None
    )


width_width = declarations(
    ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py",
    "qwen38_width_cpu",
    namespace={"PreparedQSAForwardMetadata": NS},
    names={
        "qsa_column_buckets",
        "qsa_score_columns",
        "qsa_graph_columns",
        "qsa_cudagraph_key",
    },
)


@pytest.mark.parametrize(
    ("length", "columns"),
    [
        (0, 512),
        (1024, 512),
        (2051, 512),
        (2052, 2048),
        (8195, 2048),
        (8196, 8192),
        (32771, 8192),
        (32772, 17600),
        (69632, 17600),
    ],
)
def test_complete_group_boundaries(length, columns):
    assert width_width.qsa_score_columns(length, 4, 17600) == columns


def test_capacity_guard_and_page_alignment():
    assert width_width.qsa_column_buckets(17600) == (512, 2048, 8192, 17600)
    assert width_width.qsa_column_buckets(1600) == (512, 1600)
    with pytest.raises(ValueError, match="capacity"):
        width_width.qsa_score_columns(70404, 4, 17600)
    for length in range(0, 70401, 127):
        assert length // 4 <= width_width.qsa_score_columns(length, 4, 17600) <= 17600


def test_graph_key_separates_lengths_without_changing_token_descriptor():
    descriptor = (8, 4, True)
    one = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=512)})
    two = NS(attn_metadata={"qsa": NS(compress_ratio=4, score_columns=2048)})
    assert width_width.qsa_cudagraph_key(
        descriptor, one
    ) != width_width.qsa_cudagraph_key(descriptor, two)
    assert width_width.qsa_cudagraph_key(descriptor, NS(attn_metadata={})) == descriptor
    assert (
        width_width.qsa_cudagraph_key(descriptor, NS(attn_metadata=None)) == descriptor
    )


def test_graph_key_rejects_inconsistent_layer_widths():
    with pytest.raises(ValueError, match="disagree"):
        width_width.qsa_graph_columns(
            {
                "a": NS(compress_ratio=4, score_columns=512),
                "b": NS(compress_ratio=4, score_columns=2048),
            }
        )


binding_SOURCE = ROOT / "vllm_fl/models/qwen3_8_flash_next/common/qsa_cache.py"


def binding_cache_type():
    tree = ast.parse(binding_SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "QSAKeyStateCache"
    )

    class StateCache:
        def __init__(self, head_size, **kwargs):
            self.head_size = head_size
            self.kv_cache = torch.tensor([])

        def bind_kv_cache(self, tensor):
            self.kv_cache = tensor

    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(code)
    scope = {"torch": torch, "_QSAStateCache": StateCache}
    exec(compile(code, str(binding_SOURCE), "exec"), scope)
    return scope["QSAKeyStateCache"]


def test_legacy_assignment_views_track_rebinding_and_padding():
    cache = binding_cache_type()(head_size=128, cache_rope_positions=True)
    storage = torch.zeros(3, 4, 1, 160, dtype=torch.bfloat16)
    bound = storage[..., :140]
    cache.kv_cache = bound  # Exact vLLM 0.24 binding ABI.
    assert cache.key_cache.shape == (3, 4, 1, 128)
    assert cache.rope_position_cache.shape == (3, 4, 1, 3)
    cache.key_cache[1, 2, 0, 7] = 5
    assert storage[1, 2, 0, 7].item() == 5
    positions = torch.tensor([2**40 + 1, 123, -7], dtype=torch.int64)
    cache.rope_position_cache[1, 2, 0].copy_(positions)
    torch.testing.assert_close(storage[1, 2, 0, 128:140].view(torch.int64), positions)
    assert cache.key_cache.stride(0) == storage.stride(0)
    previous = weakref.ref(bound)
    replacement = torch.zeros_like(storage)[..., :140]
    cache.kv_cache = replacement
    del bound
    assert previous() is None
    assert cache.key_cache.data_ptr() == replacement.data_ptr()
    assert cache.key_cache[1, 2, 0, 7].item() == 0
    cache.kv_cache = torch.tensor([])
    assert cache.key_cache.numel() == 0
    assert cache.rope_position_cache.numel() == 0


def test_optional_hook_validates_and_text_cache_has_no_positions():
    cache = binding_cache_type()(head_size=128, cache_rope_positions=False)
    data = torch.zeros(2, 4, 1, 128, dtype=torch.bfloat16)
    cache.bind_kv_cache(data)
    assert cache.rope_position_cache is None
    assert cache.key_cache.data_ptr() == data.data_ptr()
    with pytest.raises(ValueError, match="packed BF16"):
        cache.bind_kv_cache(data.float())
    with pytest.raises(ValueError, match="QSA raw cache"):
        cache.bind_kv_cache(data.squeeze(2))
