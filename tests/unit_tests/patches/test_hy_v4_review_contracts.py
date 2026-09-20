# SPDX-License-Identifier: Apache-2.0
import importlib
from types import SimpleNamespace

import pytest
import torch

from vllm_fl.models import hy_v4_attention as attention, hy_v4_flashmla_sparse as sparse
from vllm_fl.patches import hy_v4_runtime as runtime, hy_v4_v024 as compat


@pytest.mark.parametrize("kind", ["selection", "capability", "configuration"])
def test_sink_unavailable_is_fatal(monkeypatch, kind):
    layer = object.__new__(attention.HYV4MLAAttention)
    layer.kv_lora_rank, layer.qk_rope_head_dim = 512, 64
    layer.is_sparse, layer.num_local_heads = True, 4
    backend = SimpleNamespace(supports_sink=lambda: False, get_name=lambda: "NO_SINK")

    def select(**kwargs):
        if kind == "selection":
            raise ValueError("backend unavailable")
        return backend

    monkeypatch.setattr(attention, "get_attn_backend", select)
    monkeypatch.setattr(
        attention,
        "current_platform",
        SimpleNamespace(
            device_type="cuda",
            get_device_capability=lambda: None if kind == "capability" else (9, 0),
        ),
    )
    monkeypatch.setattr(
        attention, "get_current_vllm_config", lambda: SimpleNamespace(cache_config=None)
    )
    monkeypatch.setattr(
        sparse.HYV4FlashMLASparseBackend,
        "validate_configuration",
        lambda **kw: ["unsupported KV"],
    )
    with pytest.raises(RuntimeError, match="requires attention sink support"):
        layer._resolve_sink_backend("auto")


@pytest.mark.parametrize("tokens", [1, 5])
def test_sink_reaches_bf16_kernel_for_decode_and_prefill(monkeypatch, tokens):
    seen = []

    def kernel(q, cache, indices, scale, *, attn_sink, topk_length):
        seen.append(attn_sink)
        return (q * torch.sigmoid(-attn_sink)[None, :, None],)

    monkeypatch.setattr(sparse, "flash_mla_sparse_fwd", kernel)
    layer = object.__new__(sparse.HYV4FlashMLASparseImpl)
    layer.num_heads = layer.prefill_padding = 4
    layer.softmax_scale = 1.0
    layer.sinks = torch.arange(4, dtype=torch.float32)
    q = torch.ones(tokens, 4, 4)
    out = layer._bf16_flash_mla_kernel(
        q, torch.ones(8, 4), torch.zeros(tokens, 2, dtype=torch.int32)
    )
    assert seen == [layer.sinks]
    torch.testing.assert_close(
        out, torch.sigmoid(-layer.sinks)[None, :, None].expand_as(q)
    )
    assert not torch.equal(out, q)


@pytest.mark.parametrize(
    "indexer_types,layer_types",
    [
        (["shared"], ["sparse"]),
        (["full", "shared"], ["dense", "sparse"]),
    ],
)
def test_shared_indexer_needs_sparse_producer(indexer_types, layer_types):
    config = SimpleNamespace(
        index_topk=8,
        num_hidden_layers=len(indexer_types),
        indexer_types=indexer_types,
        layer_types=layer_types,
    )
    with pytest.raises(ValueError, match="no preceding full sparse producer"):
        attention.compute_skip_topk_layers(config)


def test_shared_indexer_pp_contract():
    config = SimpleNamespace(
        index_topk=8, num_hidden_layers=2, indexer_types=["full", "shared"]
    )
    attention.validate_hy4_parallel_config(config, 1)
    with pytest.raises(ValueError, match="pipeline_parallel_size=1"):
        attention.validate_hy4_parallel_config(config, 2)
    config.indexer_types = ["full", "full"]
    attention.validate_hy4_parallel_config(config, 2)


def test_schema_is_not_a_kernel_and_composite_is_valid():
    lib = torch.library.Library("hy4_capability_test", "DEF")
    try:
        lib.define("only_schema(Tensor x) -> Tensor")
        assert callable(torch.ops.hy4_capability_test.only_schema)
        assert not runtime.has_device_kernel("hy4_capability_test::only_schema", "cpu")
        lib.impl("only_schema", lambda x: x + 1, "CompositeImplicitAutograd")
        assert runtime.has_device_kernel("hy4_capability_test::only_schema", "cpu")
    finally:
        lib._destroy()


@pytest.mark.parametrize(
    "env,value",
    [
        ("USE_FLAGGEMS", "0"),
        ("VLLM_FL_PREFER", "reference"),
        ("VLLM_FL_FLAGOS_BLACKLIST", "per_token_group_quant_fp8"),
        ("VLLM_FL_PREFER_ENABLED", "false"),
    ],
)
def test_quantizer_respects_backend_policy(monkeypatch, env, value):
    import flag_gems

    from vllm_fl import utils

    monkeypatch.setattr(utils, "get_op_config", lambda: None)
    monkeypatch.delenv("VLLM_FL_FLAGOS_WHITELIST", raising=False)
    monkeypatch.setenv(env, value)
    monkeypatch.setattr(
        flag_gems,
        "per_token_group_quant_fp8",
        lambda *a, **k: pytest.fail("forbidden FlagGems call"),
    )
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: True)
    expected = object()
    monkeypatch.setattr(
        fp8_utils, "per_token_group_quant_fp8", lambda *a, **k: expected
    )
    assert (
        runtime._resolve_query_quantizer()(torch.zeros(1, 32), 32, use_ue8m0=False)
        is expected
    )
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: False)
    with pytest.raises(RuntimeError, match="no permitted implementation"):
        runtime._resolve_query_quantizer()(torch.zeros(1, 32), 32, use_ue8m0=False)


def test_quantization_wrapper_is_lazy_and_preserves_other_models():
    calls = []

    class Config:
        @classmethod
        def override_quantization_method(cls, *a, **kw):
            return "native-result"

    def get(name):
        calls.append(name)
        return Config

    quant = SimpleNamespace(get_quantization_config=get)
    compat._patch_mxfp8_override_order(quant)
    assert not calls
    alias = quant.get_quantization_config("mxfp8")
    assert (
        alias.override_quantization_method(
            {}, None, SimpleNamespace(model_type="other")
        )
        == "native-result"
    )
    assert (
        alias.override_quantization_method(
            {}, None, SimpleNamespace(model_type="hy_v4")
        )
        is None
    )


def test_registration_never_calls_quantization_getter(monkeypatch):
    from vllm.model_executor.layers import quantization

    monkeypatch.setattr(
        quantization,
        "get_quantization_config",
        lambda *a: pytest.fail("heavy quantization probe"),
    )
    # Test the real registration hook, with registry mutation restored afterward.
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry
    from vllm.transformers_utils import config, model_arch_config_convertor

    monkeypatch.setattr(config, "_CONFIG_REGISTRY", dict(config._CONFIG_REGISTRY))
    monkeypatch.setattr(
        model_arch_config_convertor,
        "MODEL_ARCH_CONFIG_CONVERTORS",
        dict(model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS),
    )
    monkeypatch.setattr(
        model_loader,
        "_LOAD_FORMAT_TO_MODEL_LOADER",
        dict(model_loader._LOAD_FORMAT_TO_MODEL_LOADER),
    )
    monkeypatch.setattr(registry.ModelRegistry, "register_model", lambda *a: None)
    monkeypatch.setattr(compat, "is_vllm_024", lambda *_: True)
    assert compat.apply_hy_v4_v024_patches()


def test_fallback_failure_rolls_back_real_module_attributes(monkeypatch):
    import vllm._custom_ops as ops
    import vllm.model_executor.layers.attention.mla_attention as mla
    import vllm.model_executor.layers.sparse_attn_indexer as indexer
    import vllm.v1.attention.backends.mla.flashmla_sparse as native
    from vllm import platforms

    monkeypatch.delattr(indexer, "_hy4_runtime_installed", raising=False)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: False)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(
        mla,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda: False),
    )
    plan = runtime.validate_hy4_runtime(_prefill_config())
    modules = [indexer, native, ops, sparse, mla]
    modules += [
        importlib.import_module("flag_gems.fused." + n)
        for n in ("top_k_per_row_prefill", "top_k_per_row_decode", "flashmla_sparse")
    ]
    snapshots = [dict(vars(m)) for m in modules]
    original_set = runtime.PatchTransaction.set
    error = RuntimeError("late installation failure")

    def fail_last(tx, module, name, value):
        if name == "_hy4_runtime_installed":
            raise error
        original_set(tx, module, name, value)

    monkeypatch.setattr(runtime.PatchTransaction, "set", fail_last)
    for _ in range(2):
        with pytest.raises(RuntimeError) as exc:
            runtime.install_hy4_flaggems_fallback(plan)
        assert exc.value is error
        assert not getattr(indexer, "_hy4_runtime_installed", False)
        for module, before in zip(modules, snapshots):
            for name, value in before.items():
                assert getattr(module, name) is value, (module.__name__, name)


def test_fp8_kv_requires_native_metadata(monkeypatch):
    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm import platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(
        mla,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda: False),
    )
    config = _prefill_config()
    config.cache_config.cache_dtype = "fp8"
    with pytest.raises(ValueError, match="FP8 KV requires a complete native path"):
        runtime.validate_hy4_runtime(config)


def _prefill_config(explicit=None):
    return SimpleNamespace(
        attention_config=SimpleNamespace(mla_prefill_backend=explicit),
        cache_config=SimpleNamespace(cache_dtype="auto"),
        model_config=SimpleNamespace(
            dtype=torch.bfloat16,
            hf_text_config=SimpleNamespace(
                index_topk=2048,
                qk_nope_head_dim=192,
                qk_rope_head_dim=64,
                v_head_dim=256,
            ),
        ),
    )


def test_explicit_valid_prefill_is_preserved():
    backend = SimpleNamespace(is_available=lambda: True, get_name=lambda: "CUSTOM")
    assert (
        runtime.select_hy4_prefill_backend(
            _prefill_config("CUSTOM"), lambda cfg: backend
        )
        is backend
    )


@pytest.mark.parametrize(
    "error",
    [
        ValueError("invalid explicit selection"),
        ImportError("missing extension"),
        AssertionError("invalid dimensions"),
    ],
)
def test_explicit_prefill_errors_are_not_swallowed(error):
    def selector(cfg):
        raise error

    with pytest.raises(type(error), match=str(error)):
        runtime.select_hy4_prefill_backend(_prefill_config("EXPLICIT"), selector)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No valid MLA prefill backend found with test"),
        ImportError("missing extension"),
        OSError("missing library"),
    ],
)
def test_automatic_prefill_absence_uses_fallback(error):

    def selector(cfg):
        raise error

    assert runtime.select_hy4_prefill_backend(_prefill_config(), selector) is None


@pytest.mark.parametrize(
    "error",
    [ValueError("invalid model dimensions"), AssertionError("invalid dimensions")],
)
def test_automatic_prefill_does_not_hide_configuration_errors(error):
    def selector(cfg):
        raise error

    with pytest.raises(type(error), match=str(error)):
        runtime.select_hy4_prefill_backend(_prefill_config(), selector)


@pytest.mark.parametrize(
    "name", ["per_token_group_quant_fp8", "flash_attn_varlen_func"]
)
def test_preflight_requires_callable_implementations(monkeypatch, name):
    import flag_gems

    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm import platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda *_: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: False)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(
        mla,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda *_: False),
    )
    monkeypatch.setattr(flag_gems, name, None)
    with pytest.raises(RuntimeError, match=name):
        runtime.validate_hy4_runtime(_prefill_config())


def test_fallback_success_idempotence_and_prefill_selection(monkeypatch):
    import vllm.model_executor.layers.attention.mla_attention as mla
    import vllm.model_executor.layers.sparse_attn_indexer as indexer
    from vllm import platforms

    monkeypatch.delattr(indexer, "_hy4_runtime_installed", raising=False)
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: False)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    selections = []

    def no_backend(cfg):
        selections.append(cfg)
        raise ValueError("No valid MLA prefill backend found with test")

    monkeypatch.setattr(mla, "get_mla_prefill_backend", no_backend)
    config = _prefill_config()
    plan = runtime.validate_hy4_runtime(config)
    # Installation consumes the resolved plan without policy/capability probes.
    monkeypatch.setattr(
        runtime, "use_flaggems_op", lambda *a: pytest.fail("policy recheck")
    )
    monkeypatch.setattr(
        runtime, "has_device_kernel", lambda *a: pytest.fail("kernel recheck")
    )
    with runtime.patch_transaction() as tx:
        try:
            assert runtime._install_fallback(tx, plan)
            installed = indexer.ops
            assert runtime._install_fallback(tx, plan)
            assert indexer.ops is installed
            assert mla.get_mla_prefill_backend(config) is plan.prefill_backend
            assert selections == [config]
        finally:
            tx.rollback()
    assert not getattr(indexer, "_hy4_runtime_installed", False)


@pytest.mark.parametrize("cache_dtype", ["auto", "fp8"])
def test_native_core_without_prefill_is_not_a_complete_native_plan(
    monkeypatch, cache_dtype
):
    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm import platforms
    from vllm.v1.attention.ops import flashmla

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda *_: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: True)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: True)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(flashmla, "is_flashmla_sparse_supported", lambda: (True, None))
    monkeypatch.setattr(
        mla,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda *_: False),
    )
    config = _prefill_config()
    config.cache_config.cache_dtype = cache_dtype
    if cache_dtype == "fp8":
        with pytest.raises(
            ValueError, match="complete native path including MLA prefill"
        ):
            runtime.validate_hy4_runtime(config)
    else:
        plan = runtime.validate_hy4_runtime(config)
        assert plan.provider == "native"
        assert callable(plan.query_quantizer)
        assert plan.prefill_backend.is_available()
        assert set(plan.operations) == {"flash_attn_varlen_func"}


def test_query_resolver_rejects_missing_native_callable(monkeypatch):
    from vllm.model_executor.layers.quantization.utils import fp8_utils

    monkeypatch.setattr(runtime, "use_flaggems_op", lambda *a: False)
    monkeypatch.setattr(runtime, "has_device_kernel", lambda *a: True)
    monkeypatch.setattr(fp8_utils, "per_token_group_quant_fp8", None)
    with pytest.raises(
        RuntimeError, match="per_token_group_quant_fp8 has no permitted implementation"
    ):
        runtime._resolve_query_quantizer()


@pytest.mark.parametrize("qk_width,v_width", [(576, 512), (192, 256)])
def test_portable_prefill_rejects_unsupported_dimensions_before_install(
    monkeypatch, qk_width, v_width
):
    import vllm.model_executor.layers.attention.mla_attention as mla
    from vllm import platforms

    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(is_cuda=lambda *_: True)
    )
    monkeypatch.setattr(runtime, "native_hy4_available", lambda *_: False)
    monkeypatch.setattr(runtime, "use_flaggems_op", lambda name: True)
    monkeypatch.setattr(
        mla,
        "get_mla_prefill_backend",
        lambda cfg: SimpleNamespace(is_available=lambda *_: False),
    )
    config = _prefill_config()
    config.model_config.hf_text_config.qk_nope_head_dim = qk_width - 64
    config.model_config.hf_text_config.v_head_dim = v_width
    monkeypatch.setattr(
        runtime,
        "install_hy4_flaggems_fallback",
        lambda *args: pytest.fail("invalid plan must not install"),
    )
    with pytest.raises(ValueError, match="qk_head_dim <= 256"):
        runtime.prepare_hy4_runtime(config)
