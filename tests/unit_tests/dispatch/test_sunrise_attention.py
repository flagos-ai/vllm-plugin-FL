# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU contract tests for the Sunrise vLLM 0.28 attention bridge.

Only external vLLM and device kernel imports are stubbed. The Sunrise
metadata builder and forward path execute with real CPU tensors.
"""

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def sunrise_modules(monkeypatch):
    def install(name, **attrs):
        if "." in name:
            parent_name, child_name = name.rsplit(".", 1)
            parent = sys.modules.get(parent_name)
            if parent is None:
                parent = install(parent_name)
        module = ModuleType(name)
        module.__path__ = []
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            monkeypatch.setattr(parent, child_name, module, raising=False)
        return module

    # Isolate these imports from installed vLLM plugins and GPU runtimes.
    install("vllm")
    envs = install("vllm.envs", VLLM_BATCH_INVARIANT=False)
    config = SimpleNamespace(
        model_config=SimpleNamespace(
            is_hybrid=False,
            get_num_attention_heads=lambda parallel: 2,
            get_num_kv_heads=lambda parallel: 1,
            get_head_size=lambda: 8,
        ),
        parallel_config=SimpleNamespace(cp_kv_cache_interleave_size=1),
        cache_config=SimpleNamespace(),
        compilation_config=SimpleNamespace(
            cudagraph_mode=SimpleNamespace(has_full_cudagraphs=lambda: False),
            max_cudagraph_capture_size=8,
        ),
        attention_config=SimpleNamespace(),
    )
    install(
        "vllm.config",
        VllmConfig=SimpleNamespace,
        get_current_vllm_config=lambda: config,
        get_layers_from_vllm_config=lambda *args: {},
    )
    install("vllm.config.cache", CacheDType=str)
    install(
        "vllm.distributed.parallel_state",
        get_dcp_group=lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )
    install("vllm.logger", init_logger=logging.getLogger)
    install("vllm.model_executor.layers.attention", Attention=type("Attention", (), {}))
    install("vllm.platforms.interface", DeviceCapability=tuple)
    install("vllm.utils.math_utils", cdiv=lambda n, d: (n + d - 1) // d)
    install(
        "vllm.utils.torch_utils",
        is_quantized_kv_cache=lambda dtype: dtype.startswith(("fp8", "int")),
    )

    class AttentionBackend:
        pass

    class AttentionImpl:
        def __new__(cls, *args, **kwargs):
            # vLLM 0.28 initializes context parallel state in the base __new__.
            instance = super().__new__(cls)
            instance.dcp_world_size = 1
            instance.dcp_rank = 0
            return instance

    class AttentionMetadataBuilder:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, spec, layers, vllm_config, device):
            self.vllm_config = vllm_config
            self.device = device

    class MultipleOf:
        def __init__(self, base):
            self.base = base

    attention_types = SimpleNamespace(
        DECODER="decoder",
        ENCODER="encoder",
        ENCODER_ONLY="encoder_only",
        ENCODER_DECODER="encoder_decoder",
    )
    install(
        "vllm.v1.attention.backend",
        AttentionBackend=AttentionBackend,
        AttentionImpl=AttentionImpl,
        AttentionMetadataBuilder=AttentionMetadataBuilder,
        AttentionCGSupport=SimpleNamespace(UNIFORM_BATCH=2),
        AttentionType=attention_types,
        CommonAttentionMetadata=SimpleNamespace,
        MultipleOf=MultipleOf,
    )
    install(
        "vllm.v1.attention.backends.registry",
        AttentionBackendEnum=SimpleNamespace(CUSTOM="CUSTOM"),
        register_backend=Mock(),
    )
    layout = SimpleNamespace(value="NHD")
    install(
        "vllm.v1.attention.backends.utils",
        get_dcp_local_seq_lens=lambda seq_lens, *args: seq_lens,
        get_kv_cache_layout=lambda: layout.value,
    )
    install("vllm.v1.attention.ops.common", cp_lse_ag_out_rs=Mock())
    install("vllm.v1.attention.ops.merge_attn_states", merge_attn_states=Mock())
    install("vllm.v1.kv_cache_interface", AttentionSpec=SimpleNamespace)

    def write_cache(key, value, key_cache, value_cache, slots, *args):
        block_size = key_cache.shape[1]
        for token, slot in enumerate(slots.tolist()):
            if slot >= 0:
                block, offset = divmod(slot, block_size)
                key_cache[block, offset].copy_(key[token])
                value_cache[block, offset].copy_(value[token])

    cache_kernel = Mock(side_effect=write_cache)
    flash_kernel = Mock(side_effect=lambda **kwargs: kwargs["out"].copy_(kwargs["q"]))
    install(
        "flaggems_vllm",
        flash_attn_varlen_func=flash_kernel,
        reshape_and_cache_flash=cache_kernel,
    )
    install("vllm_fl.dispatch.backends.base", Backend=type("Backend", (), {}))
    root = Path(__file__).resolve().parents[3]
    vendor = root / "vllm_fl/dispatch/backends/vendor/sunrise"

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    return SimpleNamespace(
        attention=load("_sunrise_attention_test", vendor / "impl/attention.py"),
        backend=load("_sunrise_backend_test", vendor / "sunrise.py"),
        config=config,
        envs=envs,
        layout=layout,
        cache_kernel=cache_kernel,
        flash_kernel=flash_kernel,
    )


def make_impl(modules, **kwargs):
    return modules.attention.AttentionFLImpl(
        num_heads=2,
        head_size=8,
        scale=0.5,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        **kwargs,
    )


def make_metadata(modules):
    common = SimpleNamespace(
        num_reqs=1,
        num_actual_tokens=2,
        max_query_len=2,
        max_seq_len=2,
        query_start_loc=torch.tensor([0, 2], dtype=torch.int32),
        seq_lens=torch.tensor([2], dtype=torch.int32),
        block_table_tensor=torch.tensor([[0]], dtype=torch.int32),
        slot_mapping=torch.tensor([3, -1]),
        causal=True,
    )
    builder = modules.attention.AttentionFLMetadataBuilder(
        SimpleNamespace(dtype=torch.bfloat16, block_size=16),
        ["layer"],
        modules.config,
        torch.device("cpu"),
    )
    return builder.build(0, common)


def test_v028_backend_validation_signature(sunrise_modules):
    backend = sunrise_modules.attention.AttentionFLBackend
    assert (
        backend.supports_combination(
            128, torch.bfloat16, "auto", None, False, False, False, False, None
        )
        is None
    )
    assert backend.supports_sliding_window()
    assert not backend.supports_non_causal()
    assert backend.forward_includes_kv_cache_update


@pytest.mark.parametrize(
    "feature", ["has_sink", "use_mla", "use_sparse", "use_mm_prefix"]
)
def test_unsupported_attention_features_are_rejected(sunrise_modules, feature):
    args = dict(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        device_capability=None,
    )
    args[feature] = True
    assert sunrise_modules.attention.AttentionFLBackend.supports_combination(**args)


@pytest.mark.parametrize(
    ("layout", "layers", "expected"),
    [
        ("NHD", False, (0, 1, 2, 3, 4)),
        ("HND", False, (0, 1, 3, 2, 4)),
        ("NHD", True, (2, 0, 1, 3, 4, 5)),
        ("HND", True, (2, 4, 0, 1, 3, 5)),
    ],
)
def test_vendor_cache_layout_is_preserved(sunrise_modules, layout, layers, expected):
    sunrise_modules.layout.value = layout
    backend = sunrise_modules.attention.AttentionFLBackend
    assert backend.get_kv_cache_shape(3, 16, 2, 8) == (2, 3, 16, 2, 8)
    assert backend.get_kv_cache_stride_order(layers) == expected


def test_impl_uses_v028_batch_invariant_setting(sunrise_modules):
    sunrise_modules.envs.VLLM_BATCH_INVARIANT = True
    assert make_impl(sunrise_modules, sinks=None).batch_invariant_enabled
    with pytest.raises(NotImplementedError, match="sinks"):
        make_impl(sunrise_modules, sinks=torch.zeros(2))


def test_metadata_and_forward_keep_cache_update_inside_attention(sunrise_modules):
    impl = make_impl(sunrise_modules)
    metadata = make_metadata(sunrise_modules)
    assert metadata.query_start_loc.tolist() == [0, 2]
    assert metadata.slot_mapping.tolist() == [3, -1]
    assert metadata.max_num_splits == 0
    query = torch.arange(48, dtype=torch.float32).reshape(3, 2, 8)
    key = torch.arange(24, dtype=torch.float32).reshape(3, 1, 8)
    value = key + 100
    cache = torch.zeros(2, 1, 16, 1, 8)
    output = torch.full_like(query, -1)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    result = impl.forward(layer, query, key, value, cache, metadata, output)
    assert result is output
    torch.testing.assert_close(output[:2], query[:2])
    assert torch.all(output[2] == -1)
    torch.testing.assert_close(cache[0, 0, 3], key[0])
    torch.testing.assert_close(cache[1, 0, 3], value[0])
    assert torch.count_nonzero(cache[:, :, :3]) == 0
    sunrise_modules.cache_kernel.assert_called_once()
    assert sunrise_modules.flash_kernel.call_args.kwargs["causal"] is True


def test_kv_sharing_does_not_rewrite_cache(sunrise_modules):
    impl = make_impl(sunrise_modules, kv_sharing_target_layer_name="previous")
    tensor = torch.ones(2, 2, 8)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    cache = torch.zeros(2, 1, 16, 1, 8)
    impl.forward(
        layer,
        tensor,
        None,
        None,
        cache,
        make_metadata(sunrise_modules),
        torch.empty_like(tensor),
    )
    sunrise_modules.cache_kernel.assert_not_called()


def test_profile_forward_initializes_output_without_device_kernels(sunrise_modules):
    output = torch.ones(2, 2, 8)
    result = make_impl(sunrise_modules).forward(
        None, None, None, None, None, None, output
    )
    assert result is output
    assert torch.count_nonzero(output) == 0
    sunrise_modules.cache_kernel.assert_not_called()
    sunrise_modules.flash_kernel.assert_not_called()


def test_encoder_attention_is_bidirectional_and_does_not_write_cache(sunrise_modules):
    impl = make_impl(sunrise_modules, attn_type="encoder_only")
    tensor = torch.ones(2, 2, 8)
    layer = SimpleNamespace(
        _q_scale=torch.ones(1), _k_scale=torch.ones(1), _v_scale=torch.ones(1)
    )
    output = torch.empty_like(tensor)
    impl.forward(
        layer, tensor, tensor, tensor, None, make_metadata(sunrise_modules), output
    )
    torch.testing.assert_close(output, tensor)
    sunrise_modules.cache_kernel.assert_not_called()
    assert sunrise_modules.flash_kernel.call_args.kwargs["causal"] is False


@pytest.mark.parametrize("kind", ["missing", "empty", "error", "available"])
def test_backend_detection_is_safe_on_other_platforms(
    sunrise_modules, monkeypatch, kind
):
    if kind == "missing":
        monkeypatch.delattr(torch, "ptpu", raising=False)
    else:
        is_available = Mock(return_value=True)
        if kind == "error":
            is_available.side_effect = RuntimeError("driver not initialized")
        monkeypatch.setattr(
            torch,
            "ptpu",
            SimpleNamespace(
                is_available=is_available,
                device_count=lambda: 0 if kind == "empty" else 8,
            ),
            raising=False,
        )
    backend = sunrise_modules.backend.SunriseBackend()
    assert backend.is_available() is (kind == "available")
    if kind == "available":
        assert backend.is_available()
        is_available.assert_called_once()
