# SPDX-License-Identifier: Apache-2.0
"""Compressed-tensors matching, loading and dynamic Linear NPU contracts."""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(
        current_platform.device_type != "npu", reason="Requires Ascend NPU"
    ),
]


@pytest.fixture(scope="module", autouse=True)
def single_rank_runtime():
    from tests.utils.device_utils import ascend_single_rank_runtime
    from vllm.config import (
        CompilationConfig,
        CompilationMode,
        CUDAGraphMode,
        DeviceConfig,
        VllmConfig,
    )

    def config_factory():
        return VllmConfig(
            additional_config={"weight_nz_mode": 0},
            device_config=DeviceConfig(device="npu"),
            compilation_config=CompilationConfig(
                mode=CompilationMode.NONE, cudagraph_mode=CUDAGraphMode.NONE
            ),
        )

    # Upstream weight/scale parameter constructors record the real TP group.
    with ascend_single_rank_runtime(config_factory) as config:
        yield config


def _config(targets=None, ignore=None, model_class=None):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )
    from vllm.model_executor.model_loader.utils import configure_quant_config

    from vllm_fl import register_quant_linear
    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        bind_compressed_tensors_model,
    )

    register_quant_linear()
    result = CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "quantization_status": "compressed",
            "config_groups": {
                "w8a8": {
                    "targets": targets
                    if targets is not None
                    else [
                        "layers.0.attn.wq_a",
                        "layers.0.attn.wq_b",
                        r"re:.*\.experts\.\d+\.(?:w1|w2|w3)$",
                        r"re:.*\.shared_experts\.(?:w1|w2|w3)$",
                    ],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                    "format": "int-quantized",
                }
            },
            "ignore": ignore or [],
        }
    )
    selected_model = model_class or DeepseekV4ForCausalLM
    bind_compressed_tensors_model(
        result,
        getattr(selected_model, "hf_to_vllm_mapper", None),
        getattr(selected_model, "packed_modules_mapping", None),
    )
    configure_quant_config(result, selected_model)
    return result


@pytest.fixture
def config():
    return _config()


def _linear_layer():
    from vllm.model_executor.layers.linear import LinearBase

    layer = object.__new__(LinearBase)
    torch.nn.Module.__init__(layer)
    return layer


def test_method_resolution(config):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )

    layer = _linear_layer()
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.wq_a"),
        CompressedTensorsLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(layer, "lm_head"), UnquantizedLinearMethod
    )
    assert config.get_quant_method(torch.nn.Identity(), "irrelevant") is None


def test_raw_regex_ignore_survives_model_mapper():
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )

    config = _config([r"re:.*\.attn\..*"], [r"re:.*layers\.0\.attn\.compressor\.wkv$"])
    layer = _linear_layer()
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.compressor.wkv"),
        UnquantizedLinearMethod,
    )
    assert isinstance(
        config.get_quant_method(layer, "model.layers.0.self_attn.wq_a"),
        CompressedTensorsLinearMethod,
    )


def test_mixed_packed_rejected():
    config = _config(["layers.0.ffn.shared_experts.w1"])
    with pytest.raises(ValueError, match="Mixed"):
        config.get_quant_method(
            _linear_layer(), "model.layers.0.mlp.shared_experts.gate_up_proj"
        )


def test_mixed_expertbank_rejected():
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts

    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = SimpleNamespace(num_experts=2)
    config = _config([r"re:.*\.experts\.0\.(?:w1|w2|w3)$"])
    with pytest.raises(ValueError, match="Mixed"):
        config.get_quant_method(layer, "model.layers.0.mlp.experts")


def test_upstream_config_and_model_contract_are_idempotent():
    from vllm.model_executor.layers.quantization import get_quantization_config
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        bind_compressed_tensors_model,
    )

    config = _config()
    assert type(config) is CompressedTensorsConfig
    assert get_quantization_config("compressed-tensors") is CompressedTensorsConfig
    assert not hasattr(config, "weight_nz_mode")
    mapper = DeepseekV4ForCausalLM.hf_to_vllm_mapper
    packed = DeepseekV4ForCausalLM.packed_modules_mapping
    assert bind_compressed_tensors_model(config, mapper, packed)
    assert bind_compressed_tensors_model(config, mapper, packed)


@pytest.fixture
def native():
    def factory():
        config = _config()
        return config.get_quant_method(_linear_layer(), "model.layers.0.self_attn.wq_a")

    return factory


def make_layer(method, dtype, layer=None, prefix="model.layers.0.self_attn.wq_a"):
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    from vllm_fl.quantization.w8a8.linear import AscendInt8ScaledMMLinearKernel

    layer = _linear_layer() if layer is None else layer
    selected = method.quantization_config.get_quant_method(layer, prefix)
    assert type(selected) is type(method)

    def loader(*args):
        return default_weight_loader(*args)

    method.create_weights(layer, 128, [32, 32], 128, 64, dtype, weight_loader=loader)
    assert isinstance(layer.scheme.kernel, AscendInt8ScaledMMLinearKernel)
    assert layer.weight.tp_rank == layer.weight_scale.tp_rank == 0
    assert layer.weight.tp_size == layer.weight_scale.tp_size == 1
    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0
    assert layer.weight_scale.output_dim == 0
    assert not hasattr(layer, "weight_offset")
    assert all(p.weight_loader is loader for p in layer.parameters())
    assert layer.weight_scale.shape == (64, 1)
    assert layer.weight_scale.dtype == torch.float32
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-64, 64, (64, 128), device="npu", dtype=torch.int8)
        )
        scales = torch.linspace(0.010001, 0.020003, 64, device="npu").reshape(-1, 1)
        layer.weight_scale.weight_loader(layer.weight_scale, scales)
        torch.testing.assert_close(layer.weight_scale, scales, rtol=0, atol=0)
    checkpoint_weight = layer.weight.detach().clone()
    method.process_weights_after_loading(layer)
    assert layer.weight.shape == (128, 64)
    assert layer.weight_scale.shape == (64,)
    torch.testing.assert_close(layer.weight, checkpoint_weight.t())
    return layer, checkpoint_weight


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("shape", [(3, 128), (2, 1, 128), (2, 3, 128)])
@pytest.mark.parametrize("with_bias", [False, True])
def test_loader_and_native_reference(native, dtype, shape, with_bias):
    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        w8a8_dynamic_linear,
    )

    torch.manual_seed(17)
    method = native()
    layer, checkpoint_weight = make_layer(method, dtype)
    x = torch.randn(shape, dtype=dtype, device="npu")
    x.reshape(-1, 128)[0].zero_()
    bias = torch.randn(64, dtype=dtype, device="npu") if with_bias else None
    actual = method.apply(layer, x, bias)
    expected = w8a8_dynamic_linear(
        x.reshape(-1, 128),
        checkpoint_weight.t().contiguous(),
        layer.weight_scale,
        bias,
    ).reshape(*shape[:-1], 64)
    assert actual.dtype == dtype
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(
        actual.reshape(-1, 64)[0],
        bias if with_bias else torch.zeros(64, dtype=dtype, device="npu"),
    )
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fresh_input_graph_replay(native, dtype):
    import torch_npu  # noqa: F401

    method = native()
    layer, _ = make_layer(method, dtype)
    x = torch.randn(2, 1, 128, dtype=dtype, device="npu")
    for _ in range(3):
        method.apply(layer, x)
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = method.apply(layer, x)
    torch.npu.synchronize()
    previous = output.clone()
    for _ in range(2):
        x.copy_(torch.randn_like(x))
        expected = method.apply(layer, x)
        output.fill_(float("nan"))
        graph.replay()
        torch.npu.synchronize()
        torch.testing.assert_close(output, expected)
        assert not torch.equal(output, previous)
        previous = output.clone()


def test_asymmetric_scheme_rejected():
    config = _config()
    for scheme in config.target_scheme_map.values():
        scheme["weights"].symmetric = False
    with pytest.raises(NotImplementedError, match="Unsupported"):
        config.get_quant_method(_linear_layer(), "model.layers.0.self_attn.wq_a")


def test_default_dispatch_native(native):
    from vllm_fl.dispatch import get_default_manager, with_strict_mode

    manager = get_default_manager()
    for op in ("w8a8_prepare_weight", "w8a8_dynamic_linear"):
        assert manager.get_selected_impl_id(op) == "vendor.ascend"
    with with_strict_mode():
        method = native()
        layer, _ = make_layer(method, torch.bfloat16)
        output = method.apply(
            layer, torch.ones(2, 128, device="npu", dtype=torch.bfloat16)
        )
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


def test_scoped_reference_dispatch(native):
    from vllm_fl.dispatch import get_default_manager, with_preference

    manager = get_default_manager()
    before = manager.get_selected_impl_id("w8a8_dynamic_linear")
    with with_preference("reference"):
        for op in ("w8a8_prepare_weight", "w8a8_dynamic_linear"):
            assert manager.get_selected_impl_id(op) == "reference.torch"
        method = native()
        layer, _ = make_layer(method, torch.float16)
        output = method.apply(
            layer, torch.ones(2, 128, device="npu", dtype=torch.float16)
        )
        assert output.dtype == torch.float16
        assert torch.isfinite(output).all()
    assert manager.get_selected_impl_id("w8a8_dynamic_linear") == before


@pytest.mark.parametrize(
    "groups,input_width,output_width,tokens",
    [(1, 128, 64, 3), (2, 128, 64, 3), (2, 16, 32768, 1)],
)
def test_grouped_wo_a_dynamic_projection(groups, input_width, output_width, tokens):
    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        w8a8_dynamic_linear,
    )
    from vllm_fl.quantization.w8a8.linear import (
        AscendGroupedW8A8LinearMethod,
    )

    config = _config()
    layer = _linear_layer()
    config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")
    method = AscendGroupedW8A8LinearMethod(config)
    method.configure_grouped(groups)
    method.create_weights(
        layer,
        input_width,
        [output_width * groups],
        input_width,
        output_width * groups,
        torch.bfloat16,
        weight_loader=lambda *args: None,
    )
    layer = layer.to("npu")
    with torch.no_grad():
        layer.weight.copy_(
            torch.randint(-32, 32, layer.weight.shape, device="npu", dtype=torch.int8)
        )
        # Values deliberately not representable in BF16: verify lossless loading.
        scales = torch.linspace(
            0.010001, 0.020003, output_width * groups, device="npu"
        ).reshape(-1, 1)
        layer.weight_scale.copy_(scales)
    weight = layer.weight.detach().clone()
    torch.testing.assert_close(layer.weight_scale, scales, rtol=0, atol=0)
    assert layer.weight_scale.dtype == torch.float32
    method.process_weights_after_loading(layer)
    x = torch.randn(tokens, groups * input_width, dtype=torch.bfloat16, device="npu")
    expected = torch.cat(
        [
            w8a8_dynamic_linear(
                x.reshape(tokens, groups, input_width)[:, i].contiguous(),
                weight.chunk(groups)[i].t().contiguous(),
                scales.chunk(groups)[i].flatten(),
            )
            for i in range(groups)
        ],
        dim=-1,
    )
    actual = method.apply_grouped(layer, x)
    assert actual.shape == (tokens, groups * output_width)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.04)


@pytest.mark.parametrize("groups", [None, 2])
def test_native_output_width_guard_before_preparation(groups):
    from vllm.model_executor.kernels.linear import Int8ScaledMMLinearLayerConfig

    from vllm_fl.quantization.w8a8.linear import (
        AscendGroupedW8A8LinearMethod,
        AscendInt8ScaledMMLinearKernel,
    )

    # Metadata-only tensors test rejection without allocating a large NPU weight.
    rows = 65536 * (groups or 1)
    layer = torch.nn.Module()
    layer.weight = torch.nn.Parameter(
        torch.empty(rows, 16, device="meta", dtype=torch.int8), requires_grad=False
    )
    layer.weight_scale = torch.nn.Parameter(
        torch.empty(rows, 1, device="meta", dtype=torch.float32), requires_grad=False
    )
    if groups is None:
        method = AscendInt8ScaledMMLinearKernel(
            Int8ScaledMMLinearLayerConfig(
                is_channelwise=True, is_static_input_scheme=False, input_symmetric=True
            ),
            ["weight", "weight_scale", "input_scale", "input_zero_point", "azp_adj"],
        )
    else:
        method = AscendGroupedW8A8LinearMethod(_config())
        method.configure_grouped(groups)
    with pytest.raises(NotImplementedError, match="output width N=65536"):
        method.process_weights_after_loading(layer)


def _lm_head_layer():
    from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

    layer = object.__new__(ParallelLMHead)
    torch.nn.Module.__init__(layer)
    return layer


@pytest.mark.parametrize("target", ["head", "lm_head", "model.head", r"re:^head$"])
def test_real_lm_head_upstream_allocation(target):
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
        CompressedTensorsW8A8Int8,
    )

    config = _config([target])
    head = _lm_head_layer()
    method = config.get_quant_method(head, "lm_head")
    assert type(method) is CompressedTensorsLinearMethod
    assert isinstance(head.scheme, CompressedTensorsW8A8Int8)
    head, _ = make_layer(method, torch.bfloat16, layer=head, prefix="lm_head")
    output = method.apply(head, torch.randn(2, 128, device="npu", dtype=torch.bfloat16))
    assert output.shape == (2, 64)
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("use_flaggems", ["0", "1"])
def test_upstream_kernel_stays_native_for_flaggems_policy(monkeypatch, use_flaggems):
    from vllm_fl.quantization.w8a8.linear import AscendInt8ScaledMMLinearKernel

    monkeypatch.setenv("USE_FLAGGEMS", use_flaggems)
    config = _config()
    layer = _linear_layer()
    method = config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")
    layer, _ = make_layer(method, torch.bfloat16, layer=layer)
    assert type(layer.scheme.kernel) is AscendInt8ScaledMMLinearKernel
    import torch_npu

    assert torch_npu.get_npu_format(layer.weight) != 29


@pytest.mark.parametrize("regex", [False, True])
def test_generic_model_mapper_target_and_ignore_aliases(regex):
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsLinearMethod,
    )
    from vllm.model_executor.models.utils import WeightsMapper

    # The quantization resolver consumes any model-owned literal mapper.
    class GenericModel:
        packed_modules_mapping = {}
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={"encoder.": "network."},
            orig_to_new_substr={".ffn.": ".mlp."},
        )

    raw = "encoder.block.ffn.projection"
    canonical = "network.block.mlp.projection"
    target = r"re:^encoder\.block\.ffn\.projection$" if regex else raw
    config = _config([target], model_class=GenericModel)
    assert isinstance(
        config.get_quant_method(_linear_layer(), canonical),
        CompressedTensorsLinearMethod,
    )
    ignored = _config([target], [target], model_class=GenericModel)
    assert isinstance(
        ignored.get_quant_method(_linear_layer(), canonical), UnquantizedLinearMethod
    )
    canonical_ignore = _config([target], [canonical], model_class=GenericModel)
    assert isinstance(
        canonical_ignore.get_quant_method(_linear_layer(), canonical),
        UnquantizedLinearMethod,
    )


@pytest.mark.parametrize("ignore", ["head", "lm_head", r"re:^model\.head$"])
def test_real_lm_head_ignore_aliases(ignore):
    config = _config(["head"], [ignore])
    head = _lm_head_layer()
    assert config.get_quant_method(head, "lm_head") is None
    assert not hasattr(head, "scheme")


def test_float32_linear_allocation_rejected():
    config = _config()
    layer = _linear_layer()
    method = config.get_quant_method(layer, "model.layers.0.self_attn.wq_a")
    with pytest.raises(NotImplementedError, match="BF16/FP16"):
        method.create_weights(
            layer,
            128,
            [64],
            128,
            64,
            torch.float32,
            weight_loader=lambda *args: None,
        )
    assert not dict(layer.named_parameters())


def test_cached_ignore_matches_upstream_and_tracks_mutation():
    from vllm.model_executor.layers.quantization.compressed_tensors.utils import (
        should_ignore_layer,
    )

    config = _config()
    config.packed_modules_mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    filler = [rf"re:^unused_{index}\.projection$" for index in range(520)]
    root = "model.layers.0.gate_up_proj"
    gate = "model.layers.0.gate_proj"
    up = "model.layers.0.up_proj"

    def outcome(predicate):
        try:
            return predicate(), None
        except ValueError as error:
            return None, str(error)

    cases = (
        ("literal", ["literal"]),
        ("MODEL.projection", [r"re:(?i)^model\.projection$"]),
        (root, [gate, up]),
        (root, [gate]),
        (root, [root]),
        (root, [root, gate]),
        (root, [r"re:^model\.layers\.0\.gate_up_proj$", gate]),
        (root, [r"re:^model\.layers\.0\.gate_up_proj$"]),
        (root, [r"re:^model\.layers\.0\.(?:gate_proj|up_proj)$"]),
        (root, [r"re:^model\.layers\.0\.gate_proj$"]),
        ("unmatched", []),
        (None, []),
    )
    for name, rules in cases:
        config.ignore = filler + rules
        expected = outcome(
            lambda name=name: should_ignore_layer(
                name, config.ignore, config.packed_modules_mapping
            )
        )
        assert (
            outcome(
                lambda name=name: config._fl_checkpoint_contract.should_ignore(
                    config, name
                )
            )
            == expected
        )
        cache = config._fl_checkpoint_contract.ignore_cache
        assert (
            outcome(
                lambda name=name: config._fl_checkpoint_contract.should_ignore(
                    config, name
                )
            )
            == expected
        )
        assert config._fl_checkpoint_contract.ignore_cache is cache

    config.ignore = filler.copy()
    assert not config._fl_checkpoint_contract.should_ignore(config, "new_literal")
    previous = config._fl_checkpoint_contract.ignore_cache
    config.ignore.append("new_literal")
    assert config._fl_checkpoint_contract.should_ignore(config, "new_literal")
    assert config._fl_checkpoint_contract.ignore_cache is not previous
    config.ignore = filler + [r"re:(?i)^new_regex$"]
    assert config._fl_checkpoint_contract.should_ignore(config, "NEW_REGEX")
    assert not config._fl_checkpoint_contract.should_ignore(config, "new_literal")


@pytest.mark.parametrize("copy_kind", ["deepcopy", "pickle"])
def test_shared_checkpoint_contract_survives_serialization(copy_kind):
    import copy
    import pickle

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    from vllm_fl.models.deepseek_v4 import DeepseekV4ForCausalLM
    from vllm_fl.quantization.w8a8.compressed_tensors import (
        bind_compressed_tensors_model,
    )

    config = _config([r"re:^head$", r"re:.*\.attn\.wq_a$"])
    layer = _linear_layer()
    expected = config.get_scheme_dict(layer, "lm_head")
    clone = (
        copy.deepcopy(config)
        if copy_kind == "deepcopy"
        else pickle.loads(pickle.dumps(config))
    )
    assert type(clone) is CompressedTensorsConfig
    assert bind_compressed_tensors_model(
        clone,
        DeepseekV4ForCausalLM.hf_to_vllm_mapper,
        DeepseekV4ForCausalLM.packed_modules_mapping,
    )
    assert clone.get_scheme_dict(layer, "lm_head") == expected
    assert clone.get_scheme_dict(layer, "model.layers.0.self_attn.wq_a") is not None
    assert not any(
        callable(value) for value in vars(clone._fl_checkpoint_contract).values()
    )


def test_unmarked_checkpoint_config_retains_upstream_matching():
    import copy
    import inspect

    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (
        CompressedTensorsConfig,
    )

    opted_in = _config()
    plain = CompressedTensorsConfig.from_config(copy.deepcopy(opted_in.config))
    assert not hasattr(plain, "_fl_checkpoint_contract")
    original = inspect.unwrap(CompressedTensorsConfig.get_scheme_dict)
    layer = _linear_layer()
    for name in (
        "layers.0.attn.wq_a",
        "model.layers.0.self_attn.wq_a",
        "lm_head",
        None,
    ):
        assert plain.get_scheme_dict(layer, name) == original(plain, layer, name)


@pytest.mark.parametrize("quantized", [False, True])
def test_model_grouped_projection_uses_standard_linear_apply(quantized):
    from vllm.model_executor.layers.linear import ColumnParallelLinear

    from vllm_fl.dispatch.backends.reference.impl.quantization import (
        w8a8_dynamic_linear,
    )
    from vllm_fl.quantization.w8a8.linear import configure_grouped_projection

    prefix = "model.layers.0.self_attn.wo_a"
    layer = ColumnParallelLinear(
        128,
        128,
        bias=False,
        params_dtype=torch.bfloat16,
        quant_config=_config([prefix]) if quantized else None,
        prefix=prefix,
        return_bias=False,
    ).to("npu")
    configure_grouped_projection(layer, 2)
    with torch.no_grad():
        if quantized:
            layer.weight.copy_(
                torch.randint(
                    -24, 24, layer.weight.shape, device="npu", dtype=torch.int8
                )
            )
            layer.weight_scale.fill_(0.015625)
        else:
            layer.weight.copy_(torch.randn_like(layer.weight))
    weight = layer.weight.detach().clone()
    scale = layer.weight_scale.detach().clone() if quantized else None
    layer.quant_method.process_weights_after_loading(layer)
    x = torch.randn(2, 2, 128, device="npu", dtype=torch.bfloat16)
    if quantized:
        expected = torch.cat(
            [
                w8a8_dynamic_linear(
                    x[:, group].contiguous(),
                    weight.chunk(2)[group].t().contiguous(),
                    scale.chunk(2)[group].flatten(),
                )
                for group in range(2)
            ],
            dim=-1,
        )
    else:
        expected = (
            torch.einsum("tgi,goi->tgo", x.float(), weight.float().reshape(2, 64, 128))
            .flatten(1)
            .to(x.dtype)
        )
    actual = layer(x)
    assert actual.shape == (2, 128)
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.04)
