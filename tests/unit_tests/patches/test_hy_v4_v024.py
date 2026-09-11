from types import SimpleNamespace

import torch
from torch import nn

from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

from vllm_fl.model_loader.hy_v4_loader import (
    HYV4SafetensorsLoader,
    _contiguous_runs,
    _read_local_experts,
)
from vllm_fl.models import hy_v4
from vllm_fl.models.hy_v4 import (
    HYV4ForCausalLM,
    _hyv4_env_flag,
    _hyv4_fp32_combine_and_reduce,
    _HYV4FP32RoutedOutput,
    _try_load_mxfp8_indexer_wk,
)
from vllm_fl.patches import hy_v4_v024 as compat


def test_hy4_convertor_uses_compressed_mla_dimensions():
    config = SimpleNamespace(
        model_type="hy_v4",
        architectures=["HYV4ForCausalLM"],
        hidden_size=6144,
        num_hidden_layers=78,
        num_attention_heads=64,
        num_key_value_heads=8,
        vocab_size=120832,
        n_routed_experts=256,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        max_position_embeddings=1048576,
        quantization_config=None,
    )
    converted = compat.HYV4ModelArchConfigConvertor(config, config).convert()

    assert converted.head_size == 576
    assert converted.total_num_kv_heads == 8
    assert converted.is_deepseek_mla is True


def test_hy4_convertor_normalizes_mxfp8_for_override_detection():
    quant_config = {
        "quant_method": "mxfp8",
        "ignored_layers": ["lm_head"],
        "kv_cache_quant_algo": None,
    }
    config = SimpleNamespace(
        model_type="hy_v4",
        quantization_config=quant_config,
    )
    converted = compat.HYV4ModelArchConfigConvertor(
        config, config
    ).get_quantization_config()

    assert converted == {
        "quant_method": "modelopt",
        "quantization": {
            "quant_algo": "MXFP8",
            "kv_cache_quant_algo": None,
            "exclude_modules": ["lm_head"],
        },
    }
    assert quant_config["quant_method"] == "mxfp8"


def test_mxfp8_alias_is_probed_after_canonical_override():
    class FakeMXFP8Config:
        @classmethod
        def override_quantization_method(cls, hf_quant_cfg, user_quant, hf_config=None):
            return "modelopt_mxfp8"

    class OtherConfig:
        pass

    configs = {
        "mxfp8": FakeMXFP8Config,
        "modelopt_mxfp8": FakeMXFP8Config,
        "other": OtherConfig,
    }
    quantization = SimpleNamespace(get_quantization_config=lambda name: configs[name])

    compat._patch_mxfp8_override_order(quantization)
    alias = quantization.get_quantization_config("mxfp8")
    canonical = quantization.get_quantization_config("modelopt_mxfp8")

    assert alias.override_quantization_method({}, None) is None
    assert canonical.override_quantization_method({}, None) == "modelopt_mxfp8"
    assert quantization.get_quantization_config("other") is OtherConfig

    # Re-applying the runtime hook must not stack wrappers.
    getter = quantization.get_quantization_config
    compat._patch_mxfp8_override_order(quantization)
    assert quantization.get_quantization_config is getter


def test_apply_registers_plugin_owned_hy4_components(monkeypatch):
    from vllm.model_executor import model_loader
    from vllm.model_executor.models import registry as model_registry
    from vllm.transformers_utils import (
        config as transformers_config,
        model_arch_config_convertor,
    )

    registered_models = {}
    fake_registry = SimpleNamespace(
        register_model=lambda architecture, model: registered_models.__setitem__(
            architecture, model
        )
    )
    registered_loaders = {}

    def register_loader(load_format):
        def register(loader):
            registered_loaders[load_format] = loader
            model_loader._LOAD_FORMAT_TO_MODEL_LOADER[load_format] = loader
            return loader

        return register

    monkeypatch.setattr(compat, "is_vllm_024", lambda: True)
    monkeypatch.setattr(transformers_config, "_CONFIG_REGISTRY", {})
    monkeypatch.setattr(model_arch_config_convertor, "MODEL_ARCH_CONFIG_CONVERTORS", {})
    monkeypatch.setattr(model_registry, "ModelRegistry", fake_registry)
    monkeypatch.setattr(model_loader, "_LOAD_FORMAT_TO_MODEL_LOADER", {})
    monkeypatch.setattr(model_loader, "register_model_loader", register_loader)

    assert compat.apply_hy_v4_v024_patches() is True

    assert {"hy_v4": compat.HYV4Config} == transformers_config._CONFIG_REGISTRY
    assert {
        "hy_v4": compat.HYV4ModelArchConfigConvertor
    } == model_arch_config_convertor.MODEL_ARCH_CONFIG_CONVERTORS
    assert registered_models == {
        "HYV4ForCausalLM": "vllm_fl.models.hy_v4:HYV4ForCausalLM"
    }
    assert registered_loaders == {"hy4_safetensors": compat.HYV4SafetensorsLoader}


def test_flagos_oot_platform_inherits_mxfp8_linear_candidates():
    from vllm.model_executor.kernels.linear import _POSSIBLE_MXFP8_KERNELS
    from vllm.model_executor.kernels.linear.mxfp8.emulation import (
        EmulationMxfp8LinearKernel,
    )
    from vllm.platforms import PlatformEnum

    from vllm_fl.quantization.quant_linear import add_oot_quant_kernel

    add_oot_quant_kernel()

    assert PlatformEnum.OOT in _POSSIBLE_MXFP8_KERNELS
    assert EmulationMxfp8LinearKernel in _POSSIBLE_MXFP8_KERNELS[PlatformEnum.OOT]


def test_plugin_loader_delegates_discovery_as_safetensors(monkeypatch):
    loader = object.__new__(HYV4SafetensorsLoader)
    loader.load_config = SimpleNamespace(load_format="hy4_safetensors")
    observed = []

    def fake_prepare(self, *args):
        observed.append(self.load_config.load_format)
        return "/weights", ["/weights/model.safetensors"], True

    monkeypatch.setattr(DefaultModelLoader, "_prepare_weights", fake_prepare)
    result = loader._prepare_weights("/weights", None, None, False, None)

    assert result == ("/weights", ["/weights/model.safetensors"], True)
    assert observed == ["safetensors"]
    assert loader.load_config.load_format == "hy4_safetensors"


def test_hy4_contiguous_expert_runs_and_slices():
    assert _contiguous_runs([]) == []
    assert _contiguous_runs([0, 1, 2, 7, 9, 10]) == [
        (0, 3),
        (7, 8),
        (9, 11),
    ]
    packed = torch.arange(12).view(6, 2)
    torch.testing.assert_close(
        _read_local_experts(packed, [1, 2, 5]),
        packed[[1, 2, 5]],
    )


def test_hy4_packed_experts_do_not_match_shared_expert():
    shared = "model.layers.1.mlp.shared_experts.down_proj.weight"
    routed = "model.layers.1.mlp.experts.down_proj"
    assert HYV4ForCausalLM._packed_expert_target(shared) is None
    assert HYV4ForCausalLM._packed_expert_target(routed) == (
        "model.layers.1.mlp.experts.routed_experts.w2_weight",
        False,
    )


def test_hy4_mxfp8_indexer_wk_is_dequantized_into_fused_projection():
    param = torch.nn.Parameter(torch.zeros(4, 64, dtype=torch.bfloat16))

    def load_shard(target, weight, shard_id):
        assert shard_id == 0
        with torch.no_grad():
            target[: weight.shape[0]].copy_(weight)

    param.weight_loader = load_shard
    params = {"model.layers.0.self_attn.indexer.wk_weights_proj.weight": param}
    pending = {}
    loaded = set()
    prefix = "model.layers.0.self_attn.indexer.wk"
    scale = torch.tensor([[127, 128], [126, 127]], dtype=torch.uint8)
    weight = torch.ones(2, 64, dtype=torch.float8_e4m3fn)

    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight_scale", scale, pending, params, loaded
    )
    assert _try_load_mxfp8_indexer_wk(
        f"{prefix}.weight", weight, pending, params, loaded
    )

    expected = torch.tensor(
        [[1.0] * 32 + [2.0] * 32, [0.5] * 32 + [1.0] * 32],
        dtype=torch.bfloat16,
    )
    torch.testing.assert_close(param[:2], expected)
    assert not pending
    assert loaded == set(params)


def test_hy4_fp32_combine_reduce_order_and_world_size():
    """The final add stays FP32 on both late-reduce paths."""
    torch.manual_seed(17)
    routed = torch.randn(3, 5, dtype=torch.bfloat16)
    shared = torch.randn(3, 5, dtype=torch.bfloat16)
    peer_routed = torch.randn(3, 5, dtype=torch.bfloat16)
    peer_shared = torch.randn(3, 5, dtype=torch.bfloat16)

    calls: list[torch.Tensor] = []

    def mock_all_reduce(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.detach().clone())
        assert value.dtype == torch.float32
        return value + peer_routed.float() + peer_shared.float()

    result = _hyv4_fp32_combine_and_reduce(
        routed,
        shared,
        tp_size=2,
        ep_size=1,
        is_sequence_parallel=False,
        routed_output_is_global=False,
        shared_output_is_global=False,
        all_reduce=mock_all_reduce,
    )
    expected = (
        routed.float() + shared.float() + peer_routed.float() + peer_shared.float()
    )
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    late_calls = len(calls)

    # If routed output was reduced by the fused kernel, only the shared branch
    # is reduced here.  This checks the collective ordering contract.
    calls.clear()
    routed_global = (routed.float() + peer_routed.float()).to(torch.bfloat16)

    def mock_shared_all_reduce(value: torch.Tensor) -> torch.Tensor:
        calls.append(value.detach().clone())
        assert value.dtype == torch.float32
        return value + peer_shared.float()

    reduced = _hyv4_fp32_combine_and_reduce(
        routed_global,
        shared,
        tp_size=2,
        ep_size=1,
        is_sequence_parallel=False,
        routed_output_is_global=True,
        shared_output_is_global=False,
        all_reduce=mock_shared_all_reduce,
    )
    expected_reduced = routed_global.float() + shared.float() + peer_shared.float()
    torch.testing.assert_close(reduced, expected_reduced, rtol=0, atol=0)
    early_calls = len(calls)

    # TP=1/EP=1 must not issue a collective.
    no_collective = _hyv4_fp32_combine_and_reduce(
        routed,
        shared,
        tp_size=1,
        ep_size=1,
        is_sequence_parallel=False,
        routed_output_is_global=True,
        shared_output_is_global=True,
        all_reduce=lambda _: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    torch.testing.assert_close(no_collective, routed.float() + shared.float())

    preserved_fallback = _hyv4_fp32_combine_and_reduce(
        routed_global,
        shared,
        tp_size=2,
        ep_size=1,
        is_sequence_parallel=False,
        routed_output_is_global=True,
        shared_output_is_global=True,
        all_reduce=lambda _: (_ for _ in ()).throw(AssertionError("unexpected")),
    )
    torch.testing.assert_close(
        preserved_fallback, routed_global.float() + shared.float()
    )

    bf16_reference = (routed + peer_routed).to(torch.bfloat16).float() + (
        shared + peer_shared
    ).to(torch.bfloat16).float()
    diff = (result - bf16_reference).abs()
    print(
        "HY4 FP32 combine vs BF16-two-reduce "
        f"max_diff={diff.max().item():.8g} "
        f"mean_diff={diff.mean().item():.8g} "
        f"calls_late={late_calls} calls_early={early_calls}"
    )


def test_hy4_shared_expert_runner_flag_and_fp32_transform(monkeypatch):
    flag = "VLLM_HY4_SHARED_EXPERTS_RUNNER"
    monkeypatch.delenv(flag, raising=False)
    assert _hyv4_env_flag(flag) is False
    monkeypatch.setenv(flag, "1")
    assert _hyv4_env_flag(flag) is True
    monkeypatch.setenv(flag, "not-a-flag")
    assert _hyv4_env_flag(flag) is False

    value = torch.ones(2, 3, dtype=torch.bfloat16)
    transformed = _HYV4FP32RoutedOutput()(value)
    assert transformed.dtype == torch.float32
    torch.testing.assert_close(transformed, value.float())


def test_hy4_sequence_parallel_chunk_add_gather_and_truncate(monkeypatch):
    """Mock the DeepSeek SP contract without initializing distributed vLLM."""
    torch.manual_seed(23)
    original = torch.randn(5, 4, dtype=torch.bfloat16)
    calls: list[str] = []
    peer = torch.full((3, 4), 9.0, dtype=torch.float32)

    def mock_chunk(value: torch.Tensor) -> torch.Tensor:
        calls.append("chunk")
        return value[:3].clone()

    def mock_gather(value: torch.Tensor, dim: int) -> torch.Tensor:
        calls.append("all_gather")
        assert dim == 0
        assert value.dtype == torch.float32
        return torch.cat([value, peer], dim=0)

    class FakeGate:
        def __call__(self, value: torch.Tensor):
            calls.append("gate")
            return value.float(), None

    class FakeExperts:
        is_internal_router = False
        _fused_output_is_reduced = False
        moe_config = SimpleNamespace(tp_size=2, ep_size=1)

        def __call__(self, *, hidden_states, router_logits):
            del router_logits
            calls.append("routed")
            return (hidden_states + 1).to(hidden_states.dtype)

    class FakeShared:
        def __call__(self, value):
            calls.append("shared")
            return (value + 2).to(value.dtype)

    fake_moe = SimpleNamespace(
        is_sequence_parallel=True,
        _use_shared_experts_runner=False,
        gate=FakeGate(),
        experts=FakeExperts(),
        shared_experts=FakeShared(),
    )
    monkeypatch.setattr(hy_v4, "sequence_parallel_chunk", mock_chunk)
    monkeypatch.setattr(hy_v4, "tensor_model_parallel_all_gather", mock_gather)

    result = hy_v4.HYV4MoE.forward(fake_moe, original)
    local = original[:3]
    # Each fake branch keeps the custom-op/linear BF16 output contract; only
    # the branch add itself is required to happen in FP32.
    expected_local = (local + 1.0).float() + (local + 2.0).float()
    expected = torch.cat([expected_local, peer], dim=0)[:5].to(original.dtype)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.dtype == original.dtype
    assert calls == ["chunk", "gate", "routed", "shared", "all_gather"]


def test_hy4_moe_constructor_wires_fallback_runner_and_sp(monkeypatch):
    observed = []

    class FakeGate(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.e_score_correction_bias = None

    class FakeDense(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            observed.append(("dense", kwargs))

    class FakeFused(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            observed.append(("fused", kwargs))
            self.routed_experts = SimpleNamespace(local_num_experts=16)

    monkeypatch.setattr(hy_v4, "GateLinear", FakeGate)
    monkeypatch.setattr(hy_v4, "HYV4DenseMLP", FakeDense)
    monkeypatch.setattr(hy_v4, "FusedMoE", FakeFused)

    config = SimpleNamespace(
        hidden_act="silu",
        hidden_size=64,
        n_routed_experts=16,
        moe_intermediate_size=32,
        n_shared_experts=1,
        num_experts_per_tok=2,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        scoring_func="sigmoid",
        routed_scaling_factor=1.0,
        swiglu_limit=None,
    )

    def construct(*, runner: bool, sequence_parallel: bool):
        observed.clear()
        monkeypatch.setenv("VLLM_HY4_SHARED_EXPERTS_RUNNER", "1" if runner else "0")
        parallel = SimpleNamespace(
            use_sequence_parallel_moe=sequence_parallel,
            eplb_config=SimpleNamespace(num_redundant_experts=0),
            enable_eplb=False,
        )
        module = hy_v4.HYV4MoE(
            config,
            SimpleNamespace(parallel_config=parallel, quant_config=None),
            "model.layers.3.mlp",
        )
        dense = next(kwargs for kind, kwargs in observed if kind == "dense")
        fused = next(kwargs for kind, kwargs in observed if kind == "fused")
        return module, dense, fused

    fallback, dense, fused = construct(runner=False, sequence_parallel=False)
    assert dense["reduce_results"] is True
    assert dense["disable_tp"] is False
    assert fused["shared_experts"] is None
    assert fused["routed_output_transform"] is None
    assert fused["is_sequence_parallel"] is False
    assert fallback._use_shared_experts_runner is False

    runner, dense, fused = construct(runner=True, sequence_parallel=False)
    assert dense["reduce_results"] is False
    assert dense["disable_tp"] is False
    assert fused["shared_experts"] is runner.shared_experts
    assert isinstance(fused["routed_output_transform"], _HYV4FP32RoutedOutput)
    assert fused["is_sequence_parallel"] is False

    sp, dense, fused = construct(runner=False, sequence_parallel=True)
    assert dense["reduce_results"] is False
    assert dense["disable_tp"] is True
    assert fused["shared_experts"] is None
    assert fused["routed_output_transform"] is None
    assert fused["is_sequence_parallel"] is True
    assert sp.is_sequence_parallel is True


def test_hy4_dense_mlp_sequence_parallel_disables_tp(monkeypatch):
    linear_calls = []

    class FakeColumn(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            linear_calls.append(("column", kwargs["disable_tp"]))

    class FakeRow(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            linear_calls.append(("row", kwargs["disable_tp"], kwargs["reduce_results"]))

    monkeypatch.setattr(hy_v4, "MergedColumnParallelLinear", FakeColumn)
    monkeypatch.setattr(hy_v4, "RowParallelLinear", FakeRow)
    monkeypatch.setattr(hy_v4, "SiluAndMul", lambda: nn.Identity())
    hy_v4.HYV4DenseMLP(
        hidden_size=8,
        intermediate_size=16,
        hidden_act="silu",
        reduce_results=False,
        disable_tp=True,
    )
    assert linear_calls == [("column", True), ("row", True, False)]
