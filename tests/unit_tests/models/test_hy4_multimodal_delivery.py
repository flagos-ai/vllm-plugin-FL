# SPDX-License-Identifier: Apache-2.0
"""CPU processor semantics and isolated model/normalization contracts.

The processor and configuration imports below use real Transformers/Torch.
Selected model methods and normalization registration use small host fixtures;
these contracts do not qualify the vision tower, CUDA kernels or serving.
"""

import ast
import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / "vllm_fl/models/hy_v4"


def load_module(name, path, monkeypatch):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def model_weights(monkeypatch):
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(
        info_once=lambda *args: None,
        warning=lambda *args: None,
    )
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    load_module("hy4_weights_fixture.mtp_config", MODEL / "mtp_config.py", monkeypatch)
    original_import = builtins.__import__

    def import_without_numerical_backend(name, *args, **kwargs):
        if name.startswith(
            (
                "vllm.model_executor.layers.quantization",
                "vllm_fl.models.hy_v4.nvidia",
                "flaggems_vllm",
            )
        ):
            raise AssertionError(f"Unexpected numerical backend import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_numerical_backend)
    return load_module(
        "hy4_weights_fixture.model_weights", MODEL / "model_weights.py", monkeypatch
    )


def test_packed_mappings_are_independent_for_target_draft_and_vision(model_weights):
    mappings = {}
    for relative, names in (
        ("nvidia/model.py", {"HYV4ForCausalLM", "HYV4MTP"}),
        ("multimodal.py", {"HYV4VLForConditionalGeneration"}),
    ):
        for node in ast.parse((MODEL / relative).read_text()).body:
            if not isinstance(node, ast.ClassDef) or node.name not in names:
                continue
            assignment = next(
                field
                for field in node.body
                if isinstance(field, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "packed_modules_mapping"
                    for target in field.targets
                )
            )
            expression = ast.Expression(body=assignment.value)
            mappings[node.name] = eval(
                compile(expression, str(MODEL / relative), "eval"),
                {
                    "hyv4_packed_modules_mapping": model_weights.hyv4_packed_modules_mapping
                },
            )
    target = mappings["HYV4ForCausalLM"]
    draft = mappings["HYV4MTP"]
    vision = mappings["HYV4VLForConditionalGeneration"]
    assert (
        target
        == draft
        == {
            "gate_up_proj": ["gate_proj", "up_proj"],
            "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"],
            "wk_weights_proj": ["wk", "weights_proj"],
        }
    )
    assert vision == {**target, "qkv": ["q_proj", "k_proj", "v_proj"]}
    target["gate_up_proj"].append("target-only")
    target["target-only"] = []
    assert draft == model_weights.hyv4_packed_modules_mapping()
    assert vision["gate_up_proj"] == ["gate_proj", "up_proj"]
    assert "target-only" not in draft and "target-only" not in vision


def test_checkpoint_scale_names_and_quant_exclusions_preserve_target(model_weights):
    config = SimpleNamespace(num_hidden_layers=78, num_nextn_predict_layers=2)
    assert (
        model_weights._get_spec_layer_idx_from_weight_name(
            config, "model.layers.78.mlp.experts.gate_up_proj_scale_inv"
        )
        == 78
    )
    assert (
        model_weights._get_spec_layer_idx_from_weight_name(
            config, "model.layers.79.mlp.experts.gate_up_proj_scale_inv"
        )
        is None
    )
    parameter = "model.layers.78.mlp.experts.routed_experts.w13_weight"
    assert (
        model_weights._resolve_fused_expert_param(
            parameter, "_scale_inv", {f"{parameter}_scale_inv": object()}
        )
        == f"{parameter}_scale_inv"
    )
    original = SimpleNamespace(
        get_name=lambda: "fp8",
        ignored_layers=["model.mtp_layers.0.self_attn.linear_gate"],
    )
    mapped = model_weights._remap_mtp_quant_exclusions(original, 78, 1)
    assert mapped is not original
    assert mapped.ignored_layers == [
        "model.mtp_layers.0.self_attn.linear_gate",
        "model.layers.78.self_attn.linear_gate",
    ]
    assert original.ignored_layers == ["model.mtp_layers.0.self_attn.linear_gate"]


@pytest.fixture
def processor(monkeypatch):
    config = load_module(
        "vllm_fl.models.hy_v4.config", MODEL / "config.py", monkeypatch
    )
    module = load_module("hy4_processor_contract", MODEL / "processor.py", monkeypatch)
    module.contract_config = config
    return module


def test_nested_and_flat_multimodal_config_keeps_vision_parameters(processor):
    config = processor.contract_config.HYV4VLConfig(
        hidden_size=32,
        text_config={"hidden_size": 16},
        vision_config={"patch_size": 4, "spatial_merge_size": 2},
        image_token_id=701,
    )
    assert config.text_config.hidden_size == 16
    assert config.vision_config.out_hidden_size == 16
    assert config.vision_config.patch_size == 4
    assert config.image_token_id == config.image_token_index == 701
    assert config.architectures == ["HYV4VLForConditionalGeneration"]
    with pytest.raises(ValueError, match="projector output"):
        processor.contract_config.HYV4VLConfig(
            text_config={"hidden_size": 16}, vision_config={"out_hidden_size": 8}
        )


def test_patchify_preserves_frame_patch_channel_and_pixel_order(processor):
    frames = torch.arange(2 * 3 * 4 * 6).reshape(2, 3, 4, 6)
    actual = processor.patchify(frames, 2)
    expected = []
    for frame in range(2):
        for row in range(0, 4, 2):
            for column in range(0, 6, 2):
                expected.append(
                    [
                        int(frames[frame, channel, row + dy, column + dx])
                        for channel in range(3)
                        for dy in range(2)
                        for dx in range(2)
                    ]
                )
    assert torch.equal(actual, torch.tensor(expected))
    with pytest.raises(ValueError, match="divisible"):
        processor.patchify(frames[..., :5], 2)


def test_image_processor_restores_input_order_after_shape_grouping(processor):
    image_processor = processor.HYV4VLImageProcessor(
        patch_size=2,
        merge_size=1,
        size={"shortest_edge": 1, "longest_edge": 1024},
    )
    images = [
        torch.arange(3 * 4 * 6, dtype=torch.uint8).reshape(3, 4, 6),
        torch.full((3, 2, 4), 8, dtype=torch.uint8),
        torch.full((3, 4, 6), 16, dtype=torch.uint8),
    ]
    result = image_processor(
        images,
        do_resize=False,
        do_rescale=False,
        do_normalize=False,
        return_tensors="pt",
    )
    assert torch.equal(
        result["image_grid_thw"], torch.tensor([[1, 2, 3], [1, 1, 2], [1, 2, 3]])
    )
    assert result["pixel_values"].shape == (14, 12)
    assert torch.equal(
        result["pixel_values"][6:8], torch.full((2, 12), 8, dtype=torch.uint8)
    )
    assert torch.equal(
        result["pixel_values"][8:], torch.full((6, 12), 16, dtype=torch.uint8)
    )


@pytest.mark.parametrize("height,width", [(37, 61), (512, 512), (1024, 2048)])
def test_smart_resize_respects_patch_alignment_and_pixel_budget(
    processor, height, width
):
    rows, columns = processor.smart_resize(
        height, width, factor=32, min_pixels=4096, max_pixels=262144
    )
    assert rows % 32 == columns % 32 == 0
    assert 4096 <= rows * columns <= 262144


def test_smart_resize_rejects_invalid_shape_and_aspect_ratio(processor):
    with pytest.raises(ValueError, match="positive"):
        processor.smart_resize(0, 16)
    with pytest.raises(ValueError, match="aspect ratio"):
        processor.smart_resize(1, 1024)


def test_image_and_video_token_grids_match_merger_geometry(processor):
    config = processor.contract_config.HYV4VLVisionConfig(
        spatial_merge_size=2,
        spatial_patch_size=2,
        temporal_patch_size=2,
        cat_extra_token=1,
        use_image_newline_token=True,
    )
    assert processor.vision_token_grid((1, 8, 12), config, "image") == (1, 4, 7)
    assert processor.vision_token_count((1, 8, 12), config, "image") == 30
    assert processor.vision_token_grid((3, 8, 12), config, "video") == (2, 2, 4)
    assert processor.vision_token_count((3, 8, 12), config, "video") == 20
    with pytest.raises(ValueError, match="Single-frame"):
        processor.vision_token_grid((1, 8, 12), config, "video")
    with pytest.raises(ValueError, match="exactly one"):
        processor.vision_token_grid((2, 8, 12), config, "image")
    with pytest.raises(ValueError, match="divisible"):
        processor.vision_token_grid((1, 7, 12), config, "image")


def model_method(name, processor):
    """Execute the production method body with host-only interface fixtures."""
    source = MODEL / "multimodal.py"
    cls = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef)
        and node.name == "HYV4VLForConditionalGeneration"
    )
    node = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )

    class Features:
        @staticmethod
        def gather_kwargs(features, keys):
            gathered = {}
            for feature in features:
                for key, value in feature.items():
                    if key in keys:
                        gathered.setdefault(key, []).append(value)
            return gathered

    namespace = {
        "torch": torch,
        "MultiModalFeatureSpec": Features,
        "vision_token_grid": processor.vision_token_grid,
        "vision_token_count": processor.vision_token_count,
        "MultiModalEmbeddings": tuple,
    }
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"),
        namespace,
    )
    return namespace[name]


def test_mrope_image_grid_keeps_text_boundaries_and_spatial_axes(processor):
    config = processor.contract_config.HYV4VLConfig()
    start, end, pad = (
        config.image_start_token_id,
        config.image_end_token_id,
        config.image_token_id,
    )
    tokens = [17, start, *([pad] * 6), end, 19]
    method = model_method("get_mrope_input_positions", processor)
    actual, delta = method(
        SimpleNamespace(config=config),
        tokens,
        [{"image_grid_thw": torch.tensor([1, 4, 4])}],
    )
    expected = torch.tensor(
        [
            [0, 1, 2, 2, 2, 2, 2, 2, 5, 6],
            [0, 1, 2, 2, 2, 3, 3, 3, 5, 6],
            [0, 1, 2, 3, 4, 2, 3, 4, 5, 6],
        ]
    )
    assert torch.equal(actual, expected)
    assert delta == -3
    with pytest.raises(ValueError, match="paired"):
        method(
            SimpleNamespace(config=config),
            tokens[:-2],
            [{"image_grid_thw": torch.tensor([1, 4, 4])}],
        )
    with pytest.raises(ValueError, match="placeholders"):
        method(
            SimpleNamespace(config=config),
            tokens,
            [{"image_grid_thw": torch.tensor([1, 4, 6])}],
        )


def test_mrope_text_only_and_unused_media_grid_boundaries(processor):
    config = processor.contract_config.HYV4VLConfig()
    method = model_method("get_mrope_input_positions", processor)
    positions, delta = method(SimpleNamespace(config=config), [5, 6, 7], [])
    assert torch.equal(positions, torch.tensor([[0, 1, 2]]).repeat(3, 1))
    assert delta == 0
    with pytest.raises(ValueError, match="Unused"):
        method(
            SimpleNamespace(config=config),
            [5, 6],
            [{"image_grid_thw": torch.tensor([1, 4, 4])}],
        )


def test_mrope_video_frames_keep_separate_temporal_positions(processor):
    config = processor.contract_config.HYV4VLConfig()
    tokens = [
        config.video_start_token_id,
        config.image_start_token_id,
        *([config.video_token_id] * 6),
        config.image_end_token_id,
        91,
        config.image_start_token_id,
        *([config.video_token_id] * 6),
        config.image_end_token_id,
        config.video_end_token_id,
    ]
    method = model_method("get_mrope_input_positions", processor)
    positions, delta = method(
        SimpleNamespace(config=config),
        tokens,
        [{"video_grid_thw": torch.tensor([2, 4, 4])}],
    )
    assert positions[0, 2:8].tolist() == [2] * 6
    assert positions[0, 11:17].tolist() == [8] * 6
    assert positions[1, 11:17].tolist() == [8, 8, 8, 9, 9, 9]
    assert positions[2, 11:17].tolist() == [8, 9, 10, 8, 9, 10]
    assert positions[:, 8:11].tolist() == [[5, 6, 7]] * 3
    assert delta == -6


def normalization_module(monkeypatch):
    class CustomOp(torch.nn.Module):
        @classmethod
        def register(cls, name):
            return lambda implementation: implementation

    for name, attributes in {
        "vllm.config": {"ModelConfig": SimpleNamespace},
        "vllm.model_executor.custom_op": {"CustomOp": CustomOp},
        "vllm.transformers_utils.processor": {
            "get_processor_config": lambda *a, **kw: {}
        },
    }.items():
        module = ModuleType(name)
        vars(module).update(attributes)
        monkeypatch.setitem(sys.modules, name, module)
    return load_module(
        "hy4_normalization_contract", MODEL / "mm_input_norm.py", monkeypatch
    )


def test_processor_normalization_default_needs_no_optional_provider(monkeypatch):
    module = normalization_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "_normalization_provider",
        lambda: pytest.fail("provider must stay unloaded"),
    )
    normalizer = module.build_mm_input_norm(SimpleNamespace(multimodal_config=None))
    x = torch.tensor([[1, 2, 3]], dtype=torch.float32)
    assert isinstance(normalizer, module.IdentityInputNorm)
    assert normalizer(x, torch.float32) is x


def test_device_normalization_requires_selected_provider_before_construction(
    monkeypatch,
):
    module = normalization_module(monkeypatch)
    monkeypatch.setattr(
        module,
        "get_processor_config",
        lambda *a, **kw: {"image_mean": [0.5] * 3, "image_std": [0.25] * 3},
    )
    calls = []

    def missing_provider():
        calls.append("select")
        raise ValueError("provider disabled")

    monkeypatch.setattr(module, "_normalization_provider", missing_provider)
    config = SimpleNamespace(
        model="fixture",
        revision=None,
        multimodal_config=SimpleNamespace(mm_device_do_normalize=True),
    )
    with pytest.raises(ValueError, match="provider disabled"):
        module.build_mm_input_norm(config)
    assert calls == ["select"]


def test_normalization_provider_honors_common_policy_before_optional_import(
    monkeypatch,
):
    module = normalization_module(monkeypatch)
    policy, imports = [], []
    enabled = False

    def use_flaggems_op(name, default=False):
        policy.append((name, default))
        return enabled

    utils = ModuleType("vllm_fl.utils")
    utils.use_flaggems_op = use_flaggems_op
    monkeypatch.setitem(sys.modules, "vllm_fl.utils", utils)
    implementation = object()

    class Providers(ModuleType):
        def __getattr__(self, name):
            if name == "fused_mm_input_norm":
                imports.append(name)
                return implementation
            raise AttributeError(name)

    monkeypatch.setitem(sys.modules, "flaggems_vllm", Providers("flaggems_vllm"))
    with pytest.raises(ValueError, match="FlagGems provider"):
        module._normalization_provider()
    assert imports == []
    enabled = True
    assert module._normalization_provider() is implementation
    assert policy == [("fused_mm_input_norm", False)] * 2


def test_normalization_provider_error_propagates_without_retry(monkeypatch):
    module = normalization_module(monkeypatch)
    calls = []

    def provider(*args):
        calls.append(args)
        raise RuntimeError("fixture launch failure")

    monkeypatch.setattr(module, "_normalization_provider", lambda: provider)
    normalizer = module.FusedMMInputNorm([0.5] * 3, [0.25] * 3, 1 / 255)
    x = torch.ones((2, 12), dtype=torch.uint8)
    with pytest.raises(RuntimeError, match="fixture launch failure"):
        normalizer.forward_native(x, torch.bfloat16)
    assert len(calls) == 1
    inputs, output, weight, bias = calls[0]
    assert inputs.shape == output.shape == (2, 3, 4)
    assert output.dtype == torch.bfloat16
    assert weight.dtype == bias.dtype == torch.float32
    assert torch.equal(x, torch.ones_like(x))
