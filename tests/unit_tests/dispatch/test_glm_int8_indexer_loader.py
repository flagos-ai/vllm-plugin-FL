"""CPU contract tests; no model download or device initialization required."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_path = (Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/"
         "vendor/hygon/patches/models/glm_int8_indexer.py")
_spec = importlib.util.spec_from_file_location("glm_int8_loader_test", _path)
adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(adapter)

CONFIG = {"config_groups": {"W8A8": {"weights": {
    "num_bits": 8, "type": "int", "strategy": "channel", "symmetric": True,
}}}}
PREFIX = "layers.0.self_attn.indexer"
W = PREFIX + ".wk.weight"
S = PREFIX + ".wk.weight_scale"


@pytest.mark.parametrize("scale_first", [False, True])
def test_channel_values_and_fused_shards(scale_first):
    weight = torch.tensor([[-128, -1, 0], [1, 64, 127]], dtype=torch.int8)
    scale = torch.tensor([[0.125], [0.25]], dtype=torch.float32)
    other = torch.tensor([[3., 4., 5.]], dtype=torch.bfloat16)
    pair = [(W, weight), (S, scale)]
    if scale_first:
        pair.reverse()
    result = dict(adapter._convert_int8_wk(
        [pair[0], (PREFIX + ".weights_proj.weight", other), pair[1]], CONFIG))
    expected = torch.tensor([[-16., -.125, 0.], [.25, 16., 31.75]],
                            dtype=torch.bfloat16)
    assert set(result) == {W, PREFIX + ".weights_proj.weight"}
    assert torch.equal(result[W], expected)
    assert result[PREFIX + ".weights_proj.weight"] is other
    fused = torch.cat([result[W], other], dim=0)
    assert torch.equal(fused[:2], expected)
    assert torch.equal(fused[2:], other)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_other_dtypes_passthrough(dtype):
    tensor = torch.zeros(2, 3).to(dtype)
    scale = torch.ones(1, 1)
    pairs = [(W, tensor), (PREFIX + ".wk.weight_scale_inv", scale)]
    result = list(adapter._convert_int8_wk(pairs, None))
    assert all(a[0] == b[0] and a[1] is b[1] for a, b in zip(result, pairs))


@pytest.mark.parametrize("scale", [
    torch.ones(2), torch.ones(1, 3), torch.ones(2, 1, dtype=torch.int32),
    torch.full((2, 1), float("nan")), torch.zeros(2, 1),
    -torch.ones(2, 1), torch.full((2, 1), float("inf")),
])
def test_bad_scale_rejected(scale):
    with pytest.raises(ValueError):
        list(adapter._convert_int8_wk([(W, torch.ones(2, 3, dtype=torch.int8)),
                                     (S, scale)], CONFIG))


@pytest.mark.parametrize("missing", ["weight", "scale"])
def test_incomplete_pair_rejected(missing):
    pairs = [(W, torch.ones(2, 3, dtype=torch.int8))] if missing == "scale" else [
        (S, torch.ones(2, 1))]
    with pytest.raises(ValueError, match="Incomplete"):
        list(adapter._convert_int8_wk(pairs, CONFIG))


def test_asymmetric_rejected():
    config = {"config_groups": {"W8A8": {"weights": {
        **CONFIG["config_groups"]["W8A8"]["weights"], "symmetric": False}}}}
    with pytest.raises(ValueError, match="symmetric channel"):
        list(adapter._convert_int8_wk([(S, torch.ones(2, 1))], config))


def test_duplicate_rejected():
    pair = (W, torch.ones(2, 3, dtype=torch.int8))
    with pytest.raises(ValueError, match="Duplicate"):
        list(adapter._convert_int8_wk([pair, pair], CONFIG))


def test_patch_scope_and_idempotence():
    class Model:
        def load_weights(self, weights):
            return list(weights)

    adapter._install_glm_int8_indexer_loader(Model)
    patched = Model.load_weights
    adapter._install_glm_int8_indexer_loader(Model)
    assert Model.load_weights is patched
    model = Model()
    model.config = SimpleNamespace(model_type="deepseek_v3")
    incomplete = [(S, torch.ones(2, 1))]
    assert model.load_weights(incomplete) is not None
    model.config = SimpleNamespace(model_type="glm_moe_dsa",
                                   quantization_config=CONFIG)
    with pytest.raises(ValueError, match="Incomplete"):
        model.load_weights(incomplete)
