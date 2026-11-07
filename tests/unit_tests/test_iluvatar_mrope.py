from types import SimpleNamespace

import pytest
import torch

from vllm_fl.dispatch.backends.vendor.iluvatar.mrope import (
    _mrope_implementation,
    _wrap_mrope_forward,
    patch_mrope,
)


@pytest.fixture(autouse=True)
def isolated_environment_registry(monkeypatch):
    import vllm.envs

    monkeypatch.setattr(
        vllm.envs, "environment_variables", dict(vllm.envs.environment_variables)
    )
    monkeypatch.delenv("VLLM_FL_ILUVATAR_MROPE_IMPL", raising=False)


@pytest.mark.parametrize("vendor", ["iluvatar", "nvidia", "metax"])
def test_patch_is_vendor_scoped_and_idempotent(monkeypatch, vendor):
    import vllm.platforms
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    original = lambda *args: None
    monkeypatch.setattr(MRotaryEmbedding, "forward_native", original)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name=vendor)
    )
    patch_mrope()
    patched = MRotaryEmbedding.forward_native
    patch_mrope()
    assert MRotaryEmbedding.forward_native is patched
    assert (patched is original) is (vendor != "iluvatar")


def test_cpu_input_preserves_original_dispatch():
    calls = []

    def original(*args):
        calls.append(args)
        return args[2], args[3]

    module = SimpleNamespace(mrope_section=[11, 11, 10])
    positions = torch.zeros((3, 1), dtype=torch.int64)
    query = torch.ones((1, 256))
    key = query.clone()
    actual = _wrap_mrope_forward(original)(module, positions, query, key)
    assert actual[0] is query and actual[1] is key
    assert calls == [(module, positions, query, key, None)]


def test_implementation_changes_compilation_factors(monkeypatch):
    import vllm.envs
    import vllm.platforms
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    original = lambda *args: None
    monkeypatch.setattr(MRotaryEmbedding, "forward_native", original)
    monkeypatch.setattr(
        vllm.platforms, "current_platform", SimpleNamespace(vendor_name="iluvatar")
    )
    patch_mrope()
    assert vllm.envs.compile_factors()["VLLM_FL_ILUVATAR_MROPE_IMPL"] == "opaque-v1"
    monkeypatch.setenv("VLLM_FL_ILUVATAR_MROPE_IMPL", "native")
    patch_mrope()
    assert vllm.envs.compile_factors()["VLLM_FL_ILUVATAR_MROPE_IMPL"] == "native"
    assert MRotaryEmbedding.forward_native is original


def test_invalid_implementation_is_rejected(monkeypatch):
    monkeypatch.setenv("VLLM_FL_ILUVATAR_MROPE_IMPL", "unsupported")
    with pytest.raises(ValueError, match="VLLM_FL_ILUVATAR_MROPE_IMPL"):
        _mrope_implementation()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA device")
@pytest.mark.parametrize("interleaved", [False, True])
@pytest.mark.parametrize("neox", [False, True])
def test_mrope_matches_native_and_preserves_inputs(interleaved, neox):
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb
    from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

    torch.manual_seed(674)
    cache = torch.randn((128, 64), dtype=torch.bfloat16, device="cuda")
    module = SimpleNamespace(
        mrope_section=[11, 11, 10],
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=interleaved,
        is_neox_style=neox,
        _match_cos_sin_cache_dtype=lambda query: cache,
        apply_rotary_emb=SimpleNamespace(
            forward_native=lambda tensor, cos, sin: ApplyRotaryEmb.forward_static(
                tensor, cos, sin, neox
            )
        ),
    )
    positions = torch.tensor([[1, 119], [7, 3], [11, 41]], device="cuda")
    query = torch.randn((2, 3072), dtype=torch.bfloat16, device="cuda")[:, ::2]
    key = torch.randn((2, 512), dtype=torch.bfloat16, device="cuda")[:, ::2]
    query_before, key_before = query.clone(), key.clone()
    original = MRotaryEmbedding.forward_native
    expected = original(module, positions, query, key)
    actual = _wrap_mrope_forward(original)(module, positions, query, key)
    for output, reference, unchanged, source in zip(
        actual, expected, (query_before, key_before), (query, key)
    ):
        torch.testing.assert_close(output, reference, atol=0.0625, rtol=0.02)
        torch.testing.assert_close(source, unchanged, atol=0, rtol=0)
        torch.testing.assert_close(
            output.reshape(2, -1, 256)[..., 64:],
            source.reshape(2, -1, 256)[..., 64:],
            atol=0,
            rtol=0,
        )
