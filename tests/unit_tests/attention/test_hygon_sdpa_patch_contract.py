"""CPU numerics for the real Hygon adapter; only vLLM dependencies are stubbed."""

import importlib.machinery
import importlib.util
import os
import sys
import zipfile
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

PRODUCTION_SOURCE = Path(__file__).resolve().parents[3] / "vllm_fl/attention/utils.py"
WRAPPER_MEMBER = "vllm/v1/attention/ops/vit_attn_wrappers.py"


@pytest.fixture(scope="module")
def wrapper_source():
    wheel = os.environ.get("VLLM_SDPA_SOURCE_WHEEL")
    if wheel:
        with zipfile.ZipFile(wheel) as archive:
            return archive.read(WRAPPER_MEMBER).decode("utf-8")
    spec = importlib.machinery.PathFinder.find_spec("vllm", sys.path)
    for location in spec.submodule_search_locations if spec else ():
        source = Path(location) / "v1/attention/ops/vit_attn_wrappers.py"
        if source.is_file():
            return source.read_text(encoding="utf-8")
    return pytest.skip("vLLM source unavailable; supply VLLM_SDPA_SOURCE_WHEEL")


@pytest.fixture
def patched_sdpa(monkeypatch, wrapper_source):
    # Keep real Torch/einops. Stub only imports and the external op registry.
    modules = {}
    for leaf in (
        "vllm.logger",
        "vllm.platforms",
        "vllm._aiter_ops",
        "vllm.utils.gpu_sync_debug",
        "vllm.utils.torch_utils",
        "vllm.v1.attention.ops.vit_attn_wrappers",
        "vllm.v1.attention.backends.registry",
        "vllm.model_executor.layers.attention.mm_encoder_attention",
        "vllm_fl.attention",
    ):
        parts = leaf.split(".")
        for length in range(1, len(parts) + 1):
            name = ".".join(parts[:length])
            if name not in modules:
                modules[name] = ModuleType(name)
                modules[name].__path__ = []
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent, attribute = name.rsplit(".", 1)
            setattr(modules[parent], attribute, module)
    modules["vllm.platforms"].current_platform = SimpleNamespace(
        vendor_name="hygon", is_rocm=lambda: False, is_cuda=lambda: False
    )
    modules["vllm.logger"].init_logger = lambda _: SimpleNamespace(
        info_once=lambda *args: None
    )
    modules["vllm._aiter_ops"].rocm_aiter_ops = SimpleNamespace()
    modules["vllm.utils.gpu_sync_debug"].gpu_sync_allowed = nullcontext
    modules[
        "vllm.v1.attention.backends.registry"
    ].AttentionBackendEnum = SimpleNamespace(FLASH_ATTN="flash", ROCM_AITER_FA="aiter")
    registered = {}
    modules["vllm.utils.torch_utils"].direct_register_custom_op = (
        lambda op_name, op_func, **kwargs: registered.setdefault(op_name, op_func)
    )
    vit = modules["vllm.v1.attention.ops.vit_attn_wrappers"]
    exec(compile(wrapper_source, WRAPPER_MEMBER, "exec"), vit.__dict__)
    original_apply = vit.apply_sdpa
    spec = importlib.util.spec_from_file_location(
        "vllm_fl.attention.utils", PRODUCTION_SOURCE
    )
    adapter = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, adapter)
    spec.loader.exec_module(adapter)
    adapter.patch_mm_encoder_attention()
    return SimpleNamespace(
        vit=vit,
        wrapper=registered["torch_sdpa_wrapper"],
        original_apply=original_apply,
        patch=adapter.patch_mm_encoder_attention,
    )


def tensors(dim=72, value_dim=72, kv_heads=4, dtype=torch.float32):
    generator = torch.Generator(device="cpu").manual_seed(23)
    return tuple(
        torch.randn(
            2, 9, heads, width * 2, generator=generator, dtype=dtype, device="cpu"
        )[..., ::2]
        for heads, width in ((4, dim), (kv_heads, dim), (kv_heads, value_dim))
    )


def unpadded_sdpa(q, k, v, scale=None, enable_gqa=False):
    return F.scaled_dot_product_attention(
        *(tensor.transpose(1, 2) for tensor in (q, k, v)),
        dropout_p=0.0,
        scale=scale,
        enable_gqa=enable_gqa,
    ).transpose(1, 2)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "dim,value_dim,scale,enable_gqa",
    [
        (72, 72, None, False),
        (72, 72, 0.37, False),
        (72, 72, 0.0, False),
        (72, 72, -0.5, False),
        (72, 72, None, True),
        (72, 72, 0.37, True),
        (64, 64, None, False),
        (96, 96, None, False),
        (128, 128, None, False),
        (72, 96, None, False),
        (128, 72, None, False),
    ],
)
def test_patched_sdpa_matches_unpadded_cpu_reference(
    patched_sdpa, dim, value_dim, scale, enable_gqa, dtype
):
    inputs = tensors(dim, value_dim, kv_heads=2 if enable_gqa else 4, dtype=dtype)
    before = [(tensor.clone(), tensor.stride(), tensor.data_ptr()) for tensor in inputs]
    assert all(not tensor.is_contiguous() for tensor in inputs)
    output = patched_sdpa.wrapper(*inputs, scale=scale, enable_gqa=enable_gqa)
    expected = unpadded_sdpa(*inputs, scale=scale, enable_gqa=enable_gqa)
    assert output.shape == (2, 9, 4, value_dim)
    assert output.dtype == dtype and output.device.type == "cpu"
    torch.testing.assert_close(output, expected)
    for tensor, (values, stride, pointer) in zip(inputs, before):
        torch.testing.assert_close(tensor, values, rtol=0, atol=0)
        assert tensor.stride() == stride and tensor.data_ptr() == pointer


def test_pre_registered_wrapper_uses_patched_helper(patched_sdpa, monkeypatch):
    vit = patched_sdpa.vit
    helper = vit.apply_sdpa
    assert helper is not patched_sdpa.original_apply
    patched_sdpa.patch()
    assert vit.apply_sdpa is helper
    assert patched_sdpa.wrapper is vit.torch_sdpa_wrapper
    assert patched_sdpa.wrapper.__globals__ is vit.__dict__
    calls = []

    def tracked_helper(*args, **kwargs):
        calls.append(kwargs)
        return helper(*args, **kwargs)

    monkeypatch.setattr(vit, "apply_sdpa", tracked_helper)
    monkeypatch.setattr(
        vit,
        "torch_sdpa_wrapper",
        lambda *args, **kw: pytest.fail(
            "The pre-registered callable must not use a reassigned wrapper name"
        ),
    )
    inputs = tensors()
    output = patched_sdpa.wrapper(*inputs, scale=0.37)
    assert calls == [{"scale": 0.37, "enable_gqa": False}]
    torch.testing.assert_close(output, unpadded_sdpa(*inputs, scale=0.37))


def test_cu_seqlens_preserves_three_independent_segments(patched_sdpa):
    inputs = tensors(kv_heads=2)
    lengths = [2, 3, 4]
    output = patched_sdpa.wrapper(
        *inputs,
        cu_seqlens=torch.tensor([0, 2, 5, 9], dtype=torch.int32, device="cpu"),
        scale=0.125,
        enable_gqa=True,
    )
    expected = torch.cat(
        [
            unpadded_sdpa(q, k, v, scale=0.125, enable_gqa=True)
            for q, k, v in zip(*(tensor.split(lengths, dim=1) for tensor in inputs))
        ],
        dim=1,
    )
    assert output.shape == (2, 9, 4, 72)
    torch.testing.assert_close(output, expected)
