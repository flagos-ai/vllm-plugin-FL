# SPDX-License-Identifier: Apache-2.0
"""Capability-gated indexer storage, numerical and replay contracts."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from vllm.platforms.interface import DeviceCapability

from vllm_fl.models.glm5 import compile_ops, indexer_backend
from vllm_fl.platform import PlatformFL


@pytest.mark.parametrize(
    "vendor,capability,expected",
    [
        ("nvidia", (8, 0), False),
        ("nvidia", (8, 6), False),
        ("nvidia", (8, 9), True),
        ("nvidia", (9, 0), True),
        ("nvidia", None, False),
        ("hygon", (9, 0), False),
        ("mthreads", (3, 1), True),
        ("mthreads", (2, 2), False),
        ("gcu", (4, 0), True),
        ("gcu", (3, 0), False),
        ("enflame", (4, 0), True),
        ("enflame", (3, 0), False),
        ("unknown", (9, 0), False),
    ],
)
def test_fp8_capability_uses_device_architecture(
    monkeypatch, vendor, capability, expected
):
    monkeypatch.setattr(PlatformFL, "vendor_name", vendor)
    monkeypatch.setattr(
        PlatformFL,
        "get_device_capability",
        classmethod(lambda cls: DeviceCapability(*capability) if capability else None),
    )
    assert PlatformFL.supports_fp8() is expected


@pytest.fixture
def bf16_backend(monkeypatch):
    from vllm.platforms import current_platform

    monkeypatch.setattr(current_platform, "supports_fp8", lambda: False)
    backend = indexer_backend.Glm5NextIndexerBackend()
    monkeypatch.setattr(indexer_backend, "INDEXER_BACKEND", backend)
    original = indexer_backend._load_flaggems_op

    def resolve(module, name):
        assert "fp8" not in name and "quant" not in name
        return original(module, name)

    monkeypatch.setattr(indexer_backend, "_load_flaggems_op", resolve)
    return backend


def _hadamard_reference(x):
    matrix = torch.ones(1, 1, dtype=torch.float64)
    for _ in range(7):
        matrix = torch.cat(
            (torch.cat((matrix, matrix), 1), torch.cat((matrix, -matrix), 1)), 0
        )
    return (x.cpu().double() @ matrix / 128**0.5).to(torch.bfloat16).to(x.device)


def test_bf16_fake_and_preflight_never_resolve_fp8(bf16_backend):
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        values, scale = compile_ops.fwht128_quant_fp8(
            torch.empty(3, 128, dtype=torch.bfloat16)
        )
    assert values.dtype == torch.bfloat16 and values.shape == (3, 128)
    assert scale.dtype == torch.float32 and scale.shape == (3, 1)
    descriptions = bf16_backend.preflight_indexer()
    assert descriptions
    assert all("fp8" not in name for name in bf16_backend._bindings)


@pytest.mark.gpu
def test_bf16_query_compile_and_graph_replay(bf16_backend):
    device = indexer_backend.current_platform.device_type
    q = torch.randn(7, 128, device=device, dtype=torch.bfloat16)
    compiled = torch.compile(compile_ops.fwht128_quant_fp8, fullgraph=True)
    for _ in range(3):
        actual, scale = compiled(q)
    torch.testing.assert_close(actual, _hadamard_reference(q), rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual, scale = compiled(q)
    for multiplier in (0.5, -2):
        q.mul_(multiplier)
        actual.fill_(torch.nan)
        scale.fill_(torch.nan)
        graph.replay()
        torch.testing.assert_close(actual, _hadamard_reference(q), rtol=0, atol=0)
        torch.testing.assert_close(scale, torch.ones_like(scale), rtol=0, atol=0)


@pytest.mark.gpu
def test_bf16_pool_tail_decode_and_padded_cache_replay(bf16_backend):
    device = indexer_backend.current_platform.device_type
    torch.manual_seed(132)
    # A physical page has padding; flattening it would silently write a copy.
    storage = torch.full((3, 32, 136), -13, device=device, dtype=torch.bfloat16)
    cache = storage[:, :, :128]
    tail = torch.zeros(3, 2, 4, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(3, 4, 128, device=device, dtype=torch.bfloat16)
    scores = torch.randn_like(keys)
    ape = torch.randn(4, 128, device=device)
    slots = torch.tensor([31, 33, -1], device=device)
    bf16_backend.kpool_compress_and_write_cache(cache, keys, scores, ape, slots, 4)
    gates = torch.softmax(scores.cpu().float() + ape.cpu(), 1)
    pooled = (keys.cpu().float() * gates).sum(1).to(torch.bfloat16)
    expected = _hadamard_reference(pooled).to(device)
    torch.testing.assert_close(cache[0, 31], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(cache[1, 1], expected[1], rtol=0, atol=0)
    assert (storage[:, :, 128:] == -13).all()
    assert (cache[2] == -13).all()

    tail[0, 0, :3] = keys[0, :3]
    tail[0, 1, :3] = scores[0, :3]
    # Second request is graph padding and must not write its tail/cache.
    dec_keys, dec_scores = keys[:2, 3:].clone(), scores[:2, 3:].clone()
    tail_slots = torch.tensor([[3], [-1]], device=device)
    positions = torch.tensor([[3], [-1]], device=device)
    cache_slots = torch.tensor([[34], [-1]], device=device)

    def run():
        bf16_backend.kpool_decode_update_and_maybe_write_cache_batched(
            cache,
            tail,
            tail_slots,
            dec_keys,
            dec_scores,
            ape,
            cache_slots,
            positions,
            4,
        )

    for _ in range(3):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for multiplier in (0.5, -1):
        dec_keys.mul_(multiplier)
        cache[1, 2].fill_(torch.nan)
        graph.replay()
        ref_keys = keys[0].cpu().float()
        ref_keys[3] = dec_keys[0, 0].cpu().float()
        ref_pool = (ref_keys * gates[0]).sum(0).to(torch.bfloat16)
        torch.testing.assert_close(
            cache[1, 2], _hadamard_reference(ref_pool).to(device), rtol=0, atol=0
        )
        assert (tail[1:] == 0).all()
        assert (storage[:, :, 128:] == -13).all()


@pytest.mark.gpu
@pytest.mark.parametrize("page_size", [32, 64])
def test_bf16_gather_and_logits_match_independent_reference(bf16_backend, page_size):
    device = indexer_backend.current_platform.device_type
    torch.manual_seed(251)
    raw = torch.randn(5, page_size, 136, device=device, dtype=torch.bfloat16)
    cache = raw[:, :, :128]
    table = torch.tensor([[3, 1], [2, -1]], device=device, dtype=torch.int32)
    lengths = torch.tensor([[page_size + 7], [17]], device=device, dtype=torch.int32)
    query = torch.randn(2, 1, 3, 128, device=device, dtype=torch.bfloat16)
    weights = torch.randn(2, 3, device=device)
    actual = bf16_backend.paged_mqa_logits(
        (query, None),
        cache.unsqueeze(2),
        weights,
        lengths,
        table,
        None,
        max_model_len=page_size * 2 + 11,
        clean_logits=True,
    )
    for row in range(2):
        length = int(lengths[row, 0])
        flat = torch.cat([cache[int(page)] for page in table[row] if page >= 0])[
            :length
        ]
        ref = torch.einsum(
            "hd,nd->hn", query[row, 0].cpu().double(), flat.cpu().double()
        ).relu()
        ref = (ref * weights[row].cpu().double()[:, None]).sum(0)
        torch.testing.assert_close(
            actual[row, :length].cpu().double(), ref, rtol=2e-5, atol=2e-5
        )
        assert torch.isneginf(actual[row, length:]).all()
    cu = torch.tensor(
        [0, page_size + 7, page_size + 24], device=device, dtype=torch.int32
    )
    values = torch.empty(page_size + 24, 128, device=device, dtype=torch.bfloat16)
    scales = torch.empty(page_size + 24, 4, device=device, dtype=torch.uint8)
    bf16_backend.cp_gather_indexer_k_quant_cache(cache, values, scales, table, cu)
    expected = torch.cat((cache[3], cache[1, :7], cache[2, :17]))
    torch.testing.assert_close(values, expected, rtol=0, atol=0)
    assert (scales.view(torch.float32) == 1).all()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("glm_layout", [False, True])
def test_portable_metadata_does_not_launch_deepgemm(monkeypatch, native, glm_layout):
    from types import SimpleNamespace

    from vllm.v1.attention.backends.mla import indexer as upstream

    from vllm_fl.models.glm5 import cache, provider

    monkeypatch.setattr(provider, "use_nvidia_reference", lambda: native)
    monkeypatch.setattr(upstream, "has_deep_gemm", lambda: True)
    original = upstream.has_deep_gemm
    calls = []

    def build(*args):
        calls.append(upstream.has_deep_gemm())
        return "metadata"

    monkeypatch.setattr(cache, "_runtime_pristine", lambda *args: build)
    patch = cache._indexer_build_patch("test.bf16.metadata")
    builder = SimpleNamespace(kv_cache_spec=SimpleNamespace(block_size=64))
    if glm_layout:
        builder._glm5_physical_layout = object()
    assert patch.replacement(builder, 0, None) == "metadata"
    assert calls == [native or not glm_layout]
    assert upstream.has_deep_gemm is original

    def broken(*args):
        assert upstream.has_deep_gemm() is (native or not glm_layout)
        raise ValueError("metadata failure")

    monkeypatch.setattr(cache, "_runtime_pristine", lambda *args: broken)
    with pytest.raises(ValueError, match="metadata failure"):
        cache._indexer_build_patch("test.bf16.metadata").replacement(builder, 0, None)
    assert upstream.has_deep_gemm is original


def test_nvidia_capability_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(PlatformFL, "vendor_name", "nvidia")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda *_: pytest.fail("CUDA initialization"),
    )
    calls = []

    def capability(device_id=0):
        calls.append(device_id)
        return DeviceCapability(9, 0)

    # Vendor wheels need not ship NVIDIA's extension imported by this module.
    # Stub the NVML boundary, retaining the actual PlatformFL capability logic.
    cuda_module = ModuleType("vllm.platforms.cuda")
    cuda_module.NvmlCudaPlatform = SimpleNamespace(get_device_capability=capability)
    monkeypatch.setitem(sys.modules, "vllm.platforms.cuda", cuda_module)
    assert PlatformFL.get_device_capability(2) == DeviceCapability(9, 0)
    assert PlatformFL.supports_fp8()
    assert calls == [2, 0]
