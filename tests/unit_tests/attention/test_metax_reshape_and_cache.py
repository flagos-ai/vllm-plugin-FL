# Copyright (c) 2026 BAAI. All rights reserved.
"""C550 launch bounds and packed-cache numerics for the MetaX cache writer."""

import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/impl/attention/ops/reshape_and_cache.py"
)


class _KernelRecorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(**kwargs):
            self.calls.append((grid(kwargs), kwargs))

        return launch


def _launcher():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(
        node
        for node in tree.body
        if getattr(node, "name", None) == "triton_reshape_and_cache_flash"
    )
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    module = ast.fix_missing_locations(ast.Module(body=[future, node], type_ignores=[]))
    kernel = _KernelRecorder()
    namespace = dict(
        torch=torch,
        triton=SimpleNamespace(
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
            cdiv=lambda x, y: (x + y - 1) // y,
        ),
        reshape_and_cache_kernel_flash=kernel,
        current_platform=SimpleNamespace(fp8_dtype=lambda: torch.float8_e4m3fn),
        _is_supported_kv_cache_dtype=lambda dtype: (
            dtype in ("auto", "bfloat16", "float16", "fp8")
        ),
        is_quantized_kv_cache=lambda dtype: dtype.startswith("fp8"),
    )
    exec(compile(module, str(SOURCE), "exec"), namespace)
    return namespace["triton_reshape_and_cache_flash"], kernel


@pytest.mark.parametrize(("heads", "head_size"), [(3, 80), (5, 192)])
def test_launcher_preserves_packed_strides_and_caps_c550_resources(heads, head_size):
    launch, kernel = _launcher()
    inputs = torch.randn(8, 2, heads, head_size)
    key, value = inputs[:, 0], inputs[:, 1]
    packed = torch.zeros(3, heads, 16, 2 * head_size)
    kc, vc = packed.transpose(1, 2).split(head_size, dim=-1)
    slots = torch.tensor([0, 15, 16, -1, 34, 1])
    scale = torch.tensor(1.0)
    launch(key, value, kc, vc, slots, "auto", scale, scale)
    grid, kwargs = kernel.calls[0]
    tile = min(512, 1 << (heads * head_size - 1).bit_length())
    assert grid == (6, (heads * head_size + tile - 1) // tile)
    assert kwargs["num_warps"] == 4 and kwargs["num_stages"] == 1
    assert kwargs["TILE_SIZE"] == tile
    assert kwargs["key_ptr"] is key and kwargs["value_ptr"] is value
    assert kwargs["key_cache_ptr"] is kc and kwargs["value_cache_ptr"] is vc
    assert kwargs["slot_mapping_ptr"] is slots
    assert kwargs["key_stride"] == key.stride(0)
    assert kwargs["value_stride"] == value.stride(0)
    assert kwargs["block_stride"] == kc.stride(0)
    assert kwargs["head_stride"] == kc.stride(2)
    assert kwargs["page_stride"] == kc.stride(1)
    assert kwargs["USE_HEAD_MAJOR_LAYOUT"] is False
    assert kwargs["FP8_KV_CACHE"] is False
    assert kwargs["k_scale"] is scale and kwargs["v_scale"] is scale


def test_launcher_preserves_head_major_layout_geometry():
    launch, kernel = _launcher()
    key, value = torch.randn(4, 3, 128), torch.randn(4, 3, 128)
    kc = torch.empty(2, 3, 16, 16, 8)
    vc = torch.empty(2, 3, 128, 16)
    launch(
        key,
        value,
        kc,
        vc,
        torch.tensor([0, 1, -1]),
        "auto",
        torch.tensor(1.0),
        torch.tensor(1.0),
    )
    _, kwargs = kernel.calls[0]
    assert kwargs["USE_HEAD_MAJOR_LAYOUT"] is True and kwargs["x"] == 8
    assert kwargs["block_size"] == 16
    assert kwargs["head_stride"] == kc.stride(1)
    assert kwargs["dim_stride_k"] == kc.stride(2)
    assert kwargs["dim_stride_v"] == vc.stride(2)


def test_launcher_retains_fp8_cache_reinterpretation():
    launch, kernel = _launcher()
    key, value = torch.randn(2, 1, 64), torch.randn(2, 1, 64)
    kc, vc = (
        torch.empty(1, 16, 1, 64, dtype=torch.uint8),
        torch.empty(1, 16, 1, 64, dtype=torch.uint8),
    )
    launch(
        key,
        value,
        kc,
        vc,
        torch.tensor([0]),
        "fp8",
        torch.tensor(1.0),
        torch.tensor(1.0),
    )
    _, kwargs = kernel.calls[0]
    assert kwargs["FP8_KV_CACHE"] is True
    assert kwargs["key_cache_ptr"].dtype == torch.float8_e4m3fn
    assert kwargs["key_cache_ptr"].data_ptr() == kc.data_ptr()


def test_empty_slots_skip_kernel_launch():
    launch, kernel = _launcher()
    launch(None, None, None, None, torch.empty(0, dtype=torch.long), "auto", None, None)
    assert not kernel.calls


@pytest.mark.gpu
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("heads", "head_size"), [(3, 80), (5, 192)])
@pytest.mark.parametrize("capture", [False, True])
def test_gpu_packed_cache_skips_negative_slots_input_padding_and_tile_tails(
    dtype, heads, head_size, capture
):
    from tests.utils.device_utils import get_backend

    if get_backend() != "metax":
        pytest.skip("Requires the MetaX CUDA-like runtime")
    if not torch.cuda.is_available():
        pytest.skip("Requires a CUDA-like accelerator")
    spec = importlib.util.spec_from_file_location("metax_cache_gpu_under_test", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    device = torch.device("cuda")
    inputs = torch.randn(8, 2, heads, head_size, dtype=dtype, device=device)
    key, value = inputs[:, 0], inputs[:, 1]
    packed = torch.full((3, heads, 16, 2 * head_size), -17, dtype=dtype, device=device)
    kc, vc = packed.transpose(1, 2).split(head_size, dim=-1)
    slots_cpu = [0, 15, 16, -1, 34, 1]
    slots = torch.tensor(slots_cpu, dtype=torch.long, device=device)
    scale = torch.tensor(1.0, device=device)

    def write():
        module.triton_reshape_and_cache_flash(
            key, value, kc, vc, slots, "auto", scale, scale
        )

    write()  # Compile before graph capture.
    if capture:
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            write()
        packed.fill_(-17)
        graph.replay()
    torch.cuda.synchronize()
    expected = torch.full_like(packed, -17)
    for token, slot in enumerate(slots_cpu):
        if slot >= 0:
            block, offset = divmod(slot, 16)
            expected[block, :, offset, :head_size] = key[token]
            expected[block, :, offset, head_size:] = value[token]
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
