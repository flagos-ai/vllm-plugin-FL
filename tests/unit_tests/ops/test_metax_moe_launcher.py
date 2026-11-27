# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU launch/stride contracts; real kernel numerics live in the GPU suite."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[3]
    / "vllm_fl/dispatch/backends/vendor/metax/impl/moe_align_block_size.py"
)


class _Kernel:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))

        return launch


def _launcher(name):
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    function = next(node for node in tree.body if getattr(node, "name", None) == name)
    kernels = {
        key: _Kernel()
        for key in (
            "_count_and_initialize",
            "_prefix_partition_counts",
            "_align_expert_counts",
            "_scatter_routes",
            "_zero_invalid_routes",
        )
    }
    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            cdiv=lambda a, b: (a + b - 1) // b,
            next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
        ),
        **kernels,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace[name], kernels


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_empty_routes_have_empty_outputs_and_zero_count_without_kernels(dtype):
    launch, kernels = _launcher("moe_align_block_size")
    sorted_ids, expert_ids, post_pad = launch(torch.empty(0, 8, dtype=dtype), 16, 256)
    assert sorted_ids.shape == expert_ids.shape == (0,)
    assert sorted_ids.dtype == expert_ids.dtype == post_pad.dtype == torch.int32
    torch.testing.assert_close(post_pad, torch.zeros(1, dtype=torch.int32))
    assert not any(kernel.calls for kernel in kernels.values())


@pytest.mark.parametrize(
    "routes,experts,block,pad,capacity",
    [
        (72, 256, 16, False, 1152),
        (2048, 256, 16, False, 5888),
        (72, 7, 16, False, 177),
        (72, 7, 16, True, 192),
    ],
)
def test_capacity_and_non_power_of_two_launch_preserve_the_upstream_abi(
    routes, experts, block, pad, capacity
):
    launch, kernels = _launcher("moe_align_block_size")
    source = torch.arange(routes * 2, dtype=torch.int64)[::2]
    snapshot = source.clone()
    expert_map = torch.arange(experts * 2, dtype=torch.int32)[::2]
    actual = launch(source, block, experts, expert_map, pad, True)
    assert actual[0].shape == (capacity,)
    assert actual[1].shape == ((capacity + block - 1) // block,)
    assert actual[2].shape == (1,)
    torch.testing.assert_close(source, snapshot)
    _, args, metadata = kernels["_count_and_initialize"].calls[0]
    assert args[0].is_contiguous() and args[1].is_contiguous()
    torch.testing.assert_close(args[0], source)
    torch.testing.assert_close(args[1], expert_map)
    assert args[2].shape == (experts + 1, experts)
    assert not args[2].count_nonzero()
    for name in (
        "_count_and_initialize",
        "_prefix_partition_counts",
        "_align_expert_counts",
        "_scatter_routes",
    ):
        assert len(kernels[name].calls) == 1
        _, _, options = kernels[name].calls[0]
        assert options["num_warps"] == 4 and options["num_stages"] == 1
        assert options["NUM_EXPERTS"] == experts
    assert metadata["HAS_MAP"] is True and metadata["IGNORE_NONLOCAL"] is True
    assert kernels["_prefix_partition_counts"].calls[0][2]["BLOCK"] == (
        1 << (experts - 1).bit_length()
    )


def test_cleanup_passes_actual_output_and_id_strides_without_a_contiguous_copy():
    launch, kernels = _launcher("zero_invalid_expert_outputs")
    output = torch.randn(9, 8, 26)[..., ::2]
    ids = torch.arange(9 * 8, dtype=torch.int64).reshape(8, 9).t()
    launch(output, ids, 256)
    grid, args, metadata = kernels["_zero_invalid_routes"].calls[0]
    assert args[0] is output and args[1] is ids
    assert grid == ((output.numel() + 511) // 512,)
    assert tuple(metadata[f"OUT_S{i}"] for i in range(3)) == output.stride()
    assert tuple(metadata[f"IDS_S{i}"] for i in range(2)) == ids.stride()
    assert metadata["WIDTH"] == 13 and metadata["TOP_K"] == 8
    assert metadata["HAS_MAP"] is False
    assert metadata["num_warps"] == 4 and metadata["num_stages"] == 1


@pytest.mark.parametrize("block,experts", [(0, 256), (16, 0)])
def test_invalid_dimensions_fail_before_launch(block, experts):
    launch, kernels = _launcher("moe_align_block_size")
    with pytest.raises(ValueError, match="positive"):
        launch(torch.zeros(9, 8, dtype=torch.int32), block, experts)
    assert not any(kernel.calls for kernel in kernels.values())


def test_non_integer_ids_are_rejected_before_launch():
    launch, kernels = _launcher("moe_align_block_size")
    with pytest.raises(TypeError, match="int32 or int64"):
        launch(torch.zeros(9, 8), 16, 256)
    assert not any(kernel.calls for kernel in kernels.values())
