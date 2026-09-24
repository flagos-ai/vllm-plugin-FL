"""Correctness and CUDA-graph checks for native PLE state row I/O."""

from __future__ import annotations

import pytest
import torch

try:
    from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

    from vllm_fl.models.qwen3_8_flash_next.gpu.ops.ple_state import (
        ple_state_gather,
        ple_state_scatter_,
    )
except (ImportError, ModuleNotFoundError) as exc:  # pragma: no cover - env-specific
    pytest.skip(
        f"vLLM PLE state helpers are unavailable: {exc}", allow_module_level=True
    )


def _safe_indices(indices: torch.Tensor) -> torch.Tensor:
    return torch.where(
        indices == NULL_BLOCK_ID,
        torch.zeros_like(indices),
        indices,
    )


def _index_values(num_indices: int, cache_rows: int) -> list[int]:
    values = [1, 2, 1, NULL_BLOCK_ID, NULL_BLOCK_ID, 3, 4, 5, 6, 7]
    result = []
    for i in range(num_indices):
        value = int(values[i % len(values)])
        result.append(value if value == NULL_BLOCK_ID else value % cache_rows)
    return result


def _last_write_mask_host(values: list[int]) -> list[bool]:
    safe = [0 if value == NULL_BLOCK_ID else value for value in values]
    seen: set[int] = set()
    keep = [False] * len(safe)
    for row in range(len(safe) - 1, -1, -1):
        if safe[row] not in seen:
            keep[row] = True
            seen.add(safe[row])
    return keep


def _last_enabled_write_mask_host(
    values: list[int], enabled: list[bool]
) -> list[bool]:
    """Keep only the final enabled writer for every remapped cache row."""

    safe = [0 if value == NULL_BLOCK_ID else value for value in values]
    seen: set[int] = set()
    keep = [False] * len(safe)
    for row in range(len(safe) - 1, -1, -1):
        if enabled[row] and safe[row] not in seen:
            keep[row] = True
            seen.add(safe[row])
    return keep


def _strided_indices(
    values: list[int], *, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    storage = torch.empty(2 * len(values) - 1, dtype=torch.int64, device=device)
    storage[::2] = torch.tensor(values, dtype=torch.int64, device=device)
    if len(values) == 1:
        return storage
    return storage[::2]


def _scatter_reference(
    state: torch.Tensor, indices: torch.Tensor, rows: torch.Tensor
) -> torch.Tensor:
    """Reference with explicit last-occurrence-wins semantics."""

    expected = state.clone()
    safe = _safe_indices(indices).tolist()
    for row, index in enumerate(safe):
        expected[index].copy_(rows[row])
    return expected


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("width", [1, 2, 3, 4, 5, 63, 64, 65])
@pytest.mark.parametrize("num_indices", [1, 2, 17])
def test_ple_state_cpu_strides_dtypes_boundaries_and_duplicates(
    dtype, width, num_indices
):
    cache_rows, hidden = max(23, num_indices + 4), 3
    storage = torch.arange(cache_rows * width * hidden, dtype=torch.float32).reshape(
        cache_rows, width, hidden
    )
    state = storage.transpose(1, 2).to(dtype)
    values = _index_values(num_indices, cache_rows)
    indices = _strided_indices(values)
    safe_indices = _safe_indices(indices)

    expected_rows = torch.ops.aten.index_select.default(state, 0, safe_indices)
    output_storage = torch.empty((num_indices, width, hidden), dtype=dtype)
    output = output_storage.transpose(1, 2)
    actual_rows = ple_state_gather(state, indices, output=output)
    torch.testing.assert_close(actual_rows, expected_rows)
    assert actual_rows.stride() == output.stride()
    default_rows = ple_state_gather(state, indices)
    torch.testing.assert_close(default_rows, expected_rows)
    assert default_rows.is_contiguous()

    updated_rows = actual_rows + torch.tensor(0.5, dtype=dtype)
    expected_state = _scatter_reference(state, indices, updated_rows)
    ple_state_scatter_(state, indices, updated_rows)
    torch.testing.assert_close(state, expected_state)

    # Stage 4: repeated eager calls must be bitwise deterministic.
    for _ in range(10):
        repeated = ple_state_gather(state, indices)
        assert torch.equal(repeated, ple_state_gather(state, indices))


def test_ple_state_cpu_fallback_bypasses_python_index_overrides():
    state = torch.randn(4, 3, 5)
    indices = _strided_indices([2, NULL_BLOCK_ID, 2])
    expected_rows = torch.ops.aten.index_select.default(
        state, 0, _safe_indices(indices)
    )

    def fail(*_args, **_kwargs):
        raise AssertionError("Python index_select/index_copy_ must not be used")

    # FlagGems replaces these Python entry points.  The fallback must remain a
    # small ATen row operation even when those replacements are present.
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(torch, "index_select", fail)
    monkeypatch.setattr(torch.Tensor, "index_copy_", fail)
    try:
        actual = ple_state_gather(state, indices)
        torch.testing.assert_close(actual, expected_rows)
        expected_state = _scatter_reference(state, indices, actual)
        ple_state_scatter_(state, indices, actual)
        torch.testing.assert_close(state, expected_state)
    finally:
        monkeypatch.undo()


def test_ple_state_out_of_range_rows_are_zero_and_writes_are_skipped():
    state = torch.arange(4 * 3 * 5, dtype=torch.float32).reshape(4, 3, 5)
    original = state.clone()
    indices = torch.tensor([-7, 2, 99], dtype=torch.int64)

    gathered = ple_state_gather(state, indices)
    torch.testing.assert_close(gathered[0], torch.zeros_like(gathered[0]))
    torch.testing.assert_close(gathered[1], state[2])
    torch.testing.assert_close(gathered[2], torch.zeros_like(gathered[2]))

    rows = torch.full_like(gathered, 123)
    ple_state_scatter_(state, indices, rows, indices_are_safe=True)
    expected = original.clone()
    expected[2].fill_(123)
    torch.testing.assert_close(state, expected)


@pytest.mark.gpu
def test_ple_state_cuda_boundary_matrix_and_determinism():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda")
    for dtype in (torch.float16, torch.bfloat16, torch.float32):
        for width in (1, 2, 3, 4, 5, 63, 64, 65):
            for num_indices in (1, 2, 17):
                cache_rows, hidden = max(23, num_indices + 4), 3
                storage = torch.randn(
                    cache_rows, width, hidden, dtype=dtype, device=device
                )
                state = storage.transpose(1, 2)
                values = _index_values(num_indices, cache_rows)
                indices = _strided_indices(values, device=device)
                safe_indices = _safe_indices(indices)

                output_storage = torch.empty(
                    (num_indices, width, hidden), dtype=dtype, device=device
                )
                output = output_storage.transpose(1, 2)
                expected_rows = torch.ops.aten.index_select.default(
                    state, 0, safe_indices
                )
                actual_rows = ple_state_gather(state, indices, output=output)
                torch.testing.assert_close(actual_rows, expected_rows, rtol=0, atol=0)

                updated_rows = actual_rows + torch.tensor(
                    1.0, dtype=dtype, device=device
                )
                expected_state = state.clone()
                safe_values = [
                    0 if value == NULL_BLOCK_ID else value for value in values
                ]
                for row, keep in enumerate(_last_write_mask_host(values)):
                    if keep:
                        index = torch.tensor(
                            [safe_values[row]], dtype=torch.int64, device=device
                        )
                        torch.ops.aten.index_copy_.default(
                            expected_state, 0, index, updated_rows[row : row + 1]
                        )
                ple_state_scatter_(state, indices, updated_rows)
                torch.testing.assert_close(state, expected_state, rtol=0, atol=0)

                # Determinism, including duplicate physical rows and null
                # remapping, is checked repeatedly after the write-back.
                reference = ple_state_gather(state, indices)
                for _ in range(10):
                    torch.testing.assert_close(
                        ple_state_gather(state, indices), reference, rtol=0, atol=0
                    )


@pytest.mark.gpu
def test_ple_state_cuda_graph_capture_replay_with_preallocated_buffers():
    """Warm up eagerly, then replay the fixed-shape row I/O graph."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache_rows, hidden, width, num_indices = 23, 3, 65, 17
    storage = torch.randn(cache_rows, width, hidden, dtype=dtype, device=device)
    state = storage.transpose(1, 2)
    values = _index_values(num_indices, cache_rows)
    raw_indices = _strided_indices(values, device=device)
    safe_indices = _safe_indices(raw_indices)
    write_mask = torch.tensor(
        _last_write_mask_host(values), dtype=torch.bool, device=device
    )
    # The graph receives already-remapped indices and a precomputed mask, while
    # inputs/outputs remain fixed-address tensors across replay.
    output_storage = torch.empty(
        (num_indices, width, hidden), dtype=dtype, device=device
    )
    output = output_storage.transpose(1, 2)

    for _ in range(25):
        ple_state_gather(state, safe_indices, output=output, indices_are_safe=True)
        ple_state_scatter_(
            state,
            safe_indices,
            output,
            write_mask=write_mask,
            indices_are_safe=True,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        ple_state_gather(state, safe_indices, output=output, indices_are_safe=True)
        ple_state_scatter_(
            state,
            safe_indices,
            output,
            write_mask=write_mask,
            indices_are_safe=True,
        )
    expected = output.clone()
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.gpu
@pytest.mark.parametrize("num_indices", [1, 64])
def test_ple_state_scatter_graph_replay_updates_inputs(num_indices):
    """Replay changing rows/masks into a transposed cache with storage offset."""

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    device = torch.device("cuda")
    dtype = torch.bfloat16
    cache_rows, hidden, width = 71, 3, 5
    backing = torch.randn(
        cache_rows + 4, width, hidden, dtype=dtype, device=device
    )
    state = backing[2 : 2 + cache_rows].transpose(1, 2)
    assert state.storage_offset() > 0
    baseline = state.clone()

    indices = torch.ones(num_indices, dtype=torch.int64, device=device)
    write_mask = torch.ones(num_indices, dtype=torch.bool, device=device)
    rows_storage = torch.empty(
        num_indices, width, hidden, dtype=dtype, device=device
    )
    rows = rows_storage.transpose(1, 2)

    def set_case(offset: int) -> tuple[list[int], list[bool]]:
        values = [((row + offset) % (cache_rows - 1)) + 1 for row in range(num_indices)]
        masks = [True] * num_indices
        if num_indices:
            # Model NULL padding is already remapped to row zero and masked.
            values[0] = 0
            masks[0] = False
        if num_indices >= 3:
            # Metadata masks the earlier duplicate so parallel stores remain
            # race-free and the last valid row wins.
            values[1] = 2
            values[2] = 2
            masks[1] = False
            if num_indices >= 4:
                masks[3] = False
            # The generated sequence can wrap and naturally repeat the
            # hand-crafted destination. Explicit write masks are required to
            # retain only the final enabled writer for every physical row.
            masks = _last_enabled_write_mask_host(values, masks)
            indices.copy_(torch.tensor(values, dtype=torch.int64, device=device))
        write_mask.copy_(torch.tensor(masks, dtype=torch.bool, device=device))
        row_values = torch.arange(
            num_indices * width * hidden, dtype=torch.float32, device=device
        ).reshape(num_indices, width, hidden)
        rows_storage.copy_((row_values + offset * 1000).to(dtype))
        return values, masks

    set_case(1)
    for _ in range(25):
        state.copy_(baseline)
        ple_state_scatter_(
            state,
            indices,
            rows,
            write_mask=write_mask,
            indices_are_safe=True,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    state.copy_(baseline)
    with torch.cuda.graph(graph, capture_error_mode="thread_local"):
        ple_state_scatter_(
            state,
            indices,
            rows,
            write_mask=write_mask,
            indices_are_safe=True,
        )

    for offset in (7, 19):
        values, masks = set_case(offset)
        state.copy_(baseline)
        expected = baseline.clone()
        for row, (index, enabled) in enumerate(zip(values, masks, strict=True)):
            if enabled:
                expected[index].copy_(rows[row])
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(state, expected, rtol=0, atol=0)
        # The masked NULL row must never perturb the reserved row zero.
        torch.testing.assert_close(state[0], baseline[0], rtol=0, atol=0)
