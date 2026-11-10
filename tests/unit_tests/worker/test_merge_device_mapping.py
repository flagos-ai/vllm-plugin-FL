"""CPU regressions for the PR675 device-mapping merge resolution."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
@pytest.mark.parametrize("assigned", [None, "slice", "full"])
@pytest.mark.parametrize("dp_local", [None, 0, 1])
def test_dp_offset_respects_local_device_slice(world_size, assigned, dp_local):
    path = Path(__file__).resolve().parents[3] / "vllm_fl/worker/worker.py"
    tree = ast.parse(path.read_text())
    method = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "init_device"
    )
    branch = method.body[1]
    assert isinstance(branch, ast.If)
    start = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "parallel_config" for t in n.targets
        )
    )
    end = next(
        i
        for i, n in enumerate(branch.body)
        if isinstance(n, ast.If) and ast.unparse(n.test) == "assigned_ids is not None"
    )
    block = ast.Module(body=branch.body[start:end], type_ignores=[])
    ids = (
        None
        if assigned is None
        else list(range(world_size if assigned == "slice" else world_size * 2))
    )
    config = SimpleNamespace(
        assigned_physical_gpu_ids=ids,
        local_world_size=world_size,
        distributed_executor_backend="mp",
        data_parallel_backend="mp",
        nnodes_within_dp=1,
        data_parallel_rank_local=dp_local,
        data_parallel_index=1,
    )
    worker = SimpleNamespace(parallel_config=config, local_rank=0)
    exec(compile(block, str(path), "exec"), {"self": worker})
    expected = (
        0
        if assigned == "slice"
        else ((1 if dp_local is None else dp_local) * world_size)
    )
    assert worker.local_rank == expected
