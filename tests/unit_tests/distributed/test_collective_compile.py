"""Check the real vLLM collective boundary without launching device collectives."""

import os
import subprocess
import sys
from pathlib import Path


def _contract_platform():
    return "vllm.platforms.interface.UnspecifiedPlatform"


def _check_collective_compile():
    import ctypes
    from types import SimpleNamespace

    import torch

    import vllm.distributed.parallel_state as parallel_state
    import vllm.platforms as platforms
    from vllm.platforms.interface import UnspecifiedPlatform

    assert type(platforms.current_platform) is UnspecifiedPlatform
    assert "vllm.platforms.cuda" not in sys.modules
    assert "vllm._C_stable_libtorch" not in sys.modules

    calls = []

    def all_reduce(x):
        # Model the untraceable pointer conversion used by PyNCCL. The real
        # GroupCoordinator custom op must defer this until graph execution.
        ctypes.c_void_p(x.data_ptr())
        calls.append(x.clone())
        return x * 2

    group = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    group.world_size = 2
    group.unique_name = "fl_cpu_compile_regression"
    group.use_custom_op_call = True
    group.device_communicator = SimpleNamespace(all_reduce=all_reduce)
    parallel_state._groups[group.unique_name] = lambda: group
    # Vendor builds normally register this op on their accelerator dispatch
    # key. Use the same implementation on CPU solely for this contract test.
    cpu_lib = torch.library.Library("vllm", "IMPL", "CPU")
    if not torch._C._dispatch_has_kernel_for_dispatch_key("vllm::all_reduce", "CPU"):
        cpu_lib.impl("all_reduce", parallel_state.all_reduce)

    graphs = []

    def backend(graph, _example_inputs):
        graphs.append(graph)
        return graph.forward

    try:
        compiled = torch.compile(group.all_reduce, backend=backend, fullgraph=True)
        x = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        torch.testing.assert_close(compiled(x), x * 2)
        updated = x + 7
        torch.testing.assert_close(compiled(updated), updated * 2)
        assert len(calls) == 2
        torch.testing.assert_close(calls[-1], updated)
        assert len(graphs) == 1
        assert any(
            node.target
            in (torch.ops.vllm.all_reduce, torch.ops.vllm.all_reduce.default)
            for node in graphs[0].graph.nodes
        ), graphs[0].code
    finally:
        torch._dynamo.reset()
        cpu_lib._destroy()
        parallel_state._groups.pop(group.unique_name, None)


def test_all_reduce_custom_op_keeps_ctypes_out_of_dynamo():
    # The Kunlunxin SDK installs its torch.compile mock at Torch import time.
    # Select actual compilation in a fresh interpreter, without changing the
    # parent process's SDK defaults or loading a vendor platform/native ABI.
    env = os.environ.copy()
    env.update(
        CUDA_VISIBLE_DEVICES="",
        VLLM_PLUGINS="cpu_contract",
        XMLIR_ENABLE_MOCK_TORCH_COMPILE="false",
    )
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    from importlib import metadata

    _real_entry_points = metadata.entry_points
    _platform_entry = metadata.EntryPoint(
        name="cpu_contract",
        value="__main__:_contract_platform",
        group="vllm.platform_plugins",
    )

    def _isolated_entry_points(*args, **kwargs):
        if kwargs.get("group") == "vllm.platform_plugins":
            return metadata.EntryPoints((_platform_entry,))
        return _real_entry_points(*args, **kwargs)

    metadata.entry_points = _isolated_entry_points
    _check_collective_compile()
