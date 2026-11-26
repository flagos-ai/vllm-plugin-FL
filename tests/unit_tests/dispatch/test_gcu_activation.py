import os

import pytest
import torch

_test_platform = os.environ.get("FL_TEST_PLATFORM")
if _test_platform and _test_platform != "enflame":
    pytest.skip("GCU-specific tests", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl.activation import silu_and_mul_gcu


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("rows", [1, 32, 4096])
def test_public_silu_strided_inputs(dtype, rows):
    torch.manual_seed(875)
    cpu = torch.randn(rows, 8704, dtype=dtype)
    result = silu_and_mul_gcu(None, cpu.to("gcu"))
    reference = (
        torch.nn.functional.silu(cpu[:, :4352].float()) * cpu[:, 4352:].float()
    ).to(dtype)
    torch.testing.assert_close(result.cpu(), reference, rtol=0.01, atol=0.01)


def test_public_silu_compiled_graph_replay():
    torch.manual_seed(875)
    cpu = torch.randn(32, 8704, dtype=torch.bfloat16)
    inputs = cpu.to("gcu")

    def forward(value):
        return silu_and_mul_gcu(None, value)

    compiled = torch.compile(forward, backend="eager", fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            compiled(inputs)
        torch.gcu.synchronize()
        graph = torch.gcu.GCUGraph()
        with torch.gcu.graph(graph):
            result = compiled(inputs)
        cpu = torch.randn_like(cpu)
        inputs.copy_(cpu)
        graph.replay()
        torch.gcu.synchronize()
    reference = (
        torch.nn.functional.silu(cpu[:, :4352].float()) * cpu[:, 4352:].float()
    ).to(cpu.dtype)
    torch.testing.assert_close(result.cpu(), reference, rtol=0.01, atol=0.01)
