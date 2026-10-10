# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_private_vision_attention_eager_and_replay(dtype):
    from vllm_fl.kernels.glm5_next.vision_attention import Glm5VisionAttention

    torch.manual_seed(17)
    query, key, value = [
        torch.randn(1, 33, 4, 64, device="cuda", dtype=dtype) for _ in range(3)
    ]
    cu = torch.tensor([0, 16, 33], dtype=torch.int32, device="cuda")
    layer = Glm5VisionAttention(4, 64, 64**-0.5)

    def ref(boundary):
        chunks = []
        for start, end in ((0, boundary), (boundary, 33)):
            chunks.append(
                torch.nn.functional.scaled_dot_product_attention(
                    query[:, start:end].transpose(1, 2).float(),
                    key[:, start:end].transpose(1, 2).float(),
                    value[:, start:end].transpose(1, 2).float(),
                ).transpose(1, 2)
            )
        return torch.cat(chunks, 1).to(dtype)

    tol = 3e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(
        layer(query, key, value, cu), ref(16), atol=tol, rtol=tol
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = layer(query, key, value, cu)
    cu.copy_(torch.tensor([0, 1, 33], dtype=torch.int32, device="cuda"))
    query.copy_(query * 0.5)
    out.fill_(torch.nan)
    graph.replay()
    torch.testing.assert_close(out, ref(1), atol=tol, rtol=tol)
