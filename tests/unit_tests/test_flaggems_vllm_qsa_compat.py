# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 BAAI. All rights reserved.

import subprocess
import sys
import textwrap

import pytest
import torch


def test_qsa_import_preserves_unpacked_attention_contract():
    code = """
        import inspect
        from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa
        assert callable(qsa.qsa_sparse_split_count)
        # The Plugin owns unpacked indices and a fixed split workspace.
        # Binding must reject an accidental switch to the new packed-index ABI.
        inspect.signature(qsa._qsa_sparse_paged_attention).bind(
            *(object() for _ in range(6)),
            softmax_scale=0.2, out=object(), gate=object(), split_workspace=None,
        )
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.gpu
@pytest.mark.parametrize("with_gate", [False, True])
def test_qsa_unpacked_provider_numerical_and_changed_input_graph(with_gate):
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "thead":
        pytest.skip("PPU dependency-compatibility regression")
    from vllm_fl.models.qwen3_8_flash_next.gpu.ops import qsa

    generator = torch.Generator().manual_seed(572)
    host = [
        torch.randn(shape, generator=generator).to(torch.bfloat16)
        for shape in [(3, 2, 32), (2, 4, 1, 32), (2, 4, 1, 32), (3, 2, 32)]
    ]
    query, keys, values, gate = [x.cuda() for x in host]
    indices_host = torch.tensor([[0, 3, -1, 6]] * 3, dtype=torch.int32)
    indices = indices_host.cuda()
    table_host = torch.tensor([[1, 0]], dtype=torch.int32)
    table = table_host.cuda()
    requests = torch.zeros(3, dtype=torch.int32, device=query.device)
    out = torch.empty_like(query)
    scale = 0.2

    def invoke():
        return qsa.qsa_sparse_paged_attention(
            query,
            keys,
            values,
            indices,
            table,
            requests,
            softmax_scale=scale,
            out=out,
            gate=gate if with_gate else None,
        )

    def check(result):
        expected = torch.empty_like(host[0], dtype=torch.float32)
        for row in range(3):
            selected_keys, selected_values = [], []
            for token in indices_host[row].tolist():
                if token < 0:
                    continue
                block = int(table_host[0, token // 4])
                selected_keys.append(host[1][block, token % 4, 0].float())
                selected_values.append(host[2][block, token % 4, 0].float())
            k, v = torch.stack(selected_keys), torch.stack(selected_values)
            for head in range(2):
                weights = (host[0][row, head].float() @ k.T * scale).softmax(-1)
                expected[row, head] = weights @ v
        if with_gate:
            # The caller contract is a sigmoid gate with BF16 storage rounding.
            expected = (expected.to(torch.bfloat16) * host[3].sigmoid()).float()
        assert result is out
        torch.testing.assert_close(
            result.float().cpu(), expected, atol=0.025, rtol=0.025
        )

    check(invoke())
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            invoke()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = invoke()
    for step in range(2):
        for index, (cpu, device) in enumerate(zip(host, (query, keys, values, gate))):
            cpu.mul_(-0.6).add_((step + index + 1) * 0.1)
            device.copy_(cpu)
        indices_host[:, 1] = step + 1
        indices.copy_(indices_host)
        out.fill_(torch.nan)
        graph.replay()
        torch.cuda.synchronize()
        check(result)
