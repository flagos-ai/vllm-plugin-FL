import ast
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import torch

from vllm.v1.attention.backends.registry import AttentionBackendEnum

from vllm_fl.dispatch.backends.vendor.ascend.impl import mm_encoder_attention


def _attention(*, num_heads=2, num_kv_heads=2, head_size=4):
    attention = object.__new__(
        mm_encoder_attention.AscendMMEncoderAttention
    )
    attention.num_heads = num_heads
    attention.num_kv_heads = num_kv_heads
    attention.head_size = head_size
    attention.num_queries_per_kv = num_heads // num_kv_heads
    attention.enable_pad = (
        mm_encoder_attention.MIN_PAD_SIZE
        < head_size
        < mm_encoder_attention.MAX_PAD_SIZE
    )
    attention.scale_value = head_size**-0.5
    return attention


def test_forward_uses_unpadded_fused_attention_without_quadratic_score(
    monkeypatch,
):
    attention = _attention()
    fused_attention = MagicMock()

    def fill_output(**kwargs):
        kwargs["out"].fill_(7)

    fused_attention.side_effect = fill_output
    monkeypatch.setattr(
        mm_encoder_attention.torch_npu,
        "_npu_flash_attention_unpad",
        fused_attention,
    )
    query = torch.zeros((1, 3, 8))
    key = torch.ones((1, 3, 8))
    value = torch.full((1, 3, 8), 2.0)

    output = attention.forward_oot(
        query,
        key,
        value,
        cu_seqlens=torch.tensor([0, 1, 3], dtype=torch.int32),
    )

    assert output.shape == (1, 3, 8)
    assert torch.all(output == 7)
    kwargs = fused_attention.call_args.kwargs
    assert kwargs["query"].shape == (3, 2, 4)
    assert kwargs["key"].shape == (3, 2, 4)
    assert kwargs["value"].shape == (3, 2, 4)
    assert kwargs["out"].shape == kwargs["query"].shape
    assert kwargs["seq_len"].device.type == "cpu"
    torch.testing.assert_close(
        kwargs["seq_len"], torch.tensor([1, 2], dtype=torch.int32)
    )


def test_forward_builds_default_sequence_lengths(monkeypatch):
    attention = _attention()
    fused_attention = MagicMock(
        side_effect=lambda **kwargs: kwargs["out"].zero_()
    )
    monkeypatch.setattr(
        mm_encoder_attention.torch_npu,
        "_npu_flash_attention_unpad",
        fused_attention,
    )

    attention.forward_oot(
        torch.zeros((2, 3, 8)),
        torch.zeros((2, 3, 8)),
        torch.zeros((2, 3, 8)),
    )

    torch.testing.assert_close(
        fused_attention.call_args.kwargs["seq_len"],
        torch.tensor([3, 3], dtype=torch.int32),
    )


def test_head_padding_matches_target_backend(monkeypatch):
    attention = _attention(head_size=80)
    fused_attention = MagicMock(
        side_effect=lambda **kwargs: kwargs["out"].fill_(1)
    )
    monkeypatch.setattr(
        mm_encoder_attention.torch_npu,
        "_npu_flash_attention_unpad",
        fused_attention,
    )

    output = attention.forward_oot(
        torch.zeros((1, 2, 160)),
        torch.zeros((1, 2, 160)),
        torch.zeros((1, 2, 160)),
    )

    assert fused_attention.call_args.kwargs["query"].shape == (2, 2, 128)
    assert output.shape == (1, 2, 160)


def test_maybe_compute_seq_lens_returns_cpu_tensor():
    result = mm_encoder_attention.AscendMMEncoderAttention.maybe_compute_seq_lens(
        AttentionBackendEnum.TORCH_SDPA,
        np.array([0, 2, 7], dtype=np.int32),
        torch.device("cpu"),
    )

    assert result.device.type == "cpu"
    torch.testing.assert_close(result, torch.tensor([2, 5], dtype=torch.int32))


def test_source_does_not_materialize_attention_score():
    source_path = Path(mm_encoder_attention.__file__)
    tree = ast.parse(source_path.read_text())

    forbidden_calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr in {"matmul", "softmax"}:
            forbidden_calls.append(func.attr)

    assert forbidden_calls == []
