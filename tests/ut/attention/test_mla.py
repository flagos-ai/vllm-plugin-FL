from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.attention.backends.abstract import AttentionType

from vllm_fl.attention.mla import MLAFLBackend, MLAFLImpl


@pytest.fixture
def dummy_qkv():
    # batch=2, seq_len=4, num_heads=2, head_dim=8
    device = "cuda"
    B, N, H, D = 2, 4, 2, 8
    q = torch.randn(B, H, N, D, device=device)
    v = torch.randn(B, H, N, D, device=device)
    kv_cache = torch.randn(B, N, D, device=device)
    return q, v, kv_cache


@pytest.fixture
def dummy_metadata():
    class DummyDecode:
        def __init__(self):
            self.block_table = torch.tensor([[0, 1], [1, 2]], device="cuda")
            self.seq_lens = torch.tensor([2, 2], device="cuda")

    metadata = MagicMock()
    metadata.decode = DummyDecode()
    return metadata


def test_backend_name():
    backend_name = MLAFLBackend.get_name()
    assert backend_name == "MLAFL"


def test_impl_cls():
    impl_cls = MLAFLBackend.get_impl_cls()
    assert impl_cls == MLAFLImpl


def test_flash_attn_varlen_diff_headdims(dummy_qkv):
    q, v, _ = dummy_qkv

    impl = MLAFLImpl.__new__(MLAFLImpl)
    impl._pad_v = False
    impl.kv_cache_dtype = "fp32"

    with patch("vllm_fl.attention.mla.flash_attn_varlen_func") as mock_flash_func:
        mock_flash_func.return_value = q
        out = impl._flash_attn_varlen_diff_headdims(q, q, v)

    assert isinstance(out, torch.Tensor)
    assert out.shape[0] == q.shape[0]
    assert out.shape[1] == q.shape[1]
    assert out.device.type == "cuda"


def test_forward_decode(dummy_qkv, dummy_metadata):
    q, _, kv_cache = dummy_qkv
    metadata = dummy_metadata

    impl = MLAFLImpl.__new__(MLAFLImpl)
    impl.kv_cache_dtype = "fp32"
    impl.kv_lora_rank = kv_cache.shape[-1]
    impl.scale = 1.0
    impl._pad_v = False

    B, H, N, D = q.shape
    head_dim = D

    with patch("vllm_fl.attention.mla.flash_mla") as mock_flash_mla:
        mock_flash_mla.return_value = torch.zeros_like(q, device="cuda")
        out, lse = impl._forward_decode(q, kv_cache, metadata, layer=None)

    assert isinstance(out, torch.Tensor)
    assert isinstance(lse, torch.Tensor)
    assert out.shape[0] == q.shape[0]
    assert lse.shape[0] == q.shape[0]
    assert out.device.type == "cuda"
    assert lse.device.type == "cuda"
