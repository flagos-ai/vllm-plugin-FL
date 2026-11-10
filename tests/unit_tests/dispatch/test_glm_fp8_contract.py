"""CPU fake-op contracts for FP8 KV shape and graph padding ownership."""
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def test_fp8_custom_op_fake_shapes_and_cache_write():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_fp8_cache import (
        sparse_fp8_mla, write_fp8_cache,
    )
    with FakeTensorMode():
        q = torch.empty(7, 4, 576, dtype=torch.bfloat16)
        cache = torch.empty(11, 16, 656, dtype=torch.uint8)
        indices = torch.empty(7, 2048, dtype=torch.int32)
        output = sparse_fp8_mla(q, cache, indices, .0625)
        assert output.shape == (7, 4, 512)
        assert output.dtype == torch.bfloat16
        assert write_fp8_cache(torch.empty(7, 512, dtype=torch.bfloat16),
                               torch.empty(7, 64, dtype=torch.bfloat16),
                               cache, torch.empty(7, dtype=torch.int64)) is None


def test_fp8_backend_does_not_claim_padded_forward_cache_updates():
    from vllm_fl.dispatch.backends.vendor.hygon.impl.attention.glm_sparse import HygonGLMSparseBackend
    assert HygonGLMSparseBackend.forward_includes_kv_cache_update is False
