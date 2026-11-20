"""Cache routing must use the spec, never the empty profile placeholder."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

path = (Path(__file__).resolve().parents[3] / "vllm_fl/dispatch/backends/vendor/"
        "hygon/patches/models/deepseek_v4.py")
spec = importlib.util.spec_from_file_location("hygon_cache_format_test", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.mark.parametrize("configured,actual,size,flag,expected", [
    (torch.uint8, torch.bfloat16, 0, False, False),
    (torch.uint8, torch.uint8, 4, False, False),
    (torch.bfloat16, torch.float32, 0, False, True),
    (torch.bfloat16, torch.bfloat16, 4, False, True),
    (None, torch.bfloat16, 0, False, False),
    (None, torch.bfloat16, 4, False, True),
    (None, torch.float32, 0, True, True),
])
def test_format(configured, actual, size, flag, expected):
    cache = SimpleNamespace(kv_cache=torch.empty(size, dtype=actual))
    if configured is not None:
        cache.dtype = configured
    indexer = SimpleNamespace(k_cache=cache, use_bf16_cache=flag)
    assert module._uses_bf16_indexer_cache(indexer) is expected


def test_fp8_indexer_preserves_registered_compile_boundary(monkeypatch):
    """Export must use the fake op, never execute profiling workspace management."""
    from vllm.model_executor.layers import sparse_attn_indexer as sparse
    from vllm_fl.dispatch.backends.vendor.hygon.patches.models import deepseek_v4
    deepseek_v4._patch_sparse_attn_indexer()

    def forbidden(*args, **kwargs):
        raise AssertionError("Python indexer body entered during tracing")

    monkeypatch.setattr(sparse, "sparse_attn_indexer", forbidden)

    class Probe(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.k_cache = SimpleNamespace(prefix="test.indexer", dtype=torch.uint8,
                                          kv_cache=torch.empty(0, dtype=torch.bfloat16))
            self.skip_k_cache_insert = True
            self.use_fp4_cache = False
            self.quant_block_size = 128
            self.scale_fmt = "ue8m0"
            self.topk_tokens = 2
            self.head_dim = 128
            self.max_model_len = 16
            self.max_total_seq_len = 16
            self.register_buffer("topk_indices_buffer", torch.empty((1, 2), dtype=torch.int32))

        def forward(self, hidden, q, k, weights):
            return sparse.SparseAttnIndexer.forward_hip(self, hidden, q, k, weights)

    inputs = (torch.empty((1, 8)), torch.empty((1, 32, 128), dtype=torch.float8_e4m3fn),
              torch.empty((1, 128)), torch.empty((1, 32)))
    exported = torch.export.export(Probe(), inputs, strict=True)
    assert any(node.op == "call_function" and "vllm.sparse_attn_indexer" in str(node.target)
               for node in exported.graph.nodes), exported.graph
