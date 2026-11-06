"""Exercise the native attention consumer with 0.32 interleaved CPU views."""

import ast
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "vllm_fl/dispatch/backends/vendor/ascend/impl/attention.py"
)


@pytest.fixture(scope="module")
def attention():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    state = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionState"
    )
    impl = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionBackendImpl"
    )
    method = next(
        node
        for node in impl.body
        if isinstance(node, ast.FunctionDef) and node.name == "_get_fia_params"
    )
    namespace = {"Enum": Enum, "torch": torch, "AscendMetadata": object}
    exec(
        compile(ast.Module(body=[state, method], type_ignores=[]), str(SOURCE), "exec"),
        namespace,
    )
    return namespace["AscendAttentionState"], namespace["_get_fia_params"]


@pytest.mark.parametrize("layout", ["LBNHC", "LBHNC", "BLNHC"])
@pytest.mark.parametrize("state_index", range(5))
def test_fia_reads_interleaved_kv_without_changing_cache(
    attention, layout, state_index
):
    states, get_params = attention
    blocks, tokens, heads, dim = 3, 4, 2, 3
    if layout == "LBHNC":
        packed = torch.arange(blocks * heads * tokens * 2 * dim).reshape(
            blocks, heads, tokens, 2 * dim
        )
        packed = packed.transpose(1, 2)
    else:
        layers = 2 if layout == "BLNHC" else 1
        packed = torch.arange(blocks * layers * tokens * heads * 2 * dim).reshape(
            blocks, layers, tokens, heads, 2 * dim
        )[:, 0]
    key, value = packed.split(dim, dim=-1)
    original = packed.clone()
    cache = SimpleNamespace(key_cache=key, value_cache=value)
    metadata = SimpleNamespace(
        attn_state=list(states)[state_index],
        seq_lens=torch.tensor([5, 2]),
        seq_lens_list=[5, 2],
        actual_seq_lengths_q=[1, 2],
        block_tables=torch.tensor([[0, 1], [2, 0]], dtype=torch.int32),
    )
    incoming_key = torch.zeros(2, heads, dim)
    incoming_value = torch.ones_like(incoming_key)
    k, v, block_size, table, lengths = get_params(
        cache, incoming_key, incoming_value, metadata
    )
    if metadata.attn_state == states.PrefillNoCache:
        assert k is incoming_key and v is incoming_value
        assert block_size == 128 and table is None
        assert lengths == [1, 2]
    else:
        assert k.is_contiguous() and v.is_contiguous()
        assert k.shape == v.shape == (blocks, tokens, heads * dim)
        torch.testing.assert_close(k.reshape_as(key), key, rtol=0, atol=0)
        torch.testing.assert_close(v.reshape_as(value), value, rtol=0, atol=0)
        assert block_size == tokens and lengths == [5, 2]
        torch.testing.assert_close(table, metadata.block_tables)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    assert cache.key_cache is key and cache.value_cache is value
