# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent numerical regressions for the TP-local attention/indexer kernels."""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires an accelerator"),
]
ROOT = Path(__file__).resolve().parents[2]
IMPL = ROOT / "vllm_fl/dispatch/backends/vendor/metax/impl/dsv4"


@pytest.fixture(scope="module")
def kernels():
    # Load the actual numerical modules without installing model startup hooks.
    name = "_dsv4_numeric_test"
    package = types.ModuleType(name)
    package.__path__ = [str(IMPL)]
    sys.modules[name] = package
    attention = {
        part: importlib.import_module(f"{name}.attention.{part}")
        for part in ("prefill", "staged", "decode")
    }
    spec = importlib.util.spec_from_file_location(
        name + "_topk", IMPL / "indexer/tree_topk.py"
    )
    topk = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = topk
    spec.loader.exec_module(topk)
    yield attention, topk
    for key in list(sys.modules):
        if key.startswith(name):
            del sys.modules[key]


def assert_topk(logits, lengths, result, k):
    """Compare the score multiset; torch.topk does not specify tie index order."""
    for row, length in enumerate(lengths.tolist()):
        ids = result[row]
        count = min(length, k)
        valid_ids = ids[ids >= 0].long()
        assert len(valid_ids) == count
        assert len(valid_ids.unique()) == count
        assert (valid_ids < length).all()
        expected = logits[row, :length].topk(count).values.sort().values
        actual = logits[row, valid_ids].sort().values
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("width", [513, 1024, 2049, 8192])
@pytest.mark.parametrize(
    "pattern", ["ties", "reduce_ties", "equal", "random", "minus_inf"]
)
def test_tree_topk_ties_and_lengths(kernels, width, pattern):
    _, topk = kernels
    torch.manual_seed(541)
    logits = torch.full((6, width + 7), -1.0, device="cuda")[:, :width]
    if pattern == "random":
        logits.copy_(torch.randn_like(logits))
    elif pattern == "minus_inf":
        logits.fill_(-torch.inf)
    else:
        logits[:, :512] = 0
        if pattern == "reduce_ties":
            logits[:, 1024 : min(width, 1536)] = 0
            logits[:, -1] = 1
        elif pattern == "ties":
            logits[:, 512] = 1
        else:
            logits.fill_(0)
    lengths = torch.tensor(
        [0, 1, 511, 512, 513, width], device="cuda", dtype=torch.int32
    )
    out = torch.full((6, 512), -99, device="cuda", dtype=torch.int32)
    topk.decode_topk_tree(logits, lengths, out)
    assert_topk(logits, lengths, out, 512)


def test_tree_topk_graph_refresh(kernels):
    _, topk = kernels
    x = torch.zeros((4, 4096), device="cuda")
    lengths = torch.tensor([0, 513, 2049, 4096], device="cuda", dtype=torch.int32)
    out = torch.empty((4, 512), device="cuda", dtype=torch.int32)

    def call():
        topk.decode_topk_tree(x, lengths, out)

    call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for offset in (512, 3072):
        x.fill_(-1)
        x[:, :512] = 0
        x[:, offset] = 10
        lengths.copy_(torch.tensor([4096, 0, 511, 2049], device="cuda"))
        out.fill_(-99)
        graph.replay()
        torch.cuda.synchronize()
        assert_topk(x, lengths, out, 512)
