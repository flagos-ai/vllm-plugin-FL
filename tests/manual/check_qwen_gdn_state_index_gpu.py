"""Check GDN state writes across signed-int32 offsets on a free 80GB GPU.

Run with PYTHONPATH=. python tests/manual/check_qwen_gdn_state_index_gpu.py.
This checks native writes, strided layouts and changed-index CUDA graphs;
it does not measure model throughput. The largest case needs about 36 GiB.
"""

import gc
import importlib
import json
from types import SimpleNamespace

import torch

from vllm_fl.patches.qwen_gdn_state_index import patch_qwen_gdn_state_indices


def check(shape, stride, storage_offset, positions, index_dtype):
    elements = storage_offset + 1 + sum((n - 1) * s for n, s in zip(shape, stride))
    backing = torch.empty(elements, device="cuda", dtype=torch.float32)
    state = backing.as_strided(shape, stride, storage_offset)
    ids = torch.tensor(positions, device="cuda", dtype=index_dtype)
    values = torch.randn((len(positions), *shape[1:]), device="cuda").transpose(2, 3)
    native = torch.ops.aten.index_put_.default
    native(state, [ids], values, False)
    torch.cuda.synchronize()
    expected = values.cpu()
    torch.testing.assert_close(state[ids.long()].cpu(), expected, rtol=0, atol=0)
    native(state, [ids], torch.zeros_like(values), False)
    function = importlib.import_module("flag_gems.ops.index_put").index_put_
    assert function(state, [ids], values, False) is state
    torch.cuda.synchronize()
    torch.testing.assert_close(state[ids.long()].cpu(), expected, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        function(state, [ids], values, False)
    for phase in [1, 2, 3]:
        changed = [p - phase for p in positions]
        ids.copy_(torch.tensor(changed, device="cuda", dtype=index_dtype))
        values.fill_(phase)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(
            state[ids.long()].cpu(), values.cpu(), rtol=0, atol=0
        )
    return {
        "shape": shape,
        "stride": stride,
        "storage_offset": storage_offset,
        "index_dtype": str(index_dtype),
        "positions": positions,
        "native_and_fixed": "passed",
        "changed_index_value_graph_replays": 3,
    }


def main():
    config = SimpleNamespace(hf_config=SimpleNamespace(model_type="qwen4_exp"))
    assert patch_qwen_gdn_state_indices(config)
    assert not patch_qwen_gdn_state_indices(config)
    records = []
    for dtype in [torch.int32, torch.int64]:
        for shape, stride, offset, positions in [
            ((256, 6, 128, 128), (98304, 16384, 128, 1), 0, [251, 239]),
            ((7466, 6, 128, 128), (1279488, 16384, 128, 1), 1920, [1682, 1655]),
        ]:
            records.append(check(shape, stride, offset, positions, dtype))
            gc.collect()
            torch.cuda.empty_cache()
    print(json.dumps({"status": "passed", "records": records}, indent=2))


if __name__ == "__main__":
    main()
