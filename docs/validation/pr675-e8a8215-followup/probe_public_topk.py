import os

os.environ["TOPS_VISIBLE_DEVICES"] = "4"
import json
import time
from pathlib import Path

import flag_gems
import torch

root = Path(__file__).resolve().parent
rows = []
for batch, width in [(1, 128), (1, 151936), (4, 151936), (1, 248320), (16, 248320)]:
    torch.manual_seed(123)
    source = torch.randn(batch, width, dtype=torch.float32)
    x = source.to("gcu")
    expected = torch.topk(source, 5)
    print("START", batch, width, flush=True)
    start = time.perf_counter()
    values, indices = flag_gems.topk(x, 5)
    torch.gcu.synchronize()
    first = time.perf_counter() - start
    actual = values.cpu()
    ids = indices.cpu()
    valid = torch.equal(ids, expected.indices) and torch.equal(actual, expected.values)
    start = time.perf_counter()
    for repeat in range(10):
        flag_gems.topk(x, 5)
    torch.gcu.synchronize()
    warm = (time.perf_counter() - start) / 10
    row = dict(
        batch=batch, vocab=width, exact=valid, cold_seconds=first, warm_seconds=warm
    )
    rows.append(row)
    print(json.dumps(row), flush=True)
    (root / "public-topk-probe.json").write_text(json.dumps(rows, indent=2))
    assert valid, row
