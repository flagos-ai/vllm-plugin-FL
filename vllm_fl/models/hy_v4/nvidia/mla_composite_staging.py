# SPDX-License-Identifier: Apache-2.0
"""Default-off fresh-owner concat/pad preparation; no deferred KV state."""

from vllm_fl.utils import use_flaggems_op

import torch


def maybe_prepare_mla_composite(impl, q):
    if not use_flaggems_op("concat_pad_mla_q_and_sink", default=False):
        return None
    if (
        not isinstance(q, tuple)
        or len(q) != 2
        or getattr(impl, "dcp_world_size", None) != 1
        or getattr(impl, "pcp_world_size", None) != 1
        or getattr(impl, "kv_cache_dtype", None) not in ("auto", "bfloat16")
        or getattr(impl, "prefill_padding", None) != 64
        or getattr(impl, "num_heads", None) != 4
    ):
        return None
    q_nope, q_rope = q
    sink = getattr(impl, "sinks", None)
    if not all(isinstance(t, torch.Tensor) for t in (q_nope, q_rope, sink)) or any(
        type(dim) is not int or dim <= 0
        for t in (q_nope, q_rope, sink)
        for dim in t.shape
    ):
        return None
    from flaggems_vllm.ops.concat_pad_mla_q_and_sink import (
        _source_geometry,
        concat_pad_mla_q_and_sink,
    )

    # Only unsupported metadata is caught, before allocation or any launch.
    try:
        geometry = _source_geometry(q_nope, q_rope, sink)
    except (TypeError, ValueError, NotImplementedError):
        return None
    m, h, nope, rope = geometry
    if not 1 <= m <= 1024 or (h, nope, rope) != (4, 512, 64) or q_rope.shape[1] != 4:
        return None
    out_q = torch.empty((m, 64, 576), dtype=q_nope.dtype, device=q_nope.device)
    out_sink = torch.empty((64,), dtype=sink.dtype, device=sink.device)
    return concat_pad_mla_q_and_sink(q_nope, q_rope, sink, out_q, out_sink)
