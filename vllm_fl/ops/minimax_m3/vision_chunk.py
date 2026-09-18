# SPDX-License-Identifier: Apache-2.0
"""Bound memory by batching independent M3 vision attention segments.
No frames, patches, attention edges, projection weights or output rows are removed.
"""

import os

import torch


def vision_forward(model, pixel_values, grid_thw):
    limit = int(os.environ.get("M3_VISION_CHUNK_PATCHES", "8192"))
    if limit <= 0 or pixel_values.shape[0] <= limit:
        return model._m3_forward_unbounded(pixel_values, grid_thw)
    if torch.is_grad_enabled():
        raise RuntimeError("The bounded M3 vision path is inference-only")
    merge = model.spatial_merge_size**2
    limit = max(merge, (limit // merge) * merge)
    segments = model.vision_model._apply_max_frames_limit(grid_thw)
    lengths = [int(t * h * w) for t, h, w in segments]
    assert sum(lengths) == pixel_values.shape[0]
    assert all(n % merge == 0 for n in lengths)
    groups = []
    group = []
    count = 0
    for grid, n in zip(segments, lengths):
        group.append(grid)
        count += n
        if count >= limit:
            groups.append((group, count))
            group = []
            count = 0
    if group:
        if groups:
            grids, previous = groups[-1]
            groups[-1] = (grids + group, previous + count)
        else:
            groups.append((group, count))
    # A single attention segment larger than limit remains intact.
    # Only the token-local projector/merger is chunked within that segment.
    output = None
    input_offset = 0
    output_offset = 0
    for grids, n in groups:
        hidden = model.vision_model(
            pixel_values=pixel_values[input_offset : input_offset + n], grid_thw=grids
        )
        if hidden.dim() == 3:
            hidden = hidden.squeeze(0)
        assert hidden.shape[0] == n
        start = 0
        while start < n:
            end = min(n, start + limit)
            if 0 < n - end < min(256, limit):
                end = n
            projected = model.multi_modal_projector(hidden[start:end])
            part = model.patch_merge_mlp(projected)
            assert part.shape[0] == (end - start) // merge
            if output is None:
                output = part.new_empty(
                    (pixel_values.shape[0] // merge, part.shape[-1])
                )
            output[output_offset : output_offset + part.shape[0]].copy_(part)
            output_offset += part.shape[0]
            del projected, part
            start = end
        input_offset += n
        del hidden
    assert input_offset == pixel_values.shape[0] and output_offset == output.shape[0]
    if not getattr(model, "_m3_chunk_logged", False):
        print(
            f"[M3_VISION_CHUNK] patches={input_offset} segments={len(segments)} groups={len(groups)} max_segment={max(lengths)} projector_chunk={limit} output_rows={output_offset}",
            flush=True,
        )
        model._m3_chunk_logged = True
    return output
