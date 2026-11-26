"""Small deterministic fixtures used by attention startup validation."""

import torch

ATOL = 3e-2
RTOL = 3e-2


def _make_case(qlens, lengths, page, device):
    generator = torch.Generator(device=device).manual_seed(20261104)

    def randn(*shape):
        return torch.randn(
            shape, device=device, dtype=torch.bfloat16, generator=generator
        )

    pages = [(length + page - 1) // page for length in lengths]
    count = sum(pages)
    cache = randn(count, 2, page, 2, 128)
    permutation = torch.randperm(count, device=device, generator=generator).int()
    block_table = torch.zeros(
        (len(qlens), max(pages)), device=device, dtype=torch.int32
    )
    offset = 0
    for row, num_pages in enumerate(pages):
        block_table[row, :num_pages] = permutation[offset : offset + num_pages]
        offset += num_pages
    starts = [0]
    for query_len in qlens:
        starts.append(starts[-1] + query_len)
    key, value = cache.unbind(1)
    query = randn(starts[-1], 16, 128)
    return (
        {
            "q": query,
            "k": key,
            "v": value,
            "out": torch.empty_like(query),
            "cu_seqlens_q": torch.tensor(starts, dtype=torch.int32, device=device),
            "seqused_k": torch.tensor(lengths, dtype=torch.int32, device=device),
            "block_table": block_table,
            "softmax_scale": 128**-0.5,
        },
        max(qlens),
        max(lengths),
    )


def _reference(case):
    """Independent CPU FP32 causal GQA oracle."""
    query, key, value = (
        case[name].detach().to(device="cpu", dtype=torch.float32)
        for name in ("q", "k", "v")
    )
    table = case["block_table"].cpu().long()
    starts = case["cu_seqlens_q"].cpu().tolist()
    lengths = case["seqused_k"].cpu().tolist()
    page = key.shape[1]
    groups = query.shape[1] // key.shape[2]
    output = torch.empty_like(query)
    for row, length in enumerate(lengths):
        start, end = starts[row : row + 2]
        if start == end:
            continue
        blocks = table[row, : (length + page - 1) // page]
        keys = key[blocks].flatten(0, 1)[:length].repeat_interleave(groups, dim=1)
        values = value[blocks].flatten(0, 1)[:length].repeat_interleave(groups, dim=1)
        scores = query[start:end].transpose(0, 1) @ keys.permute(1, 2, 0)
        scores *= case["softmax_scale"]
        query_positions = length - (end - start) + torch.arange(end - start)
        mask = torch.arange(length)[None, :] > query_positions[:, None]
        scores.masked_fill_(mask[None, :, :], float("-inf"))
        output[start:end] = (scores.softmax(-1) @ values.transpose(0, 1)).transpose(
            0, 1
        )
    return output
