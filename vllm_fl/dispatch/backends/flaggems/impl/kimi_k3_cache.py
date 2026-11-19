"""K3 NoPE cache epilogues implemented with the verified open cache primitive."""

import torch
from flaggems_vllm import concat_and_cache_mla


def _check_nope(positions, cos_sin_cache):
    if positions is not None or cos_sin_cache is not None:
        raise NotImplementedError("The K3 cache adapter supports NoPE only")


def _insert(c, extra, cache, slots):
    if c.dtype != torch.bfloat16 or cache.dtype != torch.bfloat16:
        raise TypeError("The verified cache path is BF16")
    concat_and_cache_mla(
        c,
        extra,
        cache,
        slots,
        "auto",
        torch.ones(1, device=c.device, dtype=torch.float32),
    )


def prefill(
    q, k, extra, c, out, cache, slots, block_size, positions=None, cos_sin_cache=None
):
    _check_nope(positions, cos_sin_cache)
    _insert(c, extra, cache, slots)
    concat(k, extra, out)


def concat(k, extra, out):
    extra = extra.reshape(extra.shape[0], -1)
    out.copy_(torch.cat((k, extra[:, None].expand(-1, k.shape[1], -1)), dim=-1))


def decode(
    q_latent,
    q_extra,
    c,
    extra,
    out,
    cache,
    slots,
    block_size,
    positions=None,
    cos_sin_cache=None,
):
    _check_nope(positions, cos_sin_cache)
    _insert(c, extra, cache, slots)
    out.copy_(torch.cat((q_latent, q_extra), dim=-1))
