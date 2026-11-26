"""Select and validate INT8 decode configurations before graph capture."""

from __future__ import annotations

import torch

from vllm_fl import device_profiles
from vllm_fl.attention import hybrid_selfcheck, kv_int8
from vllm_fl.attention.decode_config import DecodeConfig

_validated_configs: dict[int, DecodeConfig] = {}


def _profile_config(index: int) -> DecodeConfig | None:
    profile = device_profiles.lookup(index) or {}
    config = profile.get("int8_decode")
    return DecodeConfig(**config) if config is not None else None


def _high_dynamic_query_case(device):
    q = torch.zeros((1, 16, 128), dtype=torch.bfloat16, device=device)
    q[:, :, 0] = 2**17
    q[:, :, 1] = 2**16
    k = torch.zeros((2, 2, 128), dtype=torch.bfloat16, device=device)
    k[0, :, 0] = 1
    k[0, :, 1] = -1
    v = torch.empty_like(k)
    v[0].fill_(1)
    v[1].fill_(-1)
    raw = torch.empty(kv_int8.PAGE_BYTES, dtype=torch.int8, device=device)
    cache = torch.as_strided(
        raw,
        (1, 2, 16, 2, 128),
        (kv_int8.PAGE_BYTES, 4096, 256, 128, 1),
    )
    kv_int8.kv_int8_write(
        k, v, cache, torch.arange(2, dtype=torch.int64, device=device)
    )
    case = dict(
        q=q,
        k=k,
        v=v,
        out=torch.empty_like(q),
        cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=device),
        seqused_k=torch.tensor([2], dtype=torch.int32, device=device),
        block_table=torch.zeros((1, 1), dtype=torch.int32, device=device),
        softmax_scale=128**-0.5,
    )
    return case, cache


def _reference_from_cache(case, cache):
    scales = kv_int8.scale_view(cache).cpu().float()
    values = cache.cpu().float()
    refcase = dict(case)
    refcase["k"] = values[:, 0] * scales[:, 0].permute(0, 2, 1)[..., None]
    refcase["v"] = values[:, 1] * scales[:, 1].permute(0, 2, 1)[..., None]
    return hybrid_selfcheck._reference(refcase)


def _validate(device: torch.device, config: DecodeConfig) -> None:
    case, _, _ = hybrid_selfcheck._make_case([1] * 5, [1, 15, 16, 17, 65], 16, device)
    pages = case["k"].shape[0]
    raw = torch.empty(pages * kv_int8.PAGE_BYTES, dtype=torch.int8, device=device)
    cache = torch.as_strided(
        raw, (pages, 2, 16, 2, 128), (kv_int8.PAGE_BYTES, 4096, 256, 128, 1)
    )
    kv_int8.kv_int8_write(
        case["k"].flatten(0, 1).contiguous(),
        case["v"].flatten(0, 1).contiguous(),
        cache,
        torch.arange(pages * 16, device=device, dtype=torch.int64),
    )
    ref = _reference_from_cache(case, cache).to(device)
    if not torch.isfinite(ref).all():
        raise RuntimeError("non-finite INT8 preset reference")
    workspace = kv_int8.DecodeWorkspace(device)
    for splits in sorted({s for _, s in config.table}):
        case["out"].fill_(float("nan"))
        kv_int8.kv_int8_decode(
            case["q"],
            cache,
            kv_int8.half_base(cache),
            case["out"],
            case["cu_seqlens_q"],
            case["seqused_k"],
            case["block_table"],
            case["softmax_scale"],
            workspace,
            config=config,
            segments=splits,
        )
        got = case["out"].float()
        if (
            not torch.isfinite(got).all()
            or (
                (got - ref).abs()
                > hybrid_selfcheck.ATOL + hybrid_selfcheck.RTOL * ref.abs()
            ).any()
        ):
            raise RuntimeError(f"INT8 preset numerical mismatch at {splits} splits")

    high_case, high_cache = _high_dynamic_query_case(device)
    high_ref = _reference_from_cache(high_case, high_cache).to(device)
    if not torch.allclose(high_ref, torch.ones_like(high_ref), atol=1e-3, rtol=1e-3):
        raise RuntimeError("high-dynamic-range INT8 reference did not select token 0")
    high_case["out"].fill_(float("nan"))
    kv_int8.kv_int8_decode(
        high_case["q"],
        high_cache,
        kv_int8.half_base(high_cache),
        high_case["out"],
        high_case["cu_seqlens_q"],
        high_case["seqused_k"],
        high_case["block_table"],
        high_case["softmax_scale"],
        workspace,
        config=config,
        segments=1,
    )
    got = high_case["out"].float()
    if (
        not torch.isfinite(got).all()
        or (
            (got - high_ref).abs()
            > hybrid_selfcheck.ATOL + hybrid_selfcheck.RTOL * high_ref.abs()
        ).any()
    ):
        raise RuntimeError("INT8 preset high-dynamic-range query mismatch at 1 split")


def _precision_fallback() -> DecodeConfig:
    return DecodeConfig(
        variant=3,
        tile=32,
        num_warps=2,
        num_stages=1,
        table=(
            (1, 32),
            (2, 32),
            (4, 16),
            (8, 8),
            (16, 4),
            (32, 2),
            (64, 16),
            (128, 8),
        ),
    )


def resolve_decode(device) -> DecodeConfig:
    device = torch.device(device)
    index = device.index if device.index is not None else torch.cuda.current_device()
    if index in _validated_configs:
        return _validated_configs[index]

    candidate = _profile_config(index)
    if candidate is None:
        raise RuntimeError("INT8 decode requires a validated capability profile")
    device = torch.device("cuda", index)
    try:
        with torch.cuda.device(device), torch.inference_mode():
            _validate(device, candidate)
    except Exception as error:
        fallback = _precision_fallback()
        kv_int8.logger.warning(
            "INT8 decode profile variant %d failed validation (%s); "
            "trying the conservative FP32-accumulator fallback",
            candidate.variant,
            error,
        )
        with torch.cuda.device(device), torch.inference_mode():
            _validate(device, fallback)
        candidate = fallback

    _validated_configs[index] = candidate
    kv_int8.logger.info(
        "INT8 decode profile verified: variant=%d tile=%d warps=%d stages=%d",
        candidate.variant,
        candidate.tile,
        candidate.num_warps,
        candidate.num_stages,
    )
    return candidate
