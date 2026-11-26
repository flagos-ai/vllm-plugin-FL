"""vLLM wiring for the accepted FlagGems MiniCPM operators. No compute kernels."""
from __future__ import annotations
from flag_gems.fused.minicpm import top_p as _gems
from flag_gems.fused.minicpm.top_p import (
    FAST_SAMPLE_CALLS,
    FUSED_SAMPLE_CALLS,
    INV_SAMPLE_CALLS,
    SORT_FREE_TOP_P_CALLS,
    _FAST_BLOCK,
    _FAST_BUF,
    _FAST_CAP,
    _FAST_DEPTH,
    _FAST_K,
    _FAST_SB,
    _LOG2E,
    _MIN32,
    _SEED_BUF,
    _bit_and,
    _bit_or,
    _fast_buffers,
    _fused_ok,
    _lshr,
    _ordered_key,
    _top_p_mask_kernel,
    _tp_fast_kernel,
    _tp_inv_kernel,
    _warm_up,
    logger,
    logging,
    tl,
    top_p_mask_,
    top_p_sample,
    top_p_sample_fast,
    top_p_sample_inv,
    torch,
    triton
)

def patch_top_p_for_iluvatar() -> None:
    """Top-p-only batches use the sort-free kernel; everything else is vLLM's path. Idempotent."""
    global SORT_FREE_TOP_P_CALLS
    try:
        import vllm.v1.sample.ops.topk_topp_sampler as tts

        if getattr(tts.apply_top_k_top_p, "_iluvatar_stock", None) is not None:
            return
        if not self_check(torch.device("cuda", torch.cuda.current_device())):
            logger.warning("BI-V150 sort-free top-p: self-check failed, vLLM's kernel kept")
            return
        stock = tts.apply_top_k_top_p

        def apply_top_k_top_p(logits, k, p, *args, **kwargs):
            global SORT_FREE_TOP_P_CALLS
            if k is None and p is not None and logits.dim() == 2 and logits.stride(1) == 1:
                if SORT_FREE_TOP_P_CALLS == 0:
                    logger.info("BI-V150 sort-free top-p active (rows=%d)", logits.shape[0])
                SORT_FREE_TOP_P_CALLS += 1
                return top_p_mask_(logits, p)
            return stock(logits, k, p, *args, **kwargs)

        apply_top_k_top_p._iluvatar_stock = stock
        tts.apply_top_k_top_p = apply_top_k_top_p
        logger.info("BI-V150 sort-free top-p: self-check passed, patched apply_top_k_top_p")
    except Exception as e:
        logger.warning("patch_top_p_for_iluvatar: %s", e)


def patch_fused_sampler_for_iluvatar() -> None:
    """Sampler.sample -> one kernel (temperature, top-p cut, Gumbel-max draw) for plain random top-p batches. Idempotent."""
    global FUSED_SAMPLE_CALLS
    try:
        import os

        from vllm.v1.sample.sampler import Sampler

        if getattr(Sampler.sample, "_iluvatar_stock", None) is not None:
            return
        stock = Sampler.sample

        def sample(self, logits, sampling_metadata, logprobs_mode_override=None):
            global FUSED_SAMPLE_CALLS
            if _fused_ok(logits, sampling_metadata):
                dev = logits.device
                seed = _SEED_BUF.get(dev)
                if seed is None:
                    seed = _SEED_BUF[dev] = torch.zeros(1, dtype=torch.int64, device=dev)
                seed.random_(0, 2**31 - 1)               # fresh seed per call, drawn on the device (no sync)
                if FUSED_SAMPLE_CALLS == 0:
                    logger.info("BI-V150 fused sampler active (rows=%d)", logits.shape[0])
                FUSED_SAMPLE_CALLS += 1
                inv_t = 1.0 / sampling_metadata.temperature
                return top_p_sample(logits, sampling_metadata.top_p, inv_t, seed), None
            return stock(self, logits, sampling_metadata, logprobs_mode_override)

        sample._iluvatar_stock = stock
        Sampler.sample = sample
        _warm_up(torch.device("cuda", torch.cuda.current_device()))
        logger.info("BI-V150 fused sampler: patched Sampler.sample")
    except Exception as e:
        logger.warning("patch_fused_sampler_for_iluvatar: %s", e)


def self_check(device) -> bool:
    """Sort-free mask == vLLM's PyTorch sort path on rows with bf16 ties and edge p values."""
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch

    g = torch.Generator(device=device).manual_seed(0)
    x = (torch.randn((8, 130560), device=device, generator=g) * 3).to(torch.bfloat16).float()
    x[1] = 0.0                     # all ties
    x[2, :5] = 40.0                # mass in a few tokens
    p = torch.tensor([0.95, 0.95, 0.95, 0.5, 0.999, 1e-3, 1.0, 0.9], device=device)
    ref = apply_top_k_top_p_pytorch(x.clone(), None, p)
    ours = top_p_mask_(x.clone(), p)
    return bool(torch.equal(torch.isinf(ref), torch.isinf(ours)))

