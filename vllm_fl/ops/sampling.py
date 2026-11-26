"""Optional row-parallel dispatch for the upstream top-k/top-p Triton kernel."""

import statistics

import torch

from vllm.logger import init_logger

from vllm_fl import device_profiles, envs

logger = init_logger("vllm.vllm_fl.sampling")
_SWITCH = "VLLM_FL_SAMPLER_ROW_PARALLEL"


def _make_wrapper(upstream, original):
    """Keep upstream masking and caches; increase only the number of programs."""

    def apply(logits, k, p, mask_value=float("-inf")):
        assert logits.ndim == 2 and logits.dtype == torch.float32
        batch, vocab = logits.shape
        if batch == 0 or (k is None and p is None):
            return logits
        sm = upstream.num_compute_units(logits.device.index)
        if batch <= sm:
            return original(logits, k, p, mask_value)
        if logits.stride(1) != 1:
            logits = logits.contiguous()
        if k is not None:
            assert k.ndim == 1 and k.shape[0] == batch
            k_ptr = k.to(torch.int32)
        else:
            k_ptr = logits
        if p is not None:
            assert p.ndim == 1 and p.shape[0] == batch
            p_ptr = p.to(torch.float32)
        else:
            p_ptr = logits
        programs = min(4 * sm, batch)
        key = (logits.device, logits.dtype, vocab)
        buffer = upstream._TRITON_BUFFER_CACHE.get(key)
        if buffer is None or buffer.shape[0] < programs:
            size = min(upstream.next_power_of_2(programs), 4 * sm)
            buffer = logits.new_empty((size, vocab))
            upstream._TRITON_BUFFER_CACHE[key] = buffer
        buffer = buffer[:programs]
        tables = upstream._TRITON_TABLE_CACHE.get(logits.device)
        if tables is None:
            tables = (
                logits.new_tensor(upstream._NORMAL_CDF_TO_SIGMA_TABLE),
                logits.new_tensor(upstream._PERCENTILE_TO_STD_TABLE),
            )
            upstream._TRITON_TABLE_CACHE[logits.device] = tables
        normal, percentile = tables
        upstream._topk_topp_kernel[(programs,)](
            logits,
            logits.stride(0),
            buffer,
            percentile,
            normal,
            k_ptr,
            p_ptr,
            BATCH_SIZE=batch,
            MASK_VALUE=mask_value,
            VOCAB_SIZE=vocab,
            BLOCK_SIZE=8192,
            BLOCK_SIZE_TRUNC=4096,
            TOPK_ENABLED=k is not None,
            TOPP_ENABLED=p is not None,
        )
        return logits

    return apply


def _cpu_reference(logits, k, p):
    """Independent FP32 sort/softmax reference for finite, non-tied inputs."""
    sorted_logits, indices = logits.sort(dim=-1, descending=True)
    ranks = torch.arange(logits.shape[1], device="cpu").unsqueeze(0)
    sorted_logits = sorted_logits.masked_fill(ranks >= k[:, None], float("-inf"))
    probabilities = torch.softmax(sorted_logits, dim=-1)
    remove = probabilities.cumsum(-1) - probabilities >= p[:, None]
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    return torch.empty_like(logits).scatter(1, indices, sorted_logits)


def _validate(device, vocab, sm, original, candidate):
    batch = min(64, 4 * sm)
    if not 257 <= vocab <= 262144 or batch <= sm:
        return False
    generator = torch.Generator(device="cpu").manual_seed(72931)
    # Different rows and well-spaced logits avoid ties at the masking boundary.
    small = torch.randn(
        (batch, 257), generator=generator, dtype=torch.float32, device="cpu"
    )
    k = torch.full((batch,), 37, dtype=torch.int32, device="cpu")
    p = torch.full((batch,), 0.85, dtype=torch.float32, device="cpu")
    expected = _cpu_reference(small, k, p)
    actual = candidate(small.to(device), k.to(device), p.to(device)).cpu()
    if not torch.equal(actual, expected):
        return False
    source = torch.randn(
        (batch, vocab), generator=generator, dtype=torch.float32, device="cpu"
    ).to(device)
    p_gpu = torch.full((batch,), 0.95, dtype=torch.float32, device=device)
    a = source.clone()
    b = source.clone()
    original(a, None, p_gpu)
    candidate(b, None, p_gpu)
    torch.cuda.synchronize(device)
    if not torch.equal(a, b):
        return False
    graphs = []
    for fn, work in ((original, a), (candidate, b)):
        work.copy_(source)
        fn(work, None, p_gpu)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            work.copy_(source)
            fn(work, None, p_gpu)
        graphs.append(graph)
    samples = [[], []]
    for order in ((0, 1), (1, 0)):
        for i in order:
            for _ in range(3):
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                graphs[i].replay()
                end.record()
                end.synchronize()
                samples[i].append(start.elapsed_time(end))
    before, after = (statistics.median(x) for x in samples)
    logger.info(
        "Sampler row parallel startup: batch=%d vocab=%d original=%.3fms candidate=%.3fms",
        batch,
        vocab,
        before,
        after,
    )
    return before > 0 and after <= before * 0.9


def install(device, vocab):
    """Install only after numerical and local performance gates pass."""
    try:
        from vllm_fl.ops.topp_select import install as install_topp_select

        if install_topp_select(device, vocab):
            return True
    except Exception:
        logger.warning(
            "FP64 top-p selection unavailable; using upstream", exc_info=True
        )
    try:
        if device.type != "cuda" or torch.version.hip is not None:
            logger.info(
                "Sampler row parallel unsupported on this device; using upstream"
            )
            return False
        profile = device_profiles.lookup(device.index) or {}
        if not envs.switch(_SWITCH, auto=profile.get("sampler_row_parallel") == 4):
            return False
        from vllm.v1.sample.ops import topk_topp_sampler, topk_topp_triton

        expected_args = [
            "LOGITS",
            "LOGITS_STRIDE_0",
            "BUFFER",
            "PERCENTILE_TO_STD_TABLE",
            "NORMAL_CDF_TO_SIGMA_TABLE",
            "K",
            "P",
            "BATCH_SIZE",
            "VOCAB_SIZE",
            "MASK_VALUE",
            "BLOCK_SIZE",
            "BLOCK_SIZE_TRUNC",
            "TOPK_ENABLED",
            "TOPP_ENABLED",
        ]
        if topk_topp_triton._topk_topp_kernel.arg_names != expected_args:
            raise RuntimeError("unsupported upstream sampler kernel API")
        original = topk_topp_triton.apply_top_k_top_p_triton
        if topk_topp_sampler.apply_top_k_top_p_triton is not original:
            raise RuntimeError("sampler binding does not match upstream")
        candidate = _make_wrapper(topk_topp_triton, original)
        sm = topk_topp_triton.num_compute_units(device.index)
        if not _validate(device, vocab, sm, original, candidate):
            logger.warning("Sampler row parallel startup gate failed; using upstream")
            return False
        topk_topp_sampler.apply_top_k_top_p_triton = candidate
        logger.info("Sampler row parallel enabled")
        return True
    except Exception:
        logger.warning(
            "Sampler row parallel unavailable; using upstream", exc_info=True
        )
        return False
