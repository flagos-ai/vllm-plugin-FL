import torch
import torch.nn as nn
from vllm import envs
from vllm._aiter_ops import rocm_aiter_ops
from vllm.config.model import LogprobsMode
from vllm.logger import init_logger
from vllm.platforms import CpuArchEnum, current_platform
from vllm.triton_utils import HAS_TRITON
if HAS_TRITON:
    from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
logger = init_logger(__name__)

def flashinfer_sampler_supported() -> bool:
    """Decide whether FlashInfer's top-p/top-k sampler can be used.

    Returns False (with appropriate logging) when ``VLLM_USE_FLASHINFER_SAMPLER``
    is 0, when the platform isn't CUDA, when the GPU's compute capability is
    unsupported. Raises ``RuntimeError`` if the user explicitly opted in
    via the env var but FlashInfer is unavailable.

    Assumes flashinfer is installed, as guaranteed by ``requirements/cuda.txt``;
    otherwise importing the FlashInfer backend below raises ``ImportError``.

    Note: callers must additionally ensure ``logprobs_mode`` doesn't require
    post-top-k/top-p logits/logprobs for any request whose logprobs will be
    returned in this step, since FlashInfer doesn't expose those.
    """
    if not current_platform.is_cuda():
        return False
    if not envs.VLLM_USE_FLASHINFER_SAMPLER:
        logger.info_once('FlashInfer top-p/top-k sampling disabled via VLLM_USE_FLASHINFER_SAMPLER=0.')
        return False
    from vllm.v1.attention.backends.flashinfer import FlashInferBackend
    capability = current_platform.get_device_capability()
    assert capability is not None
    unsupported_reason: str | None = None
    if not FlashInferBackend.supports_compute_capability(capability):
        unsupported_reason = f'unsupported compute capability {capability.as_version_str()}'
    if unsupported_reason is None:
        logger.info_once('Using FlashInfer for top-p & top-k sampling.', scope='global')
        return True
    if envs.is_set('VLLM_USE_FLASHINFER_SAMPLER'):
        raise RuntimeError(f'FlashInfer top-p/top-k sampling unavailable: {unsupported_reason}. Unset VLLM_USE_FLASHINFER_SAMPLER=1.')
    logger.warning_once('FlashInfer top-p/top-k sampling unavailable: %s; falling back. Set VLLM_USE_FLASHINFER_SAMPLER=0 to silence.', unsupported_reason)
    return False

class TopKTopPSampler(nn.Module):
    """
    Module that performs optional top-k and top-p filtering followed by
    weighted random sampling of logits.

    Implementations may update the logits tensor in-place.
    """

    def __init__(self, logprobs_mode: LogprobsMode='raw_logprobs', use_fp64_gumbel: bool=False) -> None:
        super().__init__()
        self.logprobs_mode = logprobs_mode
        self.use_fp64_gumbel = use_fp64_gumbel
        if current_platform.is_cuda():
            can_use_flashinfer = logprobs_mode not in ('processed_logits', 'processed_logprobs') and flashinfer_sampler_supported()
            self.forward = self.forward_cuda if can_use_flashinfer else self.forward_native
        elif current_platform.is_cpu():
            arch = current_platform.get_cpu_architecture()
            if arch in (CpuArchEnum.RISCV, CpuArchEnum.POWERPC):
                self.forward = self.forward_native
            else:
                self.forward = self.forward_cpu
        elif current_platform.is_xpu():
            if envs.VLLM_XPU_USE_SAMPLER_KERNEL:
                self.forward = self.forward_xpu
            else:
                self.forward = self.forward_native
        elif logprobs_mode not in ('processed_logits', 'processed_logprobs') and rocm_aiter_ops.is_enabled():
            self.aiter_ops = None
            self._aiter_ops_import_failed = False
            logger.info_once('Using aiter sampler on ROCm (lazy import, sampling-only).')
            self.forward = self.forward_hip
        else:
            self.forward = self.forward_native

    def forward_native(self, logits: torch.Tensor, generators: dict[int, torch.Generator], k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling.

        The logits tensor may be updated in-place.
        """
        if getattr(torch, 'corex', False):
            from vllm_fl.ops.softmax_sample_install_r640 import eligible as _qualified
            if _qualified(self, logits):
                return _tianshu_forward_fused(self, logits, generators, k, p)
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == 'processed_logits':
            logits_to_return = logits
        elif self.logprobs_mode == 'processed_logprobs':
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        return (random_sample(probs, generators, self.use_fp64_gumbel), logits_to_return)

    def forward_cuda(self, logits: torch.Tensor, generators: dict[int, torch.Generator], k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """More optimized implementation for top-k and top-p sampling."""
        if k is None and p is None or generators:
            if generators:
                logger.debug_once('FlashInfer 0.2.3+ does not support per-request generators. Falling back to PyTorch-native implementation.')
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ('processed_logits', 'processed_logprobs'), 'FlashInfer does not support returning logits/logprobs'
        return (flashinfer_sample(logits.contiguous(), k, p, generators), None)

    def forward_cpu(self, logits: torch.Tensor, generators: dict[int, torch.Generator], k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        PyTorch-native implementation of top-k and top-p sampling for CPU.

        The logits tensor may be updated in-place.
        """
        logits = apply_top_k_top_p(logits, k, p)
        logits_to_return = None
        if self.logprobs_mode == 'processed_logits':
            logits_to_return = logits
        elif self.logprobs_mode == 'processed_logprobs':
            logits_to_return = logits.log_softmax(dim=-1, dtype=torch.float32)
        if len(generators) != logits.shape[0] and (not self.use_fp64_gumbel):
            return (compiled_random_sample(logits), logits_to_return)
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = empty_exponential_noise_like(probs, self.use_fp64_gumbel)
        q.exponential_()
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
        return (sample_with_exponential_noise(probs, q), logits_to_return)

    def _init_aiter_ops(self) -> bool:
        if self._aiter_ops_import_failed:
            return False
        try:
            import aiter.ops.sampling
        except ImportError:
            self._aiter_ops_import_failed = True
            self.forward = self.forward_native
            logger.warning_once('aiter.ops.sampling is not available on ROCm. Falling back to PyTorch-native implementation.')
            return False
        self.aiter_ops = torch.ops.aiter
        return True

    def forward_hip(self, logits: torch.Tensor, generators: dict[int, torch.Generator], k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Optimized ROCm/aiter path (same structure as forward_cuda)."""
        if k is None and p is None or generators:
            if generators:
                logger.warning_once('aiter sampler does not support per-request generators; falling back to PyTorch-native.')
            return self.forward_native(logits, generators, k, p)
        if self.use_fp64_gumbel:
            return self.forward_native(logits, generators, k, p)
        assert self.logprobs_mode not in ('processed_logits', 'processed_logprobs'), 'aiter sampler does not support returning logits/logprobs.'
        if self.aiter_ops is None and (not self._init_aiter_ops()):
            return self.forward_native(logits, generators, k, p)
        return (self.aiter_sample(logits, k, p, generators), None)

    def aiter_sample(self, logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None, generators: dict[int, torch.Generator]) -> torch.Tensor:
        """Sample from logits using aiter ops."""
        assert self.aiter_ops is not None
        use_top_k = k is not None
        use_top_p = p is not None
        if use_top_p and use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_k_top_p_sampling_from_probs(probs, None, *_to_tensor_scalar_tuple(k), *_to_tensor_scalar_tuple(p), deterministic=True)
            return next_token_ids.view(-1)
        elif use_top_p:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            next_token_ids = self.aiter_ops.top_p_sampling_from_probs(probs, None, *_to_tensor_scalar_tuple(p), deterministic=True)
            return next_token_ids.view(-1)
        elif use_top_k:
            probs = logits.softmax(dim=-1, dtype=torch.float32).contiguous()
            renorm_probs = self.aiter_ops.top_k_renorm_probs(probs, *_to_tensor_scalar_tuple(k))
            return torch.multinomial(renorm_probs, num_samples=1).view(-1)
        raise RuntimeError('aiter_sample was called with no active top-k or top-p.')

    def forward_xpu(self, logits: torch.Tensor, generators: dict[int, torch.Generator], k: torch.Tensor | None, p: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if generators:
            logger.warning_once('xpu kernel topk_topp_sampler does not support per-request generators. Falling back to PyTorch-native implementation.')
            return self.forward_native(logits, generators, k, p)
        random_sampled = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
        logits_to_return = None
        if self.logprobs_mode == 'processed_logits' or self.logprobs_mode == 'processed_logprobs':
            logits_to_return = torch.empty_like(logits)
        assert len(generators) != logits.shape[0], 'xpu kernel topk_topp_sampler does not support batch-wise generators.'
        generator = torch.xpu.default_generators[logits.device.index]
        state = generator.get_state()
        seed, offset = state.view(torch.int64)
        seeds = torch.tensor([seed, offset], dtype=torch.int64, device=torch.device('cpu'))
        if k is not None:
            k = k.to(torch.int64)
        torch.ops.vllm.xpu_topk_topp_sampler(random_sampled, logits_to_return, logits, k, p, self.logprobs_mode, seeds)
        offset = (offset + logits.numel() + 3) // 4 * 4
        state.view(torch.int64)[1] = offset
        generator.set_state(state)
        return (random_sampled, logits_to_return)

@torch.compile(dynamic=True)
def compiled_random_sample(logits: torch.Tensor) -> torch.Tensor:
    if getattr(torch, 'corex', False) and logits.ndim == 2 and (logits.dtype == torch.float32) and logits.is_cuda and (8 <= logits.shape[0] <= 64) and (logits.shape[1] == 130560):
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        q = torch.empty_like(probs)
        q.exponential_()
        return _tianshu_noise_fused(probs, q, False)
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)

def apply_top_k_top_p(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None) -> torch.Tensor:
    if p is None and k is None:
        return logits
    if current_platform.is_cpu():
        if HAS_TRITON:
            return apply_top_k_top_p_triton(logits, k, p)
        return apply_top_k_top_p_pytorch(logits, k, p, allow_cpu_sync=True)
    if HAS_TRITON and logits.shape[0] >= 8:
        return apply_top_k_top_p_triton(logits, k, p)
    return apply_top_k_top_p_pytorch(logits, k, p)

def apply_top_k_top_p_pytorch(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None, allow_cpu_sync: bool=False) -> torch.Tensor:
    """Apply top-k and top-p masks to the logits.

    If a top-p is used, this function will sort the logits tensor,
    which can be slow for large batches.

    The logits tensor may be updated in-place.
    """
    if p is None:
        if k is None:
            return logits
        if allow_cpu_sync:
            return apply_top_k_only(logits, k)
    logits_sort, logits_idx = logits.sort(dim=-1, descending=False)
    if k is not None:
        top_k_mask = logits_sort.size(1) - k.to(torch.long)
        top_k_mask = logits_sort.gather(1, top_k_mask.unsqueeze(dim=1))
        top_k_mask = logits_sort < top_k_mask
        logits_sort.masked_fill_(top_k_mask, -float('inf'))
    if p is not None:
        probs_sort = logits_sort.softmax(dim=-1)
        probs_sum = torch.cumsum(probs_sort, dim=-1, out=probs_sort)
        top_p_mask = probs_sum <= 1 - p.unsqueeze(dim=1)
        top_p_mask[:, -1] = False
        logits_sort.masked_fill_(top_p_mask, -float('inf'))
    return logits.scatter_(dim=-1, index=logits_idx, src=logits_sort)

def apply_top_k_only(logits: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """
    Apply top-k mask to the logits.

    This implementation doesn't involve sorting the entire vocab.
    Note however that it involves a GPU->CPU sync which can be detrimental for
    async scheduling performance.

    The logits tensor may be updated in-place.
    """
    no_top_k_mask = k == logits.shape[1]
    k = k.masked_fill(no_top_k_mask, 1)
    max_top_k = k.max()
    k_index = k.sub_(1).unsqueeze(1)
    top_k_mask = logits.topk(max_top_k, dim=1).values.gather(1, k_index.long())
    top_k_mask.masked_fill_(no_top_k_mask.unsqueeze(1), -float('inf'))
    return logits.masked_fill_(logits < top_k_mask, -float('inf'))

def empty_exponential_noise_like(probs: torch.Tensor, use_fp64_gumbel: bool) -> torch.Tensor:
    dtype = torch.float64 if use_fp64_gumbel else probs.dtype
    return torch.empty(probs.shape, dtype=dtype, device=probs.device)

def sample_with_exponential_noise(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if getattr(torch, 'corex', False):
        from vllm_fl.ops.noise_argmax_install_r616 import eligible as _qualified
        if _qualified(probs, q):
            return _tianshu_noise_fused(probs, q, True)
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)

def random_sample(probs: torch.Tensor, generators: dict[int, torch.Generator], use_fp64_gumbel: bool=False) -> torch.Tensor:
    """Randomly sample from the probabilities.

    We use this function instead of torch.multinomial because torch.multinomial
    causes CPU-GPU synchronization.
    """
    q = empty_exponential_noise_like(probs, use_fp64_gumbel)
    if len(generators) != probs.shape[0]:
        q.exponential_()
    if generators:
        for i, generator in generators.items():
            q[i].exponential_(generator=generator)
    return sample_with_exponential_noise(probs, q)

def flashinfer_sample(logits: torch.Tensor, k: torch.Tensor | None, p: torch.Tensor | None, generators: dict[int, torch.Generator]={}) -> torch.Tensor:
    """Sample from the logits using FlashInfer.

    Statistically, this function is equivalent to the `random_sample` function.
    However, this function is faster because it avoids sorting the logits tensor
    via rejection sampling.

    NOTE: The outputs of this function do not necessarily match the outputs of
    the `random_sample` function. It only guarantees that the outputs are
    statistically equivalent.
    """
    import flashinfer
    assert not (k is None and p is None)
    if k is None:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_p_sampling_from_probs(probs, p, deterministic=True)
    elif p is None:
        probs = logits.softmax(dim=-1, dtype=torch.float32)
        next_token_ids = flashinfer.sampling.top_k_sampling_from_probs(probs, k, deterministic=True)
    else:
        next_token_ids = flashinfer.sampling.top_k_top_p_sampling_from_logits(logits, k, p, deterministic=True)
    return next_token_ids.view(-1)

def _to_tensor_scalar_tuple(x):
    if isinstance(x, torch.Tensor):
        return (x, 0)
    else:
        return (None, x)
import sys as _tianshu_sys
import os
import hashlib
from pathlib import Path
import triton
from vllm_fl.ops.fused_noise_argmax_r614 import noise_argmax, reduce_noise_argmax
from vllm_fl.ops.softmax_sample_install_r640 import make_noise as _tianshu_make_noise, launch as _tianshu_launch_sample
_tianshu_this_module = _tianshu_sys.modules[__name__]
_tianshu_noise_stats = {'fused_inplace': 0, 'fused_out_of_place': 0}
_tianshu_noise_verified = set()
_tianshu_noise_seen = set()
_tianshu_soft_stats = {'fused_calls': 0, 'retained_calls': 0}
_tianshu_soft_verified = set()
_tianshu_soft_seen = set()
_tianshu_sampler_name = 'topk_topp_sampler'

def _tianshu_original_noise_helper(probs: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    if q.dtype == probs.dtype:
        scores = probs.div_(q)
    else:
        scores = q.reciprocal_()
        scores.mul_(probs)
    return scores.argmax(dim=-1).view(-1)

def _tianshu_original_compiled_sample(logits: torch.Tensor) -> torch.Tensor:
    probs = logits.softmax(dim=-1, dtype=torch.float32)
    q = torch.empty_like(probs)
    q.exponential_()
    return probs.div(q).argmax(dim=-1).view(-1)

def _tianshu_noise_fused(probs, noise, inplace):
    m = probs.shape[0]
    verify_dir = os.environ.get('TIANSHU_NOISE_ARGMAX_VERIFY')
    selected = False
    if verify_dir and m == 64 and (inplace not in _tianshu_noise_verified):
        phase = Path(verify_dir) / 'phase.txt'
        selected = phase.exists() and phase.read_text().strip() == 'wave'
    if selected:
        assert not torch.cuda.is_current_stream_capturing()
        _tianshu_noise_verified.add(inplace)
        before_p = probs.clone()
        before_q = noise.clone()
        before_rng = torch.cuda.get_rng_state().clone()
    answer = torch.empty((m,), device=probs.device, dtype=torch.int64)
    partial_values = torch.empty((m, 4), device=probs.device, dtype=torch.float32)
    partial_indexes = torch.empty((m, 4), device=probs.device, dtype=torch.int64)
    first = noise_argmax[m, 4](probs, noise, probs, answer, partial_values, partial_indexes, 130560, probs.stride(0), noise.stride(0), 4096, 4, inplace, num_warps=4)
    second = reduce_noise_argmax[m,](partial_values, partial_indexes, answer, 4, 4, num_warps=4)
    if selected:
        expected_scores = before_p.clone()
        expected = _tianshu_original_noise_helper(expected_scores, before_q.clone())
        torch.cuda.synchronize()
        effects = expected_scores if inplace else before_p
        checks = {'ids_exact': bool(torch.equal(answer, expected)), 'probability_side_effect_bitexact': bool(torch.equal(probs.view(torch.int32), effects.view(torch.int32))), 'noise_unchanged': bool(torch.equal(noise.view(torch.int32), before_q.view(torch.int32))), 'global_RNG_state_unchanged': bool(torch.equal(before_rng, torch.cuda.get_rng_state()))}
        root = Path(verify_dir)
        root.mkdir(exist_ok=True)
        snapshot = root / f'actual_probability_noise_{int(inplace)}.pt'
        torch.save({'probabilities': before_p.cpu(), 'noise': before_q.cpu(), 'original_ids': expected.cpu(), 'candidate_ids': answer.cpu(), 'inplace': inplace}, snapshot)
        row = {'complete': all(checks.values()), 'diagnostic_only': True, 'M': m, 'N': 130560, 'inplace': inplace, 'checks': checks, 'snapshot': str(snapshot), 'snapshot_SHA256': hashlib.sha256(snapshot.read_bytes()).hexdigest(), 'model_uses_candidate_ids': True, 'RNG_operations_added': 0, 'scope': 'OneactualC64suffixverifiedagainstoriginalhelperonidenticalprobs/noise. Extraoldhelper/copies/onesyncperturbtiming;not a performance score.'}
        (root / 'state.json').write_text(json.dumps(row, indent=2))
        assert all(checks.values()), row
    _tianshu_noise_stats['fused_inplace' if inplace else 'fused_out_of_place'] += 1
    key = (inplace, m)
    directory = os.environ.get('TIANSHU_NOISE_ARGMAX_RECEIPTS')
    if directory and key not in _tianshu_noise_seen:
        _tianshu_noise_seen.add(key)
        root = Path(directory)
        root.mkdir(exist_ok=True)
        row = {'M': m, 'inplace': inplace, 'kernel_hashes': [first.hash, second.hash], 'shared_bytes': [first.metadata.shared, second.metadata.shared], 'partial_bytes': m * 4 * (4 + 8), 'RNG_operations_inside_candidate': 0, 'original_prefix_retained': True}
        (root / f'{m}_{int(inplace)}.json').write_text(json.dumps(row, indent=2))
    return answer.view(-1)

def _tianshu_forward_fused(self, logits, generators, k, p):
    filtered = _tianshu_this_module.apply_top_k_top_p(logits, k, p)
    logits_to_return = None
    if self.logprobs_mode == 'processed_logits':
        logits_to_return = filtered
    elif self.logprobs_mode == 'processed_logprobs':
        logits_to_return = filtered.log_softmax(dim=-1, dtype=torch.float32)
    noise = _tianshu_make_noise(filtered, generators, _tianshu_this_module)
    verify_root = os.environ.get('TIANSHU_SOFTMAX_SAMPLE_VERIFY')
    selected = False
    if verify_root and filtered.shape[0] == 64 and (_tianshu_sampler_name not in _tianshu_soft_verified):
        phase = Path(verify_root) / 'phase.txt'
        selected = phase.exists() and phase.read_text().strip() == 'wave'
    if selected:
        _tianshu_soft_verified.add(_tianshu_sampler_name)
        assert not torch.cuda.is_current_stream_capturing()
        before_logits = filtered.clone()
        before_noise = noise.clone()
        rng_after_draw = torch.cuda.get_rng_state().clone()
        per_generator = {i: g.get_state().clone() for i, g in generators.items()}
    answer, kernels = _tianshu_launch_sample(filtered, noise)
    _tianshu_soft_stats['fused_calls'] += 1
    receipts = os.environ.get('TIANSHU_SOFTMAX_SAMPLE_RECEIPTS')
    key = (_tianshu_sampler_name, filtered.shape[0])
    if receipts and key not in _tianshu_soft_seen:
        _tianshu_soft_seen.add(key)
        root = Path(receipts)
        root.mkdir(exist_ok=True)
        row = {'sampler': _tianshu_sampler_name, 'M': filtered.shape[0], 'kernel_hashes': [v.hash for v in kernels], 'shared_bytes': [v.metadata.shared for v in kernels], 'no_probability_or_score_matrix_store': True, 'extra_workspace_bytes': filtered.shape[0] * (2 * 4 + 4 * (4 + 8)), 'noise_draw_moved_before_deterministic_softmax': True, 'RNG_draw_order_among_draws_unchanged': True}
        (root / f'M{filtered.shape[0]}.json').write_text(json.dumps(row, indent=2))
    if selected:
        reference_prob = filtered.softmax(dim=-1, dtype=torch.float32)
        reference_scores = reference_prob.clone()
        expected = _tianshu_this_module.sample_with_exponential_noise(reference_scores, noise.clone())
        debug_p = torch.empty_like(filtered)
        debug_s = torch.empty_like(filtered)
        debug_ids, _ = _tianshu_launch_sample(filtered, noise, debug_p, debug_s)
        torch.cuda.synchronize()
        checks = {'ids_exact': bool(torch.equal(answer, expected)), 'debug_ids_exact': bool(torch.equal(debug_ids, expected)), 'probability_FP32_bits_exact': bool(torch.equal(debug_p.view(torch.int32), reference_prob.view(torch.int32))), 'score_FP32_bits_exact': bool(torch.equal(debug_s.view(torch.int32), reference_scores.view(torch.int32))), 'filtered_logits_unchanged': bool(torch.equal(filtered.view(torch.int32), before_logits.view(torch.int32))), 'noise_unchanged': bool(torch.equal(noise.view(torch.int32), before_noise.view(torch.int32))), 'global_RNG_unchanged_during_kernel_ops': bool(torch.equal(rng_after_draw, torch.cuda.get_rng_state())), 'request_generators_unchanged_during_kernel_ops': all((torch.equal(g.get_state(), per_generator[i]) for i, g in generators.items()))}
        root = Path(verify_root)
        root.mkdir(exist_ok=True)
        path = root / 'actual_filtered_logits_noise.pt'
        torch.save({'filtered_logits': before_logits.cpu(), 'noise': before_noise.cpu(), 'original_ids': expected.cpu(), 'candidate_ids': answer.cpu()}, path)
        row = {'complete': all(checks.values()), 'diagnostic_only': True, 'shape': list(filtered.shape), 'checks': checks, 'model_returns_candidate_ids': True, 'snapshot': str(path), 'snapshot_SHA256': hashlib.sha256(path.read_bytes()).hexdigest(), 'scope': 'OneactualC64samplerfusioncall comparedonnativeprobability/scorebitsandidenticalnoise. Extraoldsoftmax/helper/debuglaunch/copies+syncperturblatency;not a performance score.'}
        (root / 'state.json').write_text(json.dumps(row, indent=2))
        assert all(checks.values()), row
    return (answer, logits_to_return)

import json  # Existing diagnostic receipt paths retained by source extraction.
