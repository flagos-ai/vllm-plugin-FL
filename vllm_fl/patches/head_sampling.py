"""Use the exact FP32 head at the final, untransformed sampling boundary."""
from functools import lru_cache
import logging
import mmap
import os

import torch
from vllm import envs

from flag_gems.fused.minicpm.head_fp32 import project


logger = logging.getLogger("vllm.flagos.head_fp32")
_SHAPE = (130560, 2048)
_PHASE = os.environ.get("E209_PHASE")
_MARKED = set()


@lru_cache(maxsize=1)
def _phase_map():
    with open(_PHASE, "rb") as file:
        return mmap.mmap(file.fileno(), 1, access=mmap.ACCESS_READ)


def sampling_logits(runner, hidden, scheduler_output, spec_metadata):
    phase = _phase_map()[0] if _PHASE else ord("1")
    if phase not in (ord("1"), ord("2"), ord("3"), ord("4")):
        return runner.model.compute_logits(hidden)

    model = runner.get_model()
    metadata = runner.input_batch.sampling_metadata
    parallel = runner.parallel_config
    head = getattr(model, "lm_head", None)
    processor = getattr(model, "logits_processor", None)
    scale = getattr(head, "_e127_head_scale", None)
    if (
        type(model).__name__ != "LlamaForCausalLM"
        or parallel.tensor_parallel_size != 1
        or parallel.pipeline_parallel_size != 1
        or runner.speculative_config is not None
        or spec_metadata is not None
        or envs.VLLM_COMPUTE_NANS_IN_LOGITS
        or scheduler_output.has_structured_output_requests
        or metadata.max_num_logprobs is not None
        or metadata.logprob_token_ids
        or scale is None
        or hidden.dtype != torch.bfloat16
        or hidden.ndim != 2
        or not 1 <= hidden.shape[0] <= 64
        or hidden.shape[1] != _SHAPE[1]
        or tuple(head.weight.shape) != _SHAPE
        or head.weight.dtype != torch.int8
        or type(processor).__name__ != "LogitsProcessor"
        or processor.logits_as_input
        or processor.scale != 1.0
        or processor.soft_cap is not None
        or processor.org_vocab_size != _SHAPE[0]
    ):
        return runner.model.compute_logits(hidden)
    logits = project(hidden, head.weight, scale)
    if phase not in _MARKED and (hidden.shape[0] == 64 or phase == ord("4")):
        logger.info("E209 exact FP32 logits M%d active phase=%s", hidden.shape[0], chr(phase))
        _MARKED.add(phase)
    return logits
