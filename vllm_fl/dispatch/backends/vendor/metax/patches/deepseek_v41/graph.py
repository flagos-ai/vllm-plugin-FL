# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""V4.1 target/draft phase boundaries outside captured GPU graphs."""

from functools import wraps
from inspect import signature

import torch

from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager
from vllm.v1.worker.gpu.spec_decode.dspark.speculator import DSparkSpeculator

logger = init_logger(__name__)


def _uses_dspark_dp_ep(config):
    spec = config.speculative_config
    return (
        config.model_config.hf_config.model_type == "deepseek_v41"
        and spec is not None
        and spec.method == "dspark"
        and config.parallel_config.enable_expert_parallel
        and config.parallel_config.data_parallel_size > 1
    )


def _wrap_prepare_forward(original):
    @wraps(original)
    def prepare_forward(self, num_unpadded_tokens):
        result = original(self, num_unpadded_tokens)
        if _uses_dspark_dp_ep(self.vllm_config):
            logger.info_once("V4.1 DSpark query-ready phase fence enabled")
            get_ep_group().barrier()
        return result

    return prepare_forward


def _wrap_replay(original):
    @wraps(original)
    def replay(self, desc):
        scoped = _uses_dspark_dp_ep(self.vllm_config)
        if scoped:
            logger.info_once(
                "V4.1 target/draft graph phases use explicit completion boundaries"
            )
            torch.cuda.synchronize()
            get_ep_group().barrier()
        result = original(self, desc)
        if scoped:
            torch.cuda.synchronize()
            get_ep_group().barrier()
        return result

    return replay


def prepare():
    """Validate both native boundaries before constructing any replacements."""
    targets = (
        (
            DSparkSpeculator,
            "_prepare_eplb_forward",
            ["self", "num_unpadded_tokens"],
            _wrap_prepare_forward,
            "_fl_v41_query_ready",
        ),
        (
            CudaGraphManager,
            "run_fullgraph",
            ["self", "desc"],
            _wrap_replay,
            "_fl_v41_completion",
        ),
    )
    bindings = []
    for target, name, parameters, wrap, marker in targets:
        original = getattr(target, name)
        if list(signature(original).parameters) != parameters:
            raise RuntimeError(f"unsupported native V4.1 graph boundary: {name}")
        if not getattr(original, marker, False):
            replacement = wrap(original)
            setattr(replacement, marker, True)
            bindings.append((target, name, replacement))
    return bindings
