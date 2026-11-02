# Copyright (c) 2026 BAAI. All rights reserved.

"""Runtime initialization shared by the NVIDIA and multi-vendor workers."""

from functools import wraps
from types import MethodType
from typing import Any

from vllm.logger import init_logger

import vllm_fl.envs as fl_envs
from vllm_fl.attention.utils import patch_mm_encoder_attention
from vllm_fl.dispatch.io_common import managed_inference_mode
from vllm_fl.utils import get_flag_gems_whitelist_blacklist

logger = init_logger(__name__)


def is_residual_scattered_for_sp(vllm_config, num_input_tokens: int) -> bool:
    """Keep the FL pipeline tensor-layout contract across vLLM versions."""
    from vllm.v1.worker import utils as worker_utils

    native = getattr(worker_utils, "is_residual_scattered_for_sp", None)
    if native is not None:
        return native(vllm_config, num_input_tokens)

    # vLLM 0.32 removed this helper; FL still prepares legacy PP transfers.
    compilation = vllm_config.compilation_config
    if not compilation.pass_config.enable_sp:
        return False
    tp = vllm_config.parallel_config.tensor_parallel_size
    if tp == 1:
        return False
    assert compilation.use_inductor_graph_partition or not compilation.splitting_ops, (
        "Sequence parallelism requires full-graph compilation"
    )
    assert num_input_tokens % tp == 0, (
        "Sequence parallelism requires TP-aligned token counts"
    )
    return True


def _initialize_fl_runtime(rank: int) -> None:
    """Install FL dispatch hooks before model/operator construction."""
    from vllm_fl.ops.custom_ops import register_oot_ops

    # CustomOp caches its selected forward method in __init__, so the
    # multimodal attention patch must run before model construction.
    patch_mm_encoder_attention()
    register_oot_ops()

    if not fl_envs.USE_FLAGGEMS:
        return

    import flag_gems

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    should_record = rank == 0
    common = {
        "record": should_record,
        "once": True,
        "path": fl_envs.FLAGGEMS_ENABLE_OPLIST_PATH,
    }
    if whitelist:
        logger.info("[FlagGems] Enable only the following ops: %s", whitelist)
        flag_gems.only_enable(include=whitelist, **common)
    elif blacklist:
        logger.info("[FlagGems] Disable the following ops: %s", blacklist)
        flag_gems.enable(unused=blacklist, **common)
    else:
        logger.info("[FlagGems] Enable all ops")
        flag_gems.enable(**common)


_NATIVE_RUNNER_IO_METHODS = ("execute_model", "sample_tokens", "pool")


def _install_native_runner_io_methods(
    model_runner: Any, method_names=_NATIVE_RUNNER_IO_METHODS
) -> None:
    """Replace native runner inference-mode wrappers on one runner instance.

    vLLM's NVIDIA V1 and V2 runners decorate their execution entry points
    with ``torch.inference_mode``. IO dumping needs ``torch.no_grad`` while its
    TorchDispatchMode is active. Unwrapping only that outer decorator keeps the
    native runner implementation and any inner vLLM decorators unchanged.
    """
    for method_name in method_names:
        bound_method = getattr(model_runner, method_name, None)
        if bound_method is None or getattr(bound_method, "_vllm_fl_managed_io", False):
            continue

        method = getattr(bound_method, "__func__", bound_method)
        native_impl = getattr(method, "__wrapped__", None)
        if native_impl is None:
            raise RuntimeError(
                f"Native model runner method {method_name!r} no longer exposes "
                "the vLLM inference-mode wrapper"
            )

        managed_impl = _native_runner_io_method(
            native_impl, method_name in ("execute_model", "sample_tokens")
        )
        managed_impl._vllm_fl_managed_io = True
        setattr(model_runner, method_name, MethodType(managed_impl, model_runner))


def prepare_native_model_runner(
    model_runner, enforce_eager, load_dummy_weights, use_v2_model_runner=False
):
    """Install common FL hooks after the native V1 runner loads its model."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    if not use_v2_model_runner and not isinstance(model_runner, GPUModelRunner):
        return
    from vllm_fl.compilation.graph import adapt_graph_wrapper
    from vllm_fl.dispatch.io_dumper import (
        init_io_dump_from_env,
        is_dump_enabled,
        register_io_module_hooks,
    )

    init_io_dump_from_env(enforce_eager)
    model_runner.model = adapt_graph_wrapper(model_runner.model)
    if is_dump_enabled():
        _install_native_runner_io_methods(
            model_runner,
            _NATIVE_RUNNER_IO_METHODS
            if use_v2_model_runner
            else (*_NATIVE_RUNNER_IO_METHODS, "_dummy_run"),
        )
        if not load_dummy_weights:
            register_io_module_hooks(model_runner.get_model())


def _native_runner_io_method(native_impl, advance_step):
    @wraps(native_impl)
    @managed_inference_mode()
    def call(*args, **kwargs):
        output = native_impl(*args, **kwargs)
        if advance_step and output is not None:
            from vllm_fl.dispatch.io_dumper import advance_io_step

            advance_io_step()
        return output

    return call
