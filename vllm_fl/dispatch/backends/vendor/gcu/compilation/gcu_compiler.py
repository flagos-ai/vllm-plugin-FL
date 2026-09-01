# Copyright (c) 2026 BAAI. All rights reserved.

"""
GCU inductor adaptor with lowering() for vllm-plugin-FL.

Provides the same mechanism as vllm_gcu's GCUInductorAdaptor:
set all aten/prims ops to ``make_fallback`` so that inductor does
NOT decompose them into low-level IR.  Instead they are passed through
as external calls to GCU's tops_extension / torch_gcu native
implementation.

Without this, inductor tries to lower ops like ``aten.mean.dim`` →
``Reduction`` → ``SchedulerNode``, which fails because "gcu" has no
scheduler registered in torch._inductor.
"""

from contextlib import contextmanager
from unittest.mock import patch

import torch
from torch._inductor.codegen.common import device_codegens, get_scheduling_for_device
from torch._inductor.codegen.triton import TritonScheduling
from vllm.compilation.compiler_interface import (
    InductorAdaptor,
    InductorStandaloneAdaptor,
)
from vllm.config.compilation import CompilationMode
from vllm.config.vllm import OptimizationLevel

import vllm.envs as envs

try:
    device_schedule = get_scheduling_for_device("gcu")
    if device_schedule == TritonScheduling:
        device_codegens.pop("gcu")
    import tops_extension.torch  # noqa
except Exception:
    pass


@contextmanager
def lowering():
    """Context manager that makes all aten/prims operators fallback.

    Inside this context, inductor will NOT decompose any operator into
    lower-level IR.  Instead every operator is treated as an opaque
    external call, which GCU executes via tops_extension / torch_gcu.
    """
    import torch._inductor.lowering as til

    BLACK_LIST = {
        "name", "__doc__", "__loader__", "__name__",
        "__package__", "__spec__", "_dir", "__file__", "__all__",
    }
    origin_lowerings = {}
    skip_lowerings = []

    for name in dir(torch.ops.aten):
        if name in BLACK_LIST:
            continue
        op = getattr(torch.ops.aten, name)

        if isinstance(op, torch._ops.OpOverloadPacket):
            for ol in op.overloads():
                op_overload = til.get_overloads(getattr(op, ol))
                if op_overload[0] in til.lowerings:
                    origin_lowerings.update(
                        dict.fromkeys(op_overload, til.lowerings[op_overload[0]])
                    )
                else:
                    skip_lowerings.append(op_overload[0])
        elif isinstance(
            op, (torch._ops.OpOverload, torch._ops.HigherOrderOperator)
        ):
            op_overload = til.get_overloads(op)
            if op_overload[0] in til.lowerings:
                origin_lowerings.update(
                    dict.fromkeys(op_overload, til.lowerings[op_overload[0]])
                )
            else:
                skip_lowerings.append(op_overload[0])

        til.make_fallback(op, warn=False, override_decomp=True)

    for name in dir(torch.ops.prims):
        if name not in BLACK_LIST:
            op = getattr(torch.ops.prims, name)
            til.make_fallback(op, warn=False, override_decomp=True)

    yield

    for op_overload, _ in origin_lowerings.items():
        til.register_lowering(op_overload, type_promotion_kind=None)(
            origin_lowerings[op_overload]
        )
    for op_overload in skip_lowerings:
        til.lowerings.pop(op_overload)


@contextmanager
def _no_fp64_for_triton_float_args():
    """Force inductor to emit ``fp32`` (not ``fp64``) for Python-float
    arguments of user-defined Triton kernels.

    Inductor's ``signature_of`` maps Python floats to ``fp64`` when
    ``config._use_fp64_for_unbacked_floats`` is True (the default).  The GCU
    backend has no fp64 support: mixing an fp64 scalar with fp32 values in a
    kernel generates ``arith.extf`` (f32->f64), which the GCU MLIR pipeline
    marks illegal, failing compilation of user-defined Triton kernels that
    reach inductor via the ``triton_kernel_wrapper_mutation`` HOP.
    """
    import torch._inductor.config as inductor_config

    with inductor_config.patch(_use_fp64_for_unbacked_floats=False):
        yield


class GCUInductorAdaptor(InductorAdaptor):
    """vllm-plugin-FL GCU compile backend.

    Wraps every compilation with ``lowering()`` so that inductor does not
    attempt to decompose aten/prims operators for the "gcu" device.
    """

    def compile(self, *args, **kwargs):
        with (
            patch(
                "torch._inductor.fx_passes.reinplace.should_reinplace_scatter",
                lambda node: False,
            ),
            lowering(),
            _no_fp64_for_triton_float_args(),
        ):
            return super().compile(*args, **kwargs)


class GCUInductorStandaloneAdaptor(InductorStandaloneAdaptor):
    """vllm-plugin-FL GCU standalone compile backend."""

    def __init__(self):
        super().__init__(save_format=envs.VLLM_COMPILE_CACHE_SAVE_FORMAT)

    def compile(self, *args, **kwargs):
        with (
            lowering(),
            _no_fp64_for_triton_float_args(),
        ):
            return super().compile(*args, **kwargs)


def update_gcu_compilation_config(vllm_config, compilation_config):
    if compilation_config.mode is None:
        if vllm_config.optimization_level > OptimizationLevel.O0:
            compilation_config.mode = CompilationMode.VLLM_COMPILE
        else:
            compilation_config.mode = CompilationMode.NONE

    if compilation_config.mode == CompilationMode.VLLM_COMPILE:
        compilation_config.pass_config.fuse_norm_quant = True
        compilation_config.pass_config.fuse_act_quant = True
        compilation_config.pass_config.fuse_attn_quant = False
        compilation_config.pass_config.eliminate_noops = True
        compilation_config.pass_config.fuse_gemm_comms = False
        compilation_config.pass_config.fuse_allreduce_rms = False
        compilation_config.pass_config.enable_qk_norm_rope_fusion = False

        compilation_config.custom_ops = ["all"]
        compilation_config.ir_enable_torch_wrap = True

        # Disable inductor features that generate invalid Triton
        # code on GCU (combo kernel xloop bug, triton autotune).
        compilation_config.inductor_compile_config["combo_kernels"] = False
        compilation_config.inductor_compile_config[
            "benchmark_combo_kernel"
        ] = False
        compilation_config.inductor_compile_config[
            "triton.autotune_at_compile_time"
        ] = False

        # GCU has no fp64 support: Python-float args of user-defined Triton
        # kernels must be emitted as fp32 (see _no_fp64_for_triton_float_args).
        compilation_config.inductor_compile_config[
            "_use_fp64_for_unbacked_floats"
        ] = False

