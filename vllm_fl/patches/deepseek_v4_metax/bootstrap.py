"""Deterministic one-shot startup. Explicit imports/providers; no runtime source edits."""

import importlib
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from .config import load
from .compat import install_vendor_guard, install_serial_execution

_initialized = False


def _prepare_compatibility(profile):
    install_vendor_guard()
    import torch

    torch._dynamo.config.ignore_logger_methods.update(
        getattr(logging.Logger, name)
        for name in (
            "debug",
            "info",
            "warning",
            "error",
            "exception",
            "critical",
            "log",
        )
    )
    # Preserve the accepted order: existing FlagGems tuners are created first;
    # the tile policy only affects subsequent config queries, as before.
    from vllm_fl import dsv4_config

    install_serial_execution(profile["compatibility"]["multi_stream_source_sha256"])
    import flag_gems
    from flag_gems.runtime import metax_launch_policy

    metax_launch_policy.configure(True)
    import triton.backends.metax.compiler as compiler

    compiler_entry = compiler.metax.translate_llvmir_to_mcfatbin
    import flaggems_vllm
    from vllm_fl.dispatch.backends.vendor.metax.impl.dsv4.attention import (
        tuning as attention_tuning,
    )

    assert asdict(attention_tuning.KernelPolicy()) == profile["kernels"]
    attention_tuning.configure(profile["kernels"])
    dsv4_config.configure(profile["framework"], profile["moe"])
    assert flaggems_vllm.current_work_registrar is None
    assert compiler.metax.translate_llvmir_to_mcfatbin is compiler_entry
    from .rms_policy import install as install_rms

    install_rms(profile["rms"], profile["audit_dir"])
    from .validation import configure as configure_validation

    configure_validation(
        profile["hash_tables"],
        profile["audit_dir"],
    )
    return


def _bind_dispatch(profile):
    from vllm_fl.dispatch import CachedOp, get_default_manager

    from vllm_fl.utils import get_flag_gems_whitelist_blacklist
    from vllm_fl.dispatch.config import get_oot_blacklist

    whitelist, blacklist = get_flag_gems_whitelist_blacklist()
    assert (
        whitelist is None and blacklist == profile["dispatch"]["flagos_blacklist"]
    ), "ATen policy differs from the sealed profile"
    assert get_oot_blacklist() == profile["dispatch"]["oot_blacklist"]
    mgr = get_default_manager()
    mgr.ensure_initialized()
    modules = (
        "vllm_fl.models.deepseek_v4",
        "vllm_fl.ops.deepseek_v4_attention",
        "vllm_fl.ops.sparse_attn_indexer",
        "vllm_fl.ops.layernorm",
        "vllm_fl.ops.rotary_embedding",
        "vllm_fl.ops.activation",
        "vllm_fl.ops.fused_moe.router",
        "vllm_fl.ops.fused_moe.fused_moe",
        "vllm_fl.ops.fused_moe.fused_moe_utils",
        "vllm_fl.ops.fused_moe.activation",
        "vllm_fl.ops.fused_moe.gated_linear",
        "vllm_fl.quantization.mxfp4.mxfp4_marlin",
        "vllm_fl.platform",
    )
    bindings = {}
    for name in modules:
        mod = importlib.import_module(name)
        for attr, obj in vars(mod).items():
            if isinstance(obj, CachedOp):
                try:
                    impl = obj.bind()
                except Exception:
                    if obj._op_name == "deepseek_v4_mega_moe_experts":
                        continue
                    raise
                bindings[name + "." + attr] = impl.impl_id
    return bindings, blacklist


def _record_startup(profile, bindings, blacklist):
    assert not any(
        n == "vllm_metax" or n.startswith("vllm_metax.") for n in sys.modules
    )
    record = {
        "release": profile["release"],
        "pid": os.getpid(),
        "bindings": bindings,
        "kernel_policy": profile["kernels"],
        "framework_policy": profile["framework"],
        "aten_blacklist": blacklist,
        "vendor_imports": [],
        "compiler_entry_unchanged": True,
        "global_aten_fgv_registration": False,
    }
    folder = Path(profile["audit_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"bootstrap_{os.getpid()}.json").write_text(json.dumps(record, indent=2))


def initialize():
    """Install the pinned compatibility policy, then bind standard dispatch ops."""
    global _initialized
    if _initialized:
        return
    profile = load()
    _prepare_compatibility(profile)
    bindings, blacklist = _bind_dispatch(profile)
    _record_startup(profile, bindings, blacklist)
    _initialized = True
    print(
        "DSV4_CLEAN_BOOTSTRAP "
        + json.dumps({"release": profile["release"], "bound_ops": len(bindings)}),
        flush=True,
    )
