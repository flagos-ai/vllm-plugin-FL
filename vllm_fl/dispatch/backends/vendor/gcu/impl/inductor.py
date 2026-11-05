# Copyright (c) 2026 BAAI. All rights reserved.

"""Adapt TorchGCU compilation metadata to the installed Torch interfaces."""

import inspect
import os
import sys
from collections.abc import Mapping
from functools import wraps
from types import ModuleType

_PARAMETERS = ("size_hints", "reduction_hint", "inductor_meta", "triton_meta")
_ADAPTER_MARKER = "_fl_gcu_persistent_reduction_adapter"
_REGISTRATION_MARKER = "_fl_gcu_persistent_reduction_registration"
_I64_OPTIONS_MARKER = "_fl_gcu_i64_compile_options_adapter"
_I64_SIGNATURES = frozenset({"i64", "u64", "*i64", "*u64"})
_RESOURCE_ERROR_MARKER = "_fl_gcu_register_resource_error_adapter"
_REGISTER_ERROR = "error: ran out of registers during register allocation"


def _install_gcu_resource_errors(heuristics) -> None:
    autotuner = getattr(heuristics, "CachingAutotuner", None)
    original = getattr(autotuner, "_precompile_config", None)
    if original is None:
        raise RuntimeError("Torch CachingAutotuner precompile interface is missing")
    if getattr(original, _RESOURCE_ERROR_MARKER, False) is True:
        return
    if tuple(inspect.signature(original).parameters) != ("self", "cfg"):
        raise RuntimeError("Unknown Torch CachingAutotuner precompile ABI")
    resource_error = getattr(heuristics, "OutOfResources", None)
    try:
        resource_parameters = tuple(inspect.signature(resource_error).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Unknown Triton OutOfResources ABI") from error
    if (
        not isinstance(resource_error, type)
        or not issubclass(resource_error, Exception)
        or resource_parameters != ("required", "limit", "name")
    ):
        raise RuntimeError("Unknown Triton OutOfResources ABI")

    compiler_error = getattr(heuristics, "PTXASError", None)
    try:
        compiler_parameters = tuple(inspect.signature(compiler_error).parameters)
    except (TypeError, ValueError) as error:
        raise RuntimeError("Unknown Triton PTXASError ABI") from error
    if (
        not isinstance(compiler_error, type)
        or not issubclass(compiler_error, Exception)
        or compiler_parameters != ("error_message",)
    ):
        raise RuntimeError("Unknown Triton PTXASError ABI")

    @wraps(original)
    def precompile_gcu_config(self, cfg):
        try:
            return original(self, cfg)
        except Exception as error:
            if self.device_props.type != "gcu" or type(error) is not Exception:
                raise
            first_line = str(error).splitlines()[:1]
            if first_line == [_REGISTER_ERROR]:
                raise resource_error("unknown", "unknown", "GCU registers") from error
            linker_errors = {
                f"ld.lld: error: relocation R_DTU_ADDR16_{part}_ICALL cannot be "
                "used against symbol tops_truncf_v2_fp32_to_v1_bf16; "
                "recompile with -fPIC"
                for part in ("LO", "HI")
            }
            if first_line and first_line[0] in linker_errors:
                raise compiler_error(str(error)) from error
            raise

    setattr(precompile_gcu_config, _RESOURCE_ERROR_MARKER, True)
    autotuner._precompile_config = precompile_gcu_config


def _install_gcu_i64_compile_options(heuristics) -> None:
    autotuner = getattr(heuristics, "CachingAutotuner", None)
    original = getattr(autotuner, "_create_compile_options", None)
    if original is None:
        raise RuntimeError("Torch CachingAutotuner compile options are missing")
    if getattr(original, _I64_OPTIONS_MARKER, False) is True:
        return
    if tuple(inspect.signature(original).parameters) != ("self", "cfg", "compile_meta"):
        raise RuntimeError("Unknown CachingAutotuner compile options ABI")

    @wraps(original)
    def create_gcu_compile_options(self, cfg, compile_meta):
        options = original(self, cfg, compile_meta)
        if getattr(self.device_props, "type", None) != "gcu":
            return options
        if "enable_i64" in options or "ENABLE_I64" in options:
            return options
        explicit = {
            key: compile_meta[key]
            for key in ("enable_i64", "ENABLE_I64")
            if key in compile_meta
        }
        if explicit:
            return {**options, **explicit}
        # Even an all-i32 signature can contain internal i64 arithmetic:
        # Inductor embedding explicitly widens token indices before lookup.
        # Argument signatures alone cannot establish that i64 is unused.
        # Torch keeps dynamic sizes as i64 to avoid overflow. Its compiler
        # option whitelist drops GCU's flag, so pass it at the actual options
        # boundary. The GCU backend requires an explicit option; setting only
        # ENABLE_I64_CHECK does not enable its 64-bit lowering.
        return {**options, "enable_i64": True}

    setattr(create_gcu_compile_options, _I64_OPTIONS_MARKER, True)
    autotuner._create_compile_options = create_gcu_compile_options


def _configuration_arity(function) -> int:
    parameters = tuple(inspect.signature(function).parameters.values())
    if (
        len(parameters) not in (3, 4)
        or tuple(parameter.name for parameter in parameters)
        != _PARAMETERS[: len(parameters)]
        or any(
            parameter.kind
            not in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            for parameter in parameters
        )
        or parameters[0].default is not inspect.Parameter.empty
        or parameters[1].default is not False
        or parameters[2].default is not None
        or (len(parameters) == 4 and parameters[3].default is not None)
    ):
        raise RuntimeError("Unknown TorchGCU persistent reduction configuration ABI")
    return len(parameters)


def _adapt_registered_persistent_reduction(provider, heuristics) -> str:
    original = heuristics._persistent_reduction_configs
    if getattr(original, _ADAPTER_MARKER, False) is True:
        return "already_adapted"
    generator_type = provider.GCUTritonConfigGenerator
    if (
        not inspect.ismethod(original)
        or not isinstance(original.__self__, generator_type)
        or original.__func__ is not generator_type.persistent_reduction_configs
    ):
        # Before TorchGCU's first lazy registration this is Torch's own
        # function. Other providers are not ours to replace either.
        return "unchanged_other_provider"
    if _configuration_arity(original) == 4:
        return "unchanged_compatible_gcu"

    # Do not use wraps here: its __wrapped__ would advertise the old three-
    # argument signature to callers inspecting this new four-argument ABI.
    def persistent_reduction_configs(
        size_hints, reduction_hint=False, inductor_meta=None, triton_meta=None
    ):
        if triton_meta is not None:
            if not isinstance(triton_meta, Mapping):
                raise TypeError(
                    "TorchGCU persistent reduction triton_meta must be a mapping"
                )
            if triton_meta.get("native_matmul"):
                raise NotImplementedError(
                    "TorchGCU's three-argument persistent reduction provider "
                    "does not support native_matmul metadata"
                )
        # Ordinary reduction metadata has no other use in the observed Torch
        # implementation. Preserve all three original objects and vendor
        # configuration results; this adapter changes no kernel algorithm.
        return original(size_hints, reduction_hint, inductor_meta)

    setattr(persistent_reduction_configs, _ADAPTER_MARKER, True)
    persistent_reduction_configs._fl_gcu_original = original
    heuristics._persistent_reduction_configs = persistent_reduction_configs
    return "adapted"


def _install_gcu_inductor_hooks(provider, heuristics) -> None:
    patcher_type = provider.GCUInductorPatcher
    register = patcher_type.patch_triton_heuristics
    if getattr(register, _REGISTRATION_MARKER, False) is not True:

        @wraps(register)
        def register_compatible(self, *args, **kwargs):
            # TorchGCU registers heuristics lazily at its first torch.compile.
            # Installing only the current slot would be overwritten then.
            result = register(self, *args, **kwargs)
            _adapt_registered_persistent_reduction(provider, heuristics)
            _install_gcu_i64_compile_options(heuristics)
            _install_gcu_resource_errors(heuristics)
            return result

        setattr(register_compatible, _REGISTRATION_MARKER, True)
        patcher_type.patch_triton_heuristics = register_compatible
    # Also handle a process where TorchGCU has already registered its provider.
    _adapt_registered_persistent_reduction(provider, heuristics)
    _install_gcu_i64_compile_options(heuristics)
    _install_gcu_resource_errors(heuristics)


def _loaded_gcu_inductor_providers(provider: ModuleType) -> list[ModuleType]:
    # TorchGCU exposes torch.gcu as well as torch_gcu.gcu. Importing an
    # inductor submodule through both names creates distinct classes and
    # registration state, even though both modules execute the same file.
    alias = sys.modules.get("torch.gcu.inductor.backend_register")
    if alias is None or alias is provider:
        return [provider]
    original_file = getattr(provider, "__file__", None)
    alias_file = getattr(alias, "__file__", None)
    try:
        same_source = bool(original_file and alias_file) and os.path.samefile(
            original_file, alias_file
        )
    except OSError:
        same_source = False
    if not same_source:
        raise RuntimeError("TorchGCU Inductor aliases resolve to different sources")
    return [provider, alias]


def _install_gcu_dtype_preserving_codegen(provider) -> None:
    from torch._inductor.graph import GraphLowering

    helper = provider.GCUCompileHelper
    original = helper.compile_to_module
    native = GraphLowering._compile_to_module
    if original is native:
        return
    if len(inspect.signature(original).parameters) != 1:
        raise RuntimeError("Unknown TorchGCU compile_to_module ABI")
    if tuple(inspect.signature(native).parameters) != ("self",):
        raise RuntimeError("Unknown Torch GraphLowering compile ABI")
    # The vendor helper globally rewrites int64/i64 to int32/i32 in source,
    # including pointer signatures, without converting input tensor storage.
    # Native Torch retains those ABIs and the current artifact-saving protocol.
    helper.compile_to_module = staticmethod(native)


def apply_inductor_gcu_patch() -> None:
    from vllm.platforms import current_platform

    if getattr(current_platform, "vendor_name", None) != "enflame":
        return
    # A changed code generator must not reuse artifacts with corrupted ABIs.
    # Preserve a caller's cache tag while adding a versioned backend salt.
    import torch.compiler.config as compiler_config
    from torch._inductor.runtime import triton_heuristics
    from torch_gcu.gcu.inductor import backend_register

    salt = "fl_gcu_preserve_dtype_v1"
    tag = compiler_config.cache_key_tag
    if salt not in tag.split(":"):
        compiler_config.cache_key_tag = f"{tag}:{salt}" if tag else salt
    for provider in _loaded_gcu_inductor_providers(backend_register):
        _install_gcu_dtype_preserving_codegen(provider)
        _install_gcu_inductor_hooks(provider, triton_heuristics)
