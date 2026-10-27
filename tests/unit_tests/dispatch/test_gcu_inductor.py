# Copyright (c) 2026 BAAI. All rights reserved.

import inspect
import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

if os.environ.get("FL_TEST_PLATFORM", "enflame") != "enflame":
    pytest.skip("Enflame-specific Inductor adapter", allow_module_level=True)

from vllm_fl.dispatch.backends.vendor.gcu.impl import inductor


def make_provider(function=None, shared_heuristics=None):
    calls = []
    result = object()

    class GCUTritonConfigGenerator:
        def persistent_reduction_configs(
            self, size_hints, reduction_hint=False, inductor_meta=None
        ):
            calls.append((size_hints, reduction_hint, inductor_meta))
            return result

    if function is not None:
        GCUTritonConfigGenerator.persistent_reduction_configs = function
    generator = GCUTritonConfigGenerator()
    heuristics = shared_heuristics or SimpleNamespace(
        _persistent_reduction_configs=generator.persistent_reduction_configs
    )
    if not hasattr(heuristics, "CachingAutotuner"):

        class CachingAutotuner:
            def __init__(self, device):
                self.device_props = SimpleNamespace(type=device)
                self.options = {"num_warps": 4, "num_stages": 1}

            def _create_compile_options(self, cfg, compile_meta):
                self.received = (cfg, compile_meta)
                return self.options

            def _precompile_config(self, cfg):
                self.compile_received = cfg
                if isinstance(cfg, BaseException):
                    raise cfg
                return cfg

        heuristics.CachingAutotuner = CachingAutotuner
    if not hasattr(heuristics, "OutOfResources"):

        class OutOfResources(Exception):
            def __init__(self, required, limit, name):
                self.required, self.limit, self.name = required, limit, name

        heuristics.OutOfResources = OutOfResources
    registrations = []

    class GCUInductorPatcher:
        def patch_triton_heuristics(self, marker=None):
            registrations.append(marker)
            heuristics._persistent_reduction_configs = (
                generator.persistent_reduction_configs
            )
            return result

    provider = SimpleNamespace(
        GCUTritonConfigGenerator=GCUTritonConfigGenerator,
        GCUInductorPatcher=GCUInductorPatcher,
    )
    return provider, heuristics, calls, result, registrations


@pytest.mark.parametrize("style", ["positional", "keyword"])
def test_gcu_resource_adapter_preserves_success_and_lazy_registration(style):
    provider, heuristics, _, _, _ = make_provider()
    original = heuristics.CachingAutotuner._precompile_config
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    wrapped = heuristics.CachingAutotuner._precompile_config
    autotuner = heuristics.CachingAutotuner("gcu")
    cfg = object()
    result = (
        autotuner._precompile_config(cfg)
        if style == "positional"
        else autotuner._precompile_config(cfg=cfg)
    )
    assert result is cfg and autotuner.compile_received is cfg
    assert inspect.signature(wrapped) == inspect.signature(original)
    provider.GCUInductorPatcher().patch_triton_heuristics()
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    assert heuristics.CachingAutotuner._precompile_config is wrapped


def test_gcu_register_exhaustion_does_not_discard_valid_autotune_candidates():
    _, heuristics, _, _, _ = make_provider()
    inductor._install_gcu_resource_errors(heuristics)
    autotuner = heuristics.CachingAutotuner("gcu")
    first, last = object(), object()
    failure = Exception(
        "error: ran out of registers during register allocation\n"
        "llc invocation failed: llcResult 1"
    )
    results, failures = [], []
    for cfg in [first, failure, last]:
        try:
            results.append(autotuner._precompile_config(cfg))
        except heuristics.OutOfResources as error:
            failures.append(error)
    assert results == [first, last]
    assert len(failures) == 1 and failures[0].__cause__ is failure
    assert failures[0].required == failures[0].limit == "unknown"
    assert failures[0].name == "GCU registers"
    # A lone failed configuration must still raise, never become a result.
    with pytest.raises(heuristics.OutOfResources):
        autotuner._precompile_config(failure)


@pytest.mark.parametrize(
    "device,error",
    [
        ("cuda", Exception(inductor._REGISTER_ERROR)),
        ("cpu", Exception(inductor._REGISTER_ERROR)),
        ("gcu", Exception("error: invalid compiler IR")),
        ("gcu", Exception("context\n" + inductor._REGISTER_ERROR)),
        ("gcu", Exception()),
        ("gcu", RuntimeError(inductor._REGISTER_ERROR)),
    ],
)
def test_gcu_resource_adapter_preserves_other_failures(device, error):
    _, heuristics, _, _, _ = make_provider()
    inductor._install_gcu_resource_errors(heuristics)
    with pytest.raises(type(error)) as caught:
        heuristics.CachingAutotuner(device)._precompile_config(error)
    assert caught.value is error


def test_gcu_resource_adapter_preserves_existing_resource_error():
    _, heuristics, _, _, _ = make_provider()
    error = heuristics.OutOfResources(512, 256, "registers")
    inductor._install_gcu_resource_errors(heuristics)
    with pytest.raises(heuristics.OutOfResources) as caught:
        heuristics.CachingAutotuner("gcu")._precompile_config(error)
    assert caught.value is error


@pytest.mark.parametrize("interface", ["missing", "precompile", "resource_error"])
def test_gcu_resource_adapter_rejects_unknown_interfaces(interface):
    _, heuristics, _, _, _ = make_provider()
    if interface == "missing":
        del heuristics.CachingAutotuner._precompile_config
    elif interface == "precompile":
        heuristics.CachingAutotuner._precompile_config = lambda self, config: config
    else:
        heuristics.OutOfResources = RuntimeError
    with pytest.raises(RuntimeError, match="(missing|ABI)"):
        inductor._install_gcu_resource_errors(heuristics)


@pytest.mark.parametrize("dtype", ["i64", "u64", "*i64", "*u64"])
@pytest.mark.parametrize("style", ["positional", "keyword"])
def test_gcu_i64_option_reaches_compiler_without_narrowing_metadata(dtype, style):
    provider, heuristics, _, _, _ = make_provider()
    original = heuristics.CachingAutotuner._create_compile_options
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    wrapped = heuristics.CachingAutotuner._create_compile_options
    autotuner = heuristics.CachingAutotuner("gcu")
    meta = {"signature": {"ks0": dtype, "xnumel": "i32"}, "constants": {}}
    before, cfg = dict(meta), object()
    if style == "positional":
        options = autotuner._create_compile_options(cfg, meta)
    else:
        options = autotuner._create_compile_options(cfg=cfg, compile_meta=meta)
    assert autotuner.received[0] is cfg and autotuner.received[1] is meta
    assert meta == before and "enable_i64" not in meta
    assert options == {**autotuner.options, "enable_i64": True}
    assert options is not autotuner.options
    assert "enable_i64" not in autotuner.options
    assert inspect.signature(wrapped) == inspect.signature(original)
    provider.GCUInductorPatcher().patch_triton_heuristics()
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    assert heuristics.CachingAutotuner._create_compile_options is wrapped


@pytest.mark.parametrize(
    "device,dtype", [("cuda", "i64"), ("cpu", "i64"), ("gcu", "i32")]
)
def test_i64_option_does_not_change_other_kernels(device, dtype):
    _, heuristics, _, _, _ = make_provider()
    inductor._install_gcu_i64_compile_options(heuristics)
    autotuner = heuristics.CachingAutotuner(device)
    options = autotuner._create_compile_options(object(), {"signature": {"ks0": dtype}})
    assert options is autotuner.options


@pytest.mark.parametrize("key", ["enable_i64", "ENABLE_I64"])
@pytest.mark.parametrize("value", [False, True])
@pytest.mark.parametrize("location", ["metadata", "options"])
def test_gcu_i64_options_preserve_explicit_preference(key, value, location):
    _, heuristics, _, _, _ = make_provider()
    inductor._install_gcu_i64_compile_options(heuristics)
    autotuner = heuristics.CachingAutotuner("gcu")
    meta = {"signature": {"ks0": "i64"}}
    if location == "metadata":
        meta[key] = value
    else:
        autotuner.options[key] = value
        meta[key] = not value
    before = dict(meta)
    options = autotuner._create_compile_options(object(), meta)
    assert options[key] is value and meta == before
    assert ("ENABLE_I64" if key == "enable_i64" else "enable_i64") not in options
    if location == "options":
        assert options is autotuner.options
    else:
        assert key not in autotuner.options


def test_gcu_i64_option_preserves_both_explicit_metadata_keys():
    _, heuristics, _, _, _ = make_provider()
    inductor._install_gcu_i64_compile_options(heuristics)
    autotuner = heuristics.CachingAutotuner("gcu")
    meta = {"signature": {"ks0": "i64"}, "enable_i64": False, "ENABLE_I64": True}
    options = autotuner._create_compile_options(object(), meta)
    assert options == {**autotuner.options, "enable_i64": False, "ENABLE_I64": True}


def test_i64_options_reject_unknown_abi():
    with pytest.raises(RuntimeError, match="compile options are missing"):
        inductor._install_gcu_i64_compile_options(SimpleNamespace())
    unknown = SimpleNamespace(_create_compile_options=lambda self, metadata: {})
    with pytest.raises(RuntimeError, match="compile options ABI"):
        inductor._install_gcu_i64_compile_options(
            SimpleNamespace(CachingAutotuner=unknown)
        )


def test_i64_options_preserve_original_failure():
    failure = ValueError("original compile option failure")

    class FailingAutotuner:
        def _create_compile_options(self, cfg, compile_meta):
            raise failure

    inductor._install_gcu_i64_compile_options(
        SimpleNamespace(CachingAutotuner=FailingAutotuner)
    )
    with pytest.raises(ValueError) as raised:
        FailingAutotuner()._create_compile_options(None, {})
    assert raised.value is failure


@pytest.mark.parametrize("style", ["legacy", "positional", "keyword"])
def test_adapter_preserves_original_objects_and_configuration(style):
    provider, heuristics, calls, result, _ = make_provider()
    size_hints = {"x": 8, "r0_": 128}
    reduction_hint = object()
    inductor_meta = {"autotune_pointwise": False}
    triton_meta = {"native_matmul": False, "signature": {0: "*bf16"}}
    before = (dict(size_hints), dict(inductor_meta), dict(triton_meta))
    assert (
        inductor._adapt_registered_persistent_reduction(provider, heuristics)
        == "adapted"
    )
    adapted = heuristics._persistent_reduction_configs
    assert tuple(inspect.signature(adapted).parameters) == inductor._PARAMETERS
    if style == "legacy":
        actual = adapted(size_hints, reduction_hint, inductor_meta)
    elif style == "positional":
        actual = adapted(size_hints, reduction_hint, inductor_meta, triton_meta)
    else:
        actual = adapted(
            size_hints=size_hints,
            reduction_hint=reduction_hint,
            inductor_meta=inductor_meta,
            triton_meta=triton_meta,
        )
    assert actual is result
    assert len(calls) == 1
    assert all(
        actual is expected
        for actual, expected in zip(
            calls[0], (size_hints, reduction_hint, inductor_meta)
        )
    )
    assert (size_hints, inductor_meta, triton_meta) == before


@pytest.mark.parametrize("native_matmul", [True, 1, "native"])
def test_native_matmul_is_rejected_before_calling_vendor(native_matmul):
    provider, heuristics, calls, _, _ = make_provider()
    inductor._adapt_registered_persistent_reduction(provider, heuristics)
    with pytest.raises(NotImplementedError, match="native_matmul"):
        heuristics._persistent_reduction_configs(
            {"x": 1, "y": 2, "r0_": 128},
            False,
            {},
            {"native_matmul": native_matmul},
        )
    assert calls == []


def test_adapter_rejects_unknown_call_arguments_and_metadata():
    provider, heuristics, calls, _, _ = make_provider()
    inductor._adapt_registered_persistent_reduction(provider, heuristics)
    adapted = heuristics._persistent_reduction_configs
    with pytest.raises(TypeError, match="unexpected keyword"):
        adapted({}, new_metadata={})
    with pytest.raises(TypeError):
        adapted({}, False, {}, {}, "extra")
    with pytest.raises(TypeError, match="mapping"):
        adapted({}, False, {}, ["native_matmul"])
    assert calls == []


def test_compatible_provider_is_preserved_and_receives_new_metadata():
    received = []
    result = object()

    def compatible(
        self, size_hints, reduction_hint=False, inductor_meta=None, triton_meta=None
    ):
        received.append((size_hints, reduction_hint, inductor_meta, triton_meta))
        return result

    provider, heuristics, _, _, _ = make_provider(compatible)
    original = heuristics._persistent_reduction_configs
    assert (
        inductor._adapt_registered_persistent_reduction(provider, heuristics)
        == "unchanged_compatible_gcu"
    )
    assert heuristics._persistent_reduction_configs is original
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    assert heuristics._persistent_reduction_configs is original
    provider.GCUInductorPatcher().patch_triton_heuristics()
    current = heuristics._persistent_reduction_configs
    assert current.__func__ is original.__func__
    assert current.__self__ is original.__self__
    metadata = {"native_matmul": True}
    assert current({}, False, {}, metadata) is result
    assert received[0][3] is metadata


def test_other_provider_is_preserved():
    provider, heuristics, _, _, _ = make_provider()
    other = Mock()
    heuristics._persistent_reduction_configs = other
    assert (
        inductor._adapt_registered_persistent_reduction(provider, heuristics)
        == "unchanged_other_provider"
    )
    assert heuristics._persistent_reduction_configs is other


@pytest.mark.parametrize(
    "function",
    [
        lambda self, sizes, reduction_hint=False, inductor_meta=None: None,
        lambda self, size_hints, reduction_hint=False, inductor_meta=None, extra=None: (
            None
        ),
        lambda self, size_hints, reduction_hint=True, inductor_meta=None: None,
        lambda self, size_hints, reduction_hint=False, inductor_meta=None, **kwargs: (
            None
        ),
    ],
)
def test_unknown_gcu_abi_is_rejected_without_replacing_slot(function):
    provider, heuristics, _, _, _ = make_provider(function)
    original = heuristics._persistent_reduction_configs
    with pytest.raises(RuntimeError, match="Unknown TorchGCU"):
        inductor._adapt_registered_persistent_reduction(provider, heuristics)
    assert heuristics._persistent_reduction_configs is original


def test_lazy_registration_then_reregistration_always_installs_adapter():
    provider, heuristics, calls, result, registrations = make_provider()
    untouched = Mock()
    heuristics._persistent_reduction_configs = untouched
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    wrapped = provider.GCUInductorPatcher.patch_triton_heuristics
    assert heuristics._persistent_reduction_configs is untouched
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    assert provider.GCUInductorPatcher.patch_triton_heuristics is wrapped
    patcher = provider.GCUInductorPatcher()
    for marker in ("first_compile", "second_registration"):
        assert patcher.patch_triton_heuristics(marker=marker) is result
        adapted = heuristics._persistent_reduction_configs
        assert getattr(adapted, inductor._ADAPTER_MARKER) is True
        assert adapted({}, False, {}, {}) is result
        inductor._install_gcu_inductor_hooks(provider, heuristics)
        assert heuristics._persistent_reduction_configs is adapted
    assert registrations == ["first_compile", "second_registration"]
    assert len(calls) == 2


def test_registration_error_propagates():
    provider, heuristics, _, _, _ = make_provider()

    def broken(self):
        raise ValueError("vendor registration failed")

    provider.GCUInductorPatcher.patch_triton_heuristics = broken
    inductor._install_gcu_inductor_hooks(provider, heuristics)
    with pytest.raises(ValueError, match="vendor registration failed"):
        provider.GCUInductorPatcher().patch_triton_heuristics()


def test_both_loaded_gcu_namespaces_keep_registered_and_lazy_abis(monkeypatch):
    canonical, heuristics, calls, result, _ = make_provider()
    alias, _, alias_calls, alias_result, _ = make_provider(shared_heuristics=heuristics)
    canonical.__file__ = alias.__file__ = __file__
    monkeypatch.setitem(sys.modules, "torch.gcu.inductor.backend_register", alias)
    # Reproduce an already-registered slot whose class belongs to the other
    # loaded module, followed by lazy registrations through either namespace.
    alias.GCUInductorPatcher().patch_triton_heuristics()
    for provider in inductor._loaded_gcu_inductor_providers(canonical):
        inductor._install_gcu_inductor_hooks(provider, heuristics)
    size_hints, reduction_hint, metadata = {}, object(), {}
    for provider, expected, observed in (
        (None, alias_result, alias_calls),
        (canonical, result, calls),
        (alias, alias_result, alias_calls),
    ):
        if provider is not None:
            provider.GCUInductorPatcher().patch_triton_heuristics()
        slot = heuristics._persistent_reduction_configs
        assert tuple(inspect.signature(slot).parameters) == inductor._PARAMETERS
        assert slot(size_hints, reduction_hint, metadata, {}) is expected
        assert all(
            actual is original
            for actual, original in zip(
                observed[-1], (size_hints, reduction_hint, metadata)
            )
        )
        for candidate in inductor._loaded_gcu_inductor_providers(canonical):
            inductor._install_gcu_inductor_hooks(candidate, heuristics)
        assert heuristics._persistent_reduction_configs is slot


@pytest.mark.parametrize("alias_source", [None, "missing-provider.py"])
def test_alias_with_unverified_source_is_rejected(monkeypatch, alias_source):
    provider, _, _, _, _ = make_provider()
    provider.__file__ = __file__
    alias, _, _, _, _ = make_provider()
    alias.__file__ = alias_source
    monkeypatch.setitem(sys.modules, "torch.gcu.inductor.backend_register", alias)
    with pytest.raises(RuntimeError, match="different sources"):
        inductor._loaded_gcu_inductor_providers(provider)


def test_alias_from_another_existing_file_is_rejected(monkeypatch, tmp_path):
    provider, _, _, _, _ = make_provider()
    provider.__file__ = __file__
    other = tmp_path / "backend_register.py"
    other.write_text("# unrelated provider\n")
    alias, _, _, _, _ = make_provider()
    alias.__file__ = str(other)
    monkeypatch.setitem(sys.modules, "torch.gcu.inductor.backend_register", alias)
    with pytest.raises(RuntimeError, match="different sources"):
        inductor._loaded_gcu_inductor_providers(provider)


def test_identical_module_alias_is_only_installed_once(monkeypatch):
    provider, _, _, _, _ = make_provider()
    monkeypatch.setitem(sys.modules, "torch.gcu.inductor.backend_register", provider)
    assert inductor._loaded_gcu_inductor_providers(provider) == [provider]


def test_non_enflame_entry_point_does_not_import_or_modify_provider(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "vllm.platforms",
        SimpleNamespace(current_platform=SimpleNamespace(vendor_name="nvidia")),
    )
    install = Mock()
    monkeypatch.setattr(inductor, "_install_gcu_inductor_hooks", install)
    inductor.apply_inductor_gcu_patch()
    install.assert_not_called()
