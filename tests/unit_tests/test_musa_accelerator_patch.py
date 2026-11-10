# Copyright (c) 2026 BAAI. All rights reserved.

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def musa_patch():
    # Load only the patch source: importing vllm_fl also initializes the
    # dispatch backends, which is unnecessary for this compatibility contract.
    source = (
        Path(__file__).resolve().parents[2]
        / "vllm_fl/dispatch/backends/vendor/musa/patch.py"
    )
    spec = importlib.util.spec_from_file_location("musa_accelerator_patch", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def vendor_modules():
    def unsupported(*args, **kwargs):
        raise RuntimeError("DeviceAllocator INTERNAL ASSERT FAILED")

    torch = ModuleType("torch")
    torch.accelerator = SimpleNamespace(
        empty_cache=unsupported,
        max_memory_allocated=unsupported,
        memory_stats=unsupported,
        memory_reserved=unsupported,
        reset_peak_memory_stats=unsupported,
        synchronize=unsupported,
    )
    musa = ModuleType("torch_musa")
    musa.empty_cache = Mock(return_value=None)
    musa.memory_allocated = Mock(return_value=64)
    musa.max_memory_allocated = Mock(return_value=128)
    musa.memory_stats = Mock(return_value={"allocated_bytes.all.peak": 128})
    musa.memory_reserved = Mock(return_value=256)
    musa.max_memory_reserved = Mock(return_value=320)
    musa.reset_peak_memory_stats = Mock(return_value=None)
    musa.mem_get_info = Mock(return_value=(512, 1024))
    musa.device = Mock(return_value=object())
    musa.synchronize = Mock(return_value=None)
    return torch, musa


def _apply_patch(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    # Replace imports only while calling the patch, restoring any real Torch
    # modules before the test returns control to the rest of the suite.
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", torch)
        imports.setitem(sys.modules, "torch_musa", musa)
        musa_patch.patch_accelerator_missing_attrs()


@pytest.mark.parametrize(
    ("accelerator_name", "vendor_name"),
    [
        ("max_memory_allocated", "max_memory_allocated"),
        ("memory_stats", "memory_stats"),
        ("memory_reserved", "memory_reserved"),
        ("reset_peak_memory_stats", "reset_peak_memory_stats"),
        ("get_memory_info", "mem_get_info"),
    ],
)
@pytest.mark.parametrize("device", [None, 1, "musa:1", object()])
def test_memory_api_forwards_explicit_device_and_result(
    monkeypatch, musa_patch, vendor_modules, accelerator_name, vendor_name, device
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    accelerator_api = getattr(torch.accelerator, accelerator_name)
    vendor_api = getattr(musa, vendor_name)
    assert accelerator_api is vendor_api
    assert accelerator_api(device=device) is vendor_api.return_value
    vendor_api.assert_called_once_with(device=device)


def test_memory_profiling_uses_current_device_when_argument_is_omitted(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.memory_stats() == {"allocated_bytes.all.peak": 128}
    assert torch.accelerator.memory_reserved() == 256
    assert torch.accelerator.get_memory_info() == (512, 1024)
    assert torch.accelerator.reset_peak_memory_stats() is None
    for name in (
        "memory_stats",
        "memory_reserved",
        "mem_get_info",
        "reset_peak_memory_stats",
    ):
        getattr(musa, name).assert_called_once_with()


@pytest.mark.parametrize("existing_device_index", [False, True])
def test_existing_empty_cache_and_device_context_behavior_is_preserved(
    monkeypatch, musa_patch, vendor_modules, existing_device_index
):
    torch, musa = vendor_modules
    if existing_device_index:
        torch.accelerator.device_index = Mock(return_value=object())
        expected_context = torch.accelerator.device_index
    else:
        expected_context = musa.device
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.empty_cache() is None
    musa.empty_cache.assert_called_once_with()
    assert torch.accelerator.device_index is expected_context
    assert torch.accelerator.device_index(1) is expected_context.return_value
    expected_context.assert_called_once_with(1)


def test_repeated_patch_is_idempotent(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)
    patched = vars(torch.accelerator).copy()

    # A second application must leave the first bindings and device context
    # intact even if another component changes torch_musa's attributes.
    musa.mem_get_info = Mock()
    musa.empty_cache = Mock()
    musa.device = Mock()
    musa.synchronize = Mock()
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert vars(torch.accelerator) == patched
    assert torch.accelerator._musa_attrs_patched is True


def test_vendor_memory_errors_are_not_hidden(monkeypatch, musa_patch, vendor_modules):
    torch, musa = vendor_modules
    musa.mem_get_info.side_effect = RuntimeError("MUSA device is unavailable")
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    with pytest.raises(RuntimeError, match="MUSA device is unavailable"):
        torch.accelerator.get_memory_info(device=2)
    musa.mem_get_info.assert_called_once_with(device=2)


@pytest.mark.parametrize(
    "vendor_name",
    [
        "empty_cache",
        "max_memory_allocated",
        "memory_stats",
        "memory_reserved",
        "reset_peak_memory_stats",
        "mem_get_info",
        "synchronize",
        "device",
    ],
)
@pytest.mark.parametrize("invalid_api", ["missing", "not_callable"])
def test_invalid_vendor_api_leaves_original_bindings_and_allows_retry(
    monkeypatch, musa_patch, vendor_modules, caplog, vendor_name, invalid_api
):
    torch, musa = vendor_modules
    original = vars(torch.accelerator).copy()
    vendor_api = getattr(musa, vendor_name)
    if invalid_api == "missing":
        delattr(musa, vendor_name)
    else:
        setattr(musa, vendor_name, None)

    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert vars(torch.accelerator) == original
    assert not getattr(torch.accelerator, "_musa_attrs_patched", False)
    assert "Failed to patch torch.accelerator attrs for MUSA" in caplog.text

    setattr(musa, vendor_name, vendor_api)
    _apply_patch(monkeypatch, musa_patch, vendor_modules)
    assert torch.accelerator._musa_attrs_patched is True
    assert torch.accelerator.get_memory_info is musa.mem_get_info
    assert torch.accelerator.device_index is musa.device


def test_patch_entry_point_retries_after_incomplete_vendor_api(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    original = vars(torch.accelerator).copy()
    memory_info = musa.mem_get_info
    del musa.mem_get_info
    other_patches = [
        "patch_topk_topp_sampler",
        "patch_cuda_get_device_properties",
        "patch_cuda_stream_for_musa",
        "patch_inductor_triton_for_musa",
        "patch_moe_topk_softmax_for_musa",
        "patch_triton_mtgpu_alias_for_musa",
        "patch_device_config_for_musa",
        "patch_musa_vision_rotary_layout",
        # These import flag_gems, so running them under the fake ``torch``
        # module below would cache a poisoned flag_gems in sys.modules.
        "patch_flag_gems_capture_safe_tuning",
        "patch_flag_gems_disable_flagtune",
    ]
    for name in other_patches:
        monkeypatch.setattr(musa_patch, name, Mock())

    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "torch", torch)
        imports.setitem(sys.modules, "torch_musa", musa)
        musa_patch.apply_musa_patches()
        assert musa_patch._patches_applied is False
        assert vars(torch.accelerator) == original

        musa.mem_get_info = memory_info
        musa_patch.apply_musa_patches()
        assert musa_patch._patches_applied is True
        assert torch.accelerator._musa_attrs_patched is True
        assert torch.accelerator.get_memory_info is memory_info

        calls = {name: getattr(musa_patch, name).call_count for name in other_patches}
        musa_patch.apply_musa_patches()
        assert calls == {
            name: getattr(musa_patch, name).call_count for name in other_patches
        }


def test_synchronize_uses_vendor_current_device_when_argument_is_omitted(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.synchronize is musa.synchronize
    assert torch.accelerator.synchronize() is None
    musa.synchronize.assert_called_once_with()


@pytest.mark.parametrize("device", [None, 1, "musa:1", object()])
def test_synchronize_forwards_explicit_device_unchanged(
    monkeypatch, musa_patch, vendor_modules, device
):
    torch, musa = vendor_modules
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    assert torch.accelerator.synchronize(device) is None
    musa.synchronize.assert_called_once_with(device)


def test_synchronize_vendor_errors_are_not_hidden(
    monkeypatch, musa_patch, vendor_modules
):
    torch, musa = vendor_modules
    musa.synchronize.side_effect = RuntimeError("MUSA device is unavailable")
    _apply_patch(monkeypatch, musa_patch, vendor_modules)

    with pytest.raises(RuntimeError, match="MUSA device is unavailable"):
        torch.accelerator.synchronize(2)
    musa.synchronize.assert_called_once_with(2)


def _fake_torch_with_device(*, accepts_bare_musa):
    """Fake ``torch`` whose ``device()`` mirrors an old torch_musa build.

    Those builds raise ``RuntimeError: Device string must not be empty`` for the
    bare ``"musa"`` string. Every accepted call records its arguments so the
    tests can assert which device object vLLM ends up with.
    """

    def device(*args):
        if args == ("musa",) and not accepts_bare_musa:
            raise RuntimeError("Device string must not be empty")
        return ("device",) + args

    torch_module = ModuleType("torch")
    torch_module.device = device
    return torch_module


def _fake_device_config_class(torch_module):
    """Minimal stand-in for the vllm 0.32 ``DeviceConfig`` post-init contract."""

    class DeviceConfig:
        def __init__(self, device="auto"):
            self.device = device
            self.device_type = None
            self.__post_init__()

        def __post_init__(self):
            if self.device == "auto":
                from vllm.platforms import current_platform

                self.device_type = current_platform.device_type
            elif isinstance(self.device, str):
                self.device_type = self.device.split(":")[0]
            # 0.32 rebuilds the device from the bare type, which is exactly the
            # call that fails on the older torch_musa build.
            self.device = torch_module.device(self.device_type)

    return DeviceConfig


def _install_device_config_modules(
    monkeypatch, torch_module, device_config_cls, device_type="musa"
):
    vllm = ModuleType("vllm")
    vllm_config = ModuleType("vllm.config")
    vllm_device = ModuleType("vllm.config.device")
    vllm_device.DeviceConfig = device_config_cls
    vllm_config.device = vllm_device
    vllm.config = vllm_config
    vllm_platforms = ModuleType("vllm.platforms")
    vllm_platforms.current_platform = SimpleNamespace(device_type=device_type)
    vllm.platforms = vllm_platforms

    for name, module in {
        "torch": torch_module,
        "torch_musa": ModuleType("torch_musa"),
        "vllm": vllm,
        "vllm.config": vllm_config,
        "vllm.config.device": vllm_device,
        "vllm.platforms": vllm_platforms,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)


@pytest.mark.parametrize("raw_device", ["auto", "musa"])
def test_device_config_patch_keeps_the_musa_device_unindexed(
    monkeypatch, musa_patch, raw_device
):
    torch_module = _fake_torch_with_device(accepts_bare_musa=False)
    device_config_cls = _fake_device_config_class(torch_module)
    _install_device_config_modules(monkeypatch, torch_module, device_config_cls)

    musa_patch.patch_device_config_for_musa()

    config = device_config_cls(raw_device)
    assert config.device_type == "musa"
    # Index-free: a hard-coded ``musa:0`` would bind every TP rank to card 0.
    assert config.device == ("device", "musa", None)


def test_device_config_patch_is_skipped_when_bare_musa_is_supported(
    monkeypatch, musa_patch
):
    torch_module = _fake_torch_with_device(accepts_bare_musa=True)
    device_config_cls = _fake_device_config_class(torch_module)
    _install_device_config_modules(monkeypatch, torch_module, device_config_cls)
    original = device_config_cls.__post_init__

    musa_patch.patch_device_config_for_musa()

    assert device_config_cls.__post_init__ is original
    assert not getattr(device_config_cls, "_musa_post_init_patched", False)
    assert device_config_cls("auto").device == ("device", "musa")


def test_device_config_patch_leaves_other_platforms_untouched(monkeypatch, musa_patch):
    torch_module = _fake_torch_with_device(accepts_bare_musa=False)
    device_config_cls = _fake_device_config_class(torch_module)
    _install_device_config_modules(
        monkeypatch, torch_module, device_config_cls, device_type="cuda"
    )

    musa_patch.patch_device_config_for_musa()

    config = device_config_cls("auto")
    assert config.device_type == "cuda"
    assert config.device == ("device", "cuda")


@pytest.fixture
def parent_registration(monkeypatch, musa_patch, vendor_modules):
    import ast
    import os

    source = Path(__file__).resolve().parents[2] / "vllm_fl/__init__.py"
    parsed = ast.parse(source.read_text(encoding="utf-8"))
    functions = [
        node
        for node in parsed.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_init_vendor_device", "register"}
    ]
    namespace = {
        "os": os,
        # The extracted hooks still call the unrelated MetaX installer.
        "_install_metax_compat": Mock(),
        "_arm_cpu_platform": lambda: None,
        "_patch_custom_ops": Mock(),
        "_patch_flash_attn_import": Mock(),
        "_get_op_config": Mock(),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"),
        namespace,
    )
    torch, musa = vendor_modules
    utils = ModuleType("vllm_fl.utils")
    utils.DeviceInfo = Mock(return_value=SimpleNamespace(vendor_name="mthreads"))
    kunlunxin = ModuleType(
        "vllm_fl.dispatch.backends.vendor.kunlunxin.patches.patch_fla_utils"
    )
    kunlunxin._patch_xpu_get_device = Mock()
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, "vllm_fl.utils", utils)
        imports.setitem(
            sys.modules, "vllm_fl.dispatch.backends.vendor.musa.patch", musa_patch
        )
        imports.setitem(sys.modules, kunlunxin.__name__, kunlunxin)
        imports.setitem(sys.modules, "torch", torch)
        imports.setitem(sys.modules, "torch_musa", musa)
        imports.setattr(musa_patch, "apply_musa_patches", Mock())
        imports.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        imports.setenv("VLLM_USE_V2_MODEL_RUNNER", "0")
        yield namespace, utils, kunlunxin
        musa_patch.apply_musa_patches.assert_not_called()


def test_parent_cleanup_preserves_original_startup_error(
    parent_registration, vendor_modules
):
    namespace, _, _ = parent_registration
    torch, musa = vendor_modules
    startup_error = RuntimeError("engine startup failed before model construction")
    namespace["_patch_custom_ops"].side_effect = startup_error

    # Registration can fail before model initialization; its allocator bridge
    # must already be present so cleanup cannot replace the original error.
    with pytest.raises(RuntimeError) as caught:
        try:
            namespace["register"]()
        finally:
            torch.accelerator.empty_cache()
    assert caught.value is startup_error
    musa.empty_cache.assert_called_once_with()
    assert torch.accelerator.get_memory_info is musa.mem_get_info
    namespace["_patch_flash_attn_import"].assert_not_called()


def test_parent_registration_allocator_binding_is_idempotent(
    parent_registration, vendor_modules
):
    namespace, _, _ = parent_registration
    torch, musa = vendor_modules
    assert namespace["register"]() == "vllm_fl.platform.PlatformFL"
    bindings = vars(torch.accelerator).copy()
    musa.empty_cache = Mock()
    musa.mem_get_info = Mock()
    assert namespace["register"]() == "vllm_fl.platform.PlatformFL"
    assert vars(torch.accelerator) == bindings


@pytest.mark.parametrize("invalid_api", ["missing", "not_callable"])
def test_parent_registration_rejects_incomplete_allocator_bridge(
    parent_registration, vendor_modules, invalid_api
):
    namespace, _, _ = parent_registration
    torch, musa = vendor_modules
    original = vars(torch.accelerator).copy()
    if invalid_api == "missing":
        del musa.mem_get_info
    else:
        musa.mem_get_info = None
    with pytest.raises(
        RuntimeError, match="initialize MUSA accelerator allocator APIs"
    ):
        namespace["register"]()
    assert vars(torch.accelerator) == original
    namespace["_patch_custom_ops"].assert_not_called()


@pytest.mark.parametrize("vendor", ["nvidia", "ascend", "kunlunxin", "metax"])
def test_parent_registration_keeps_other_vendor_allocator_unchanged(
    parent_registration, vendor_modules, vendor
):
    namespace, utils, kunlunxin = parent_registration
    torch, musa = vendor_modules
    utils.DeviceInfo.return_value.vendor_name = vendor
    original = vars(torch.accelerator).copy()
    result = namespace["register"]()
    namespace["_install_metax_compat"].assert_called_once_with()
    assert result == (
        "vllm_fl.nvidia_platform.NvidiaPlatformFL"
        if vendor == "nvidia"
        else "vllm_fl.platform.PlatformFL"
    )
    assert vars(torch.accelerator) == original
    musa.empty_cache.assert_not_called()
    assert kunlunxin._patch_xpu_get_device.call_count == (vendor == "kunlunxin")
