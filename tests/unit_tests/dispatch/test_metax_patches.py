# Copyright (c) 2026 BAAI. All rights reserved.

"""Guards against MetaX patches that silently fail to apply.

``PlatformFL.import_kernels()`` loads the patch package while vLLM is still
initialising, which used to let one circular import skip every patch after it.
Model-layer patches are deferred to
``PlatformFL.apply_config_platform_defaults()`` before any model layer is built.
"""

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "metax", reason="MetaX-only patch package"
)


def test_every_patch_applies_once_vllm_is_loaded():
    from vllm_fl.dispatch.backends.vendor.metax import patches

    # Install deferred model-layer patches and retry any earlier failures.
    assert patches.ensure_patches() == {}, (
        f"patches failed to import: {sorted(patches.PATCH_ERRORS)}"
    )
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    from vllm_fl.dispatch.backends.vendor.metax.patches import pynccl_wrapper

    installed = CudaCommunicator.all_reduce
    native = pynccl_wrapper._native_all_reduce
    assert patches.ensure_patches() == {}
    assert CudaCommunicator.all_reduce is installed
    assert pynccl_wrapper._native_all_reduce is native
    assert native is not installed


@pytest.mark.parametrize("compiling", [False, True])
def test_all_reduce_selects_path_before_execution_and_does_not_retry(
    monkeypatch, compiling
):
    from vllm.distributed.device_communicators.cuda_communicator import (
        CudaCommunicator,
    )

    from vllm_fl.dispatch.backends.vendor.metax.patches import pynccl_wrapper

    calls = []
    fail = False
    group = object()

    def native(self, value):
        assert self is communicator
        calls.append("native")
        if fail:
            raise RuntimeError("collective execution failed")
        return value * 2

    def compiled(value, op, group):
        assert group is communicator.device_group
        assert op == pynccl_wrapper.dist.ReduceOp.SUM
        calls.append("compiled")
        if fail:
            raise RuntimeError("collective execution failed")
        value.mul_(2)

    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: compiling)
    monkeypatch.setattr(pynccl_wrapper, "_native_all_reduce", native)
    monkeypatch.setattr(pynccl_wrapper.dist, "all_reduce", compiled)
    communicator = SimpleNamespace(device_group=group)
    value = torch.tensor([2.0, 3.0])
    output = CudaCommunicator.all_reduce(communicator, value)
    torch.testing.assert_close(output, torch.tensor([4.0, 6.0]))
    torch.testing.assert_close(value, torch.tensor([2.0, 3.0]))
    assert output.data_ptr() != value.data_ptr()
    expected = "compiled" if compiling else "native"
    assert calls == [expected]
    fail = True
    with pytest.raises(RuntimeError, match="collective execution failed"):
        CudaCommunicator.all_reduce(communicator, value)
    assert calls == [expected, expected]


def test_triton_kernels_disabled_for_metax():
    import vllm.utils.import_utils as iu

    from vllm_fl.dispatch.backends.vendor.metax import patches  # noqa: F401

    # vLLM modules bind this predicate when they load, so it has to be
    # replaced before any patch import runs.
    assert iu.has_triton_kernels() is False


@pytest.mark.parametrize(
    "first_import",
    ["vllm.model_executor.layers.linear", "vllm._custom_ops"],
)
def test_gdn_patch_is_deferred_until_platform_initialization(first_import):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            """
import importlib
import sys
from types import SimpleNamespace

importlib.import_module(sys.argv[1])
from vllm_fl.dispatch.backends.vendor.metax import patches

gdn_module = patches.__name__ + ".gdn_linear_attn"
assert gdn_module not in sys.modules
assert "gdn_linear_attn" not in patches.PATCH_ERRORS

from vllm.platforms import current_platform

assert current_platform.vendor_name == "metax"
config = SimpleNamespace(compilation_config=None)
current_platform.apply_config_platform_defaults(config)
assert not patches.PATCH_ERRORS, patches.PATCH_ERRORS

from vllm.model_executor.custom_op import op_registry_oot

gdn_class = sys.modules[gdn_module].MacaChunkGatedDeltaRule
assert op_registry_oot["ChunkGatedDeltaRule"] is gdn_class
current_platform.apply_config_platform_defaults(config)
assert not patches.PATCH_ERRORS, patches.PATCH_ERRORS
assert op_registry_oot["ChunkGatedDeltaRule"] is gdn_class
""",
            first_import,
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "metax patch gdn_linear_attn failed to apply" not in result.stderr


def test_deferred_patch_failure_is_reported_and_can_be_retried(monkeypatch):
    from vllm_fl.dispatch.backends.vendor.metax import patches

    assert patches.ensure_patches() == {}
    original_import = patches.importlib.import_module
    gdn_module = patches.__name__ + ".gdn_linear_attn"
    failure = ImportError("missing GDN dependency")
    attempts = []

    def fail_gdn(name, *args, **kwargs):
        if name == gdn_module:
            attempts.append(name)
            raise failure
        return original_import(name, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setattr(patches.importlib, "import_module", fail_gdn)
        errors = patches.ensure_patches()
        assert errors["gdn_linear_attn"] is failure
        assert patches.PATCH_ERRORS["gdn_linear_attn"] is failure
        assert attempts == [gdn_module]
        assert patches.ensure_patches()["gdn_linear_attn"] is failure
        assert attempts == [gdn_module, gdn_module]

    assert patches.ensure_patches() == {}
    assert "gdn_linear_attn" not in patches.PATCH_ERRORS
