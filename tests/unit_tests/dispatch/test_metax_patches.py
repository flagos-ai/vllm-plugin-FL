# Copyright (c) 2026 BAAI. All rights reserved.

"""Guards against MetaX patches that silently fail to apply.

``PlatformFL.import_kernels()`` loads the patch package while vLLM is still
initialising, which used to let one circular import skip every patch after it.
The package now retries the leftovers from
``PlatformFL.apply_config_platform_defaults()`` before any model layer is built.
"""

import pytest

from tests.utils.device_utils import get_backend

pytestmark = pytest.mark.skipif(
    get_backend() != "metax", reason="MetaX-only patch package"
)


def test_every_patch_applies_once_vllm_is_loaded():
    from vllm_fl.dispatch.backends.vendor.metax import patches

    # gdn_linear_attn imports vLLM model-layer modules, so it fails when this
    # package is loaded from inside vllm's own import chain. The platform hook
    # retries it before model construction; assert the retry leaves nothing.
    assert patches.ensure_patches() == {}, (
        f"patches failed to import: {sorted(patches.PATCH_ERRORS)}"
    )


def test_triton_kernels_disabled_for_metax():
    import vllm.utils.import_utils as iu

    from vllm_fl.dispatch.backends.vendor.metax import patches  # noqa: F401

    # vLLM modules bind this predicate when they load, so it has to be
    # replaced before any patch import runs.
    assert iu.has_triton_kernels() is False


def test_triton_cache_writer_installation_is_idempotent():
    import importlib

    from vllm.v1.attention.backends import triton_attn

    from vllm_fl.dispatch.backends.vendor.metax.patches import triton_kv_cache

    expected = triton_kv_cache.triton_reshape_and_cache_flash
    assert triton_attn.triton_reshape_and_cache_flash is expected
    importlib.reload(triton_kv_cache)
    assert triton_attn.triton_reshape_and_cache_flash is expected


def test_triton_cache_writer_rejects_signature_change(monkeypatch):
    import importlib

    from vllm.v1.attention.backends import triton_attn

    from vllm_fl.dispatch.backends.vendor.metax.patches import triton_kv_cache

    def incompatible_writer(key):
        raise AssertionError("The incompatible writer must not execute")

    with monkeypatch.context() as patch:
        patch.setattr(
            triton_attn, "triton_reshape_and_cache_flash", incompatible_writer
        )
        with pytest.raises(
            RuntimeError, match="Unexpected vLLM Triton KV cache writer"
        ):
            importlib.reload(triton_kv_cache)
        assert triton_attn.triton_reshape_and_cache_flash is incompatible_writer
    importlib.reload(triton_kv_cache)
