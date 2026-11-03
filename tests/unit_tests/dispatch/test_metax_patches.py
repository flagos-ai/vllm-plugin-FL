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
