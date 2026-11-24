# Copyright 2025 BAAI. All rights reserved.

"""Real-PPU rows of the shared ``PlatformFL`` class.

The hygon supplementary suite pins ``support_static_graph_mode`` and the
all-reduce policy with monkeypatched vendor names. These tests run the same
rows *on the actual thead PPU container* — the platform class resolved from
a live ``GEMS_VENDOR=thead`` environment — so the assertions cover the
real introspection chain, not a stubbed vendor string:

* ``support_static_graph_mode()`` — thead is on the static-graph vendor list,
* ``use_custom_op_collectives()`` — thead keeps the registered collective
  routes,
* ``get_device_communicator_cls()`` — with ``FLAGCX_PATH`` set the container
  resolves ``dist_backend`` to ``flagcx`` and the thead (non-iluvatar,
  non-amd) arm picks ``CommunicatorFL``,
* ``get_device_uuid()`` — the pynvml arm on a CUDA-compatible device,
* capability / memory / compute-unit probes against the live device.

``vllm_fl.platform`` imports fine inside the container (the FlagGems device
detector resolves the vendor without touching the missing CUDA extension),
so no sys.modules surgery is needed here — unlike the attention modules.
"""

from __future__ import annotations

import os

import pytest
import torch

pytestmark = [pytest.mark.supplementary, pytest.mark.platforms("thead")]


@pytest.fixture(scope="module")
def platform_fl():
    """Import the resolved PlatformFL class from the live container env.

    These rows assert the *resolved* platform state (vendor, dist backend,
    UUIDs), so they only run where the environment genuinely resolves to
    thead — a CPU collection host (or a session started for another vendor)
    skips rather than asserting another vendor's resolution.
    """
    module = pytest.importorskip(
        "vllm_fl.platform", reason="vllm_fl.platform must import on the PPU host"
    )
    if module.PlatformFL.vendor_name != "thead":
        pytest.skip(
            f"requires a thead-resolved platform; got "
            f"{module.PlatformFL.vendor_name!r}"
        )
    return module.PlatformFL


class TestResolvedIdentity:
    """The platform class resolved with a live thead PPU present."""

    def test_vendor_and_device_type_resolved(self, platform_fl):
        assert platform_fl.vendor_name == "thead"
        assert platform_fl.device_type == "cuda"
        # cuda_alike (nvidia/metax): device_name = vendor_name; non-alike
        # vendors (thead included) use the device_type as device_name.
        assert platform_fl.device_name == "thead"

    def test_dist_backend_is_flagcx_when_flagcx_path_set(self, platform_fl):
        """The container exports FLAGCX_PATH, so the FlagCX arm must engage."""
        assert os.environ.get("FLAGCX_PATH"), (
            "the thead container is expected to export FLAGCX_PATH"
        )
        assert platform_fl.dist_backend == "flagcx"


class TestStaticGraphMode:
    """``support_static_graph_mode`` includes the thead PPU."""

    def test_thead_supports_static_graph_mode(self, platform_fl):
        assert platform_fl.support_static_graph_mode() is True


class TestCustomOpCollectives:
    """``use_custom_op_collectives`` keeps thead's registered routes."""

    def test_thead_uses_custom_op_collectives(self, platform_fl):
        assert platform_fl.use_custom_op_collectives() is True


class TestDeviceCommunicatorCls:
    """The flagcx + thead arm of ``get_device_communicator_cls``."""

    def test_flagcx_thead_routes_to_communicator_fl(self, platform_fl):
        assert platform_fl.dist_backend == "flagcx"
        assert platform_fl.device_type == "cuda"
        assert platform_fl.vendor_name == "thead"
        assert (
            platform_fl.get_device_communicator_cls()
            == "vllm_fl.distributed.communicator.CommunicatorFL"
        )


class TestCustomAllreduce:
    """FlagCX transport disables vLLM's custom all-reduce."""

    def test_flagcx_disables_custom_allreduce(self, platform_fl):
        assert platform_fl.use_custom_allreduce() is False


class TestDeviceUuid:
    """``get_device_uuid`` on the CUDA-compatible pynvml arm."""

    @pytest.mark.gpu
    def test_uuid_is_a_nonempty_string(self, platform_fl):
        if not torch.cuda.is_available():
            pytest.skip("PPU required")
        uuid = platform_fl.get_device_uuid(0)
        assert isinstance(uuid, str) and uuid

    @pytest.mark.gpu
    def test_uuids_across_devices_differ(self, platform_fl):
        count = torch.cuda.device_count()
        if count < 2:
            pytest.skip("multiple PPUs required")
        uuids = {platform_fl.get_device_uuid(i) for i in range(count)}
        assert len(uuids) == count, "each PPU must report a distinct UUID"


class TestLiveDeviceProperties:
    """Capability / memory / compute-unit probes against the live PPU."""

    @pytest.mark.gpu
    def test_device_capability_is_cc8(self, platform_fl):
        if not torch.cuda.is_available():
            pytest.skip("PPU required")
        capability = platform_fl.get_device_capability(0)
        major, minor = capability.major, capability.minor
        assert (major, minor) == (8, 0), (
            "zw810e is documented as compute capability 8.0 "
            f"(tests/platforms/thead.yaml); got {major}.{minor}"
        )

    @pytest.mark.gpu
    def test_total_memory_matches_zw810e_budget(self, platform_fl):
        if not torch.cuda.is_available():
            pytest.skip("PPU required")
        total_gib = platform_fl.get_device_total_memory(0) / 1024**3
        # The platform YAML documents 96 GiB; allow the driver-visible range.
        assert 60 <= total_gib <= 130, f"unexpected PPU memory budget: {total_gib}"

    @pytest.mark.gpu
    def test_num_compute_units_positive(self, platform_fl):
        if not torch.cuda.is_available():
            pytest.skip("PPU required")
        assert platform_fl.num_compute_units(0) > 0
