# SPDX-License-Identifier: Apache-2.0

"""The LPU facts that live in ``vllm_fl/platform.py`` as divergences.

``platform.py`` is shared with upstream, and the LPU behaviours below sit in it as
edited lines rather than as ``PatchSpec``s: nothing in the patch layer and no
other test would notice a merge resolution that drops them, because the file
still imports and the dropped behaviour degrades silently (no exception, a
different capability answer).  These assertions pin the production values, so a
lost line fails here instead of at run time.
"""

import ast
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from vllm.platforms.interface import DeviceCapability

import vllm_fl
from vllm_fl.platform import PlatformFL, dist_backend_dict


def test_dist_backend_dict_maps_lpu_to_zccl():
    """Upstream has no ``"lpu"`` entry; a missing one does not raise.

    The lookup in ``dist_backend`` is ``dist_backend_dict.get(device_name,
    "nccl")``, so dropping the entry silently makes LPU distributed run on
    nccl -- which in turn flips the communicator class and custom allreduce.
    """
    assert dist_backend_dict.get("lpu") == "zccl"


def test_pin_memory_available_includes_lpu():
    """Upstream's allowlist dropped ``lpu``; pinning is what LPU needs here."""
    with patch.object(PlatformFL, "device_type", "lpu"):
        assert PlatformFL.is_pin_memory_available()


def test_supports_fp8_for_evas_nvidia_and_unmapped_vendors():
    """``evas`` answers yes; ``nvidia`` keeps its gate; an unmapped vendor is no.

    Called on the class rather than an instance: the body reads ``cls``.  The
    ``nvidia`` arms feed the capability probe instead of reading the host, so
    they answer the same on a GPU runner: below the sm89 floor the answer is
    "no FP8", and a probe that cannot answer at all must degrade rather than
    escape the method.
    """
    with patch.object(PlatformFL, "vendor_name", "evas"):
        assert PlatformFL.supports_fp8()

    with patch.object(PlatformFL, "vendor_name", "kunlunxin"):
        assert not PlatformFL.supports_fp8()

    def _nvidia(**probe):
        stack = ExitStack()
        stack.enter_context(patch.object(PlatformFL, "vendor_name", "nvidia"))
        stack.enter_context(patch.object(PlatformFL, "get_device_capability", **probe))
        return stack

    with _nvidia(return_value=DeviceCapability(9, 0)):
        assert PlatformFL.supports_fp8()

    # Below the floor, and the three exceptions upstream swallows there.
    with _nvidia(return_value=DeviceCapability(8, 0)):
        assert not PlatformFL.supports_fp8()
    for exc in (
        RuntimeError("NVML unavailable"),
        AttributeError("no device capability"),
        NotImplementedError("not on this build"),
    ):
        with _nvidia(side_effect=exc):
            assert not PlatformFL.supports_fp8(), (
                f"{type(exc).__name__} from the capability probe must answer "
                '"no FP8", as upstream does'
            )


def test_use_custom_allreduce_is_off_for_evas():
    """The gate is the device, so no backend name can re-enable the collective.

    The form that excluded the names ``flagcx``/``zccl`` instead re-enables a
    collective LPU has no implementation for the moment its backend is renamed,
    and it has to edit a line upstream owns.
    """
    for backend in ("flagcx", "nccl", "zccl"):
        with (
            patch.object(PlatformFL, "vendor_name", "evas"),
            patch.object(PlatformFL, "dist_backend", backend),
        ):
            assert not PlatformFL.use_custom_allreduce()

    with (  # the upstream behaviour for other vendors still holds
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.object(PlatformFL, "dist_backend", "nccl"),
    ):
        assert PlatformFL.use_custom_allreduce()
    with (
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.object(PlatformFL, "dist_backend", "flagcx"),
    ):
        assert not PlatformFL.use_custom_allreduce()


def test_lpu_compile_backend_stays_eager():
    """The ``eager`` pin for LPU, asserted on the source AST.

    It is a local variable set inside ``PlatformFL.__init__`` next to a device
    probe, so there is no way to reach it without standing up a whole platform
    object.  The hazard being guarded is a merge resolution that keeps
    upstream's assignment and loses our ``lpu`` case, so the assertion reads
    the value expression: it must be a conditional whose ``"lpu"`` branch
    yields ``"eager"``.  Matching text instead would break every time upstream
    reformats the line (v0.3.0 already turned our ``if`` into a ternary).
    """
    tree = ast.parse(
        (Path(vllm_fl.__file__).resolve().parent / "platform.py").read_text(
            encoding="utf-8"
        )
    )
    assigns = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Name) and t.id == "simple_compile_backend"
            for t in node.targets
        )
    ]
    assert assigns, "simple_compile_backend is no longer assigned in platform.py"
    for node in assigns:
        value = node.value
        assert isinstance(value, ast.IfExp), (
            "simple_compile_backend must be a device_type conditional"
        )
        tested = {
            c.value
            for c in ast.walk(value.test)
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
        }
        assert "lpu" in tested, "the LPU branch disappeared from simple_compile_backend"
        assert isinstance(value.body, ast.Constant) and value.body.value == "eager", (
            'LPU must keep simple_compile_backend = "eager": inductor against '
            "the FlagTree Triton fork crashes in the triton_key step without it"
        )


def test_get_device_communicator_cls_uses_communicatorfl_for_evas():
    """The vendor arm, separate from the ``flagcx`` arm and from allreduce.

    Upstream keys its first arm on the backend name and the vendor-specific
    communicators (thead, ascend) on the vendor, so evas joins the vendor arms:
    the answer must not depend on what our dist backend happens to be called.
    The three live on different lines and answer different questions, so a
    resolution can keep one and lose another; upstream has no evas arm, which
    would hand LPU the CUDA communicator.
    """
    communicator = "vllm_fl.distributed.communicator.CommunicatorFL"
    for backend in ("zccl", "nccl", "flagcx"):
        with (
            patch.object(PlatformFL, "vendor_name", "evas"),
            patch.object(PlatformFL, "dist_backend", backend),
        ):
            assert PlatformFL.get_device_communicator_cls() == communicator

    with (  # the upstream backend-keyed arm still holds for other vendors
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.object(PlatformFL, "dist_backend", "flagcx"),
    ):
        assert PlatformFL.get_device_communicator_cls() == communicator

    with (
        patch.object(PlatformFL, "vendor_name", "nvidia"),
        patch.object(PlatformFL, "dist_backend", "nccl"),
    ):
        assert PlatformFL.get_device_communicator_cls() != communicator


def test_get_device_capability_reports_the_lpu_value():
    """Upstream has no LPU branch, so the call falls through to ``torch.cuda``.

    ``torch.lpu`` is stubbed rather than imported: the assertion is about which
    accessor the branch consults, not about the hardware this box has.
    """
    fake_lpu = SimpleNamespace(get_device_capability=lambda device_id: (9, 0))
    with (
        patch.object(torch, "lpu", fake_lpu, create=True),
        patch.object(PlatformFL, "device_type", "lpu"),
    ):
        capability = PlatformFL.get_device_capability(0)

    assert (capability.major, capability.minor) == (9, 0)


def test_model_runner_carries_no_extra_slot_fill():
    """``model_runner.py`` must stay clean of the early slot fill we dropped.

    An older core filled ``slot_mappings_by_group`` early in ``_dummy_run``; a
    later upstream (#442, common-attention-metadata graphs) removed it and moved
    the fill to *after* the metadata producer has read those buffers.  This
    branch descends from the older core, so a refresh that re-imports the file
    wholesale brings the fill back and silently changes what the producer sees.
    Pinned as an absence, so a refresh has to argue for it here instead of
    reintroducing it.
    """
    runner = (
        Path(vllm_fl.__file__).resolve().parent / "worker" / "model_runner.py"
    ).read_text(encoding="utf-8")
    assert "sm.fill_(-1)" not in runner, (
        "the early dummy-run slot fill is back in model_runner.py; upstream "
        "removed it in #442, and other vendors would see different metadata"
    )


def test_lpu_cache_block_size_branch_stays_128():
    """The LPU block-size branch, asserted on the source.

    Reaching it needs a whole ``VllmConfig`` (the method keeps going into
    compilation config and worker wiring), so the assertion reads the branch
    the way the hazard appears: upstream has no LPU arm and the trailing
    ``else`` sets 16.
    """
    src = (Path(vllm_fl.__file__).resolve().parent / "platform.py").read_text(
        encoding="utf-8"
    )
    assert 'elif cls.device_type == "lpu":' in src, (
        "the LPU block-size branch disappeared from check_and_update_config"
    )
    lpu_branch = src.split('elif cls.device_type == "lpu":', 1)[1].split("else:")[0]
    assert "cache_config.block_size = 128" in lpu_branch, (
        "LPU must keep cache_config.block_size = 128; without the branch the "
        "trailing else silently gives it 16"
    )
