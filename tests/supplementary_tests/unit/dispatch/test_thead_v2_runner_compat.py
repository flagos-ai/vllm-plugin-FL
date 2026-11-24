# Copyright 2025 BAAI. All rights reserved.

"""V2 model-runner compatibility patches for the T-Head (PPU) backend.

``v2_runner_compat`` adapts the upstream CUDA-centric V2 runner for
``VLLM_TARGET_DEVICE=empty`` PPU builds: the UVA probe is only overridden
when the build cannot *map* pinned host memory into the device address
space, and ``cudagraph_utils.graph_capture`` is unconditionally rebound to
the device-agnostic FL context manager.

The probe helpers import from ``vllm.utils.platform_utils`` and
``vllm.utils.torch_utils`` *inside* the function, so tests monkeypatch the
modules those imports resolve from (mirroring
tests/unit_tests/worker/test_worker.py's probe pattern).

``vllm_fl.worker.model_runner`` is a heavy import on CPU builds
(``vllm_fl.compilation.graph`` raises NotImplementedError outside a
supported accelerator), so the install tests stub all three touched modules
in ``sys.modules`` (save / install / restore in ``finally``).
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.supplementary]

_GRAPH_CAPTURE_KEY = "vllm.v1.worker.gpu.cudagraph_utils.graph_capture"


@pytest.fixture(autouse=True)
def probe_recorder(monkeypatch):
    """Record every call to the accelerator view helper.

    The v2_runner_compat probe imports
    ``vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor`` inside
    ``mapped_host_buffers_available``, so patching the defining module
    attribute is enough — no sys.modules surgery needed.
    """
    import torch

    import vllm.utils.platform_utils as platform_utils
    import vllm.utils.torch_utils as torch_utils
    from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

    probed = []

    def recorder(tensor):
        probed.append(tensor)
        return tensor

    def failing_view(tensor):
        probed.append(tensor)
        raise RuntimeError("mapped view unavailable on this build")

    monkeypatch.setattr(torch, "zeros", lambda *a, **k: torch.empty(1))
    monkeypatch.setattr(
        torch_utils, "get_accelerator_view_from_cpu_tensor", recorder
    )
    monkeypatch.setattr(platform_utils, "is_uva_available", lambda: False)

    return types.SimpleNamespace(
        probed=probed,
        recorder=recorder,
        failing_view=failing_view,
        torch_utils=torch_utils,
        platform_utils=platform_utils,
        v2_runner_compat=v2_runner_compat,
        monkeypatch=monkeypatch,
    )


class TestMappedHostBuffersAvailable:
    """The pinned-memory mapping probe in mapped_host_buffers_available."""

    @pytest.mark.platforms("thead")
    def test_uva_unavailable_short_circuits_without_probing(
        self, probe_recorder
    ):
        monkeypatch = probe_recorder.monkeypatch
        monkeypatch.setattr(
            probe_recorder.platform_utils, "is_uva_available", lambda: False
        )
        assert probe_recorder.v2_runner_compat.mapped_host_buffers_available() is False
        assert probe_recorder.probed == [], "probe must short-circuit on is_uva_available"

    @pytest.mark.platforms("thead")
    def test_mapping_works_reports_available(self, probe_recorder):
        monkeypatch = probe_recorder.monkeypatch
        monkeypatch.setattr(
            probe_recorder.platform_utils, "is_uva_available", lambda: True
        )
        assert probe_recorder.v2_runner_compat.mapped_host_buffers_available() is True
        assert len(probe_recorder.probed) == 1

    @pytest.mark.platforms("thead")
    def test_mapping_runtime_error_warns_and_reports_unavailable(
        self, probe_recorder, monkeypatch
    ):
        monkeypatch = probe_recorder.monkeypatch
        monkeypatch.setattr(
            probe_recorder.platform_utils, "is_uva_available", lambda: True
        )
        monkeypatch.setattr(
            probe_recorder.torch_utils,
            "get_accelerator_view_from_cpu_tensor",
            probe_recorder.failing_view,
        )
        # Record the warning directly on the module's logger: vLLM's
        # ``warning_once`` scope check imports ``vllm.distributed`` which
        # does not resolve on every host (e.g. ``empty`` wheels without the
        # CUDA extension), so asserting on the logger call keeps this test
        # about the patch module's own contract.
        warnings = []
        monkeypatch.setattr(
            probe_recorder.v2_runner_compat.logger,
            "warning_once",
            lambda msg, *a, **k: warnings.append(msg),
        )
        result = probe_recorder.v2_runner_compat.mapped_host_buffers_available()

        assert result is False
        assert len(probe_recorder.probed) == 1
        assert len(warnings) == 1, "the probe must warn exactly once"
        assert "cannot be mapped" in warnings[0].lower()
        assert "V2 runner" in warnings[0]


class TestInstallV2RunnerCompat:
    """install_v2_runner_compat rebinds V2 runner helpers."""

    @staticmethod
    def _install_stub_modules(probe_result, fl_capture):
        """Stub the three modules install_v2_runner_compat imports.

        Returns (installer, restorer). All originals are saved and restored
        in ``finally`` so no stub ever leaks into sys.modules.
        """
        from vllm_fl.dispatch.backends.vendor.thead.patches import v2_runner_compat

        saved = {}

        buffer_utils = types.ModuleType("vllm.v1.worker.gpu.buffer_utils")
        buffer_utils.is_uva_available = lambda: probe_result
        cudagraph_utils = types.ModuleType("vllm.v1.worker.gpu.cudagraph_utils")
        cudagraph_utils.graph_capture = object()  # identity: original sentinel
        model_runner = types.ModuleType("vllm_fl.worker.model_runner")
        model_runner.graph_capture = fl_capture

        for name, module in (
            ("vllm.v1.worker.gpu.buffer_utils", buffer_utils),
            ("vllm.v1.worker.gpu.cudagraph_utils", cudagraph_utils),
            ("vllm_fl.worker.model_runner", model_runner),
            # Parent packages must also resolve for the from-imports.
            ("vllm.v1.worker.gpu", types.ModuleType("vllm.v1.worker.gpu")),
            ("vllm.v1.worker", types.ModuleType("vllm.v1.worker")),
            ("vllm.v1", types.ModuleType("vllm.v1")),
            ("vllm_fl.worker", types.ModuleType("vllm_fl.worker")),
        ):
            saved[name] = sys.modules.get(name)
            sys.modules[name] = module

        original_capture = cudagraph_utils.graph_capture

        def restore():
            for name, module in saved.items():
                if module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = module

        return (
            v2_runner_compat,
            buffer_utils,
            cudagraph_utils,
            model_runner,
            original_capture,
            restore,
        )

    @pytest.mark.platforms("thead")
    def test_probe_false_rebinds_is_uva_available_and_graph_capture(
        self, monkeypatch
    ):
        """Unmappable host memory: both helpers are rebound."""
        fl_capture = object()
        (
            compat,
            buffer_utils,
            cudagraph_utils,
            model_runner,
            original_capture,
            restore,
        ) = self._install_stub_modules(probe_result=False, fl_capture=fl_capture)
        try:
            # The probe path must report unavailable for the override to run.
            monkeypatch.setattr(
                compat, "mapped_host_buffers_available", lambda: False
            )
            compat.install_v2_runner_compat()

            assert buffer_utils.is_uva_available is compat._uva_unavailable, (
                "is_uva_available must be replaced with the unavailable stand-in"
            )
            assert buffer_utils.is_uva_available() is False
            assert cudagraph_utils.graph_capture is fl_capture, (
                "graph_capture must be rebound to the FL context manager"
            )
            assert model_runner.graph_capture is fl_capture
        finally:
            restore()

    @pytest.mark.platforms("thead")
    def test_probe_true_keeps_uva_but_still_rebinds_graph_capture(
        self, monkeypatch
    ):
        """Mappable host memory: only graph_capture is rebound (unconditional)."""
        fl_capture = object()
        original_uva = lambda: True  # noqa: E731
        (
            compat,
            buffer_utils,
            cudagraph_utils,
            model_runner,
            original_capture,
            restore,
        ) = self._install_stub_modules(probe_result=True, fl_capture=fl_capture)
        buffer_utils.is_uva_available = original_uva
        try:
            monkeypatch.setattr(
                compat, "mapped_host_buffers_available", lambda: True
            )
            compat.install_v2_runner_compat()

            assert buffer_utils.is_uva_available is original_uva, (
                "is_uva_available must stay untouched when host memory maps"
            )
            assert cudagraph_utils.graph_capture is fl_capture, (
                "graph_capture must be rebound even when the probe succeeds"
            )
        finally:
            restore()

    @pytest.mark.platforms("thead")
    def test_install_is_idempotent(self, monkeypatch):
        """A second install must not fail or double-apply changes."""
        fl_capture = object()
        (
            compat,
            buffer_utils,
            cudagraph_utils,
            model_runner,
            original_capture,
            restore,
        ) = self._install_stub_modules(probe_result=False, fl_capture=fl_capture)
        try:
            monkeypatch.setattr(
                compat, "mapped_host_buffers_available", lambda: False
            )
            compat.install_v2_runner_compat()
            after_first = (
                buffer_utils.is_uva_available,
                cudagraph_utils.graph_capture,
            )
            compat.install_v2_runner_compat()
            after_second = (
                buffer_utils.is_uva_available,
                cudagraph_utils.graph_capture,
            )
            assert after_first == after_second
            assert after_second[0] is compat._uva_unavailable
            assert after_second[1] is fl_capture
        finally:
            restore()

    @pytest.mark.platforms("thead")
    def test_uva_unavailable_stand_in_is_constant_false(self):
        from vllm_fl.dispatch.backends.vendor.thead.patches.v2_runner_compat import (
            _uva_unavailable,
        )

        assert _uva_unavailable() is False
        assert _uva_unavailable() is False  # stable across calls
