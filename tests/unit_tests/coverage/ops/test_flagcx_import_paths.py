# Copyright (c) 2026 BAAI. All rights reserved.

"""Import-time environment tests for ``flagcx_connector`` (lines 72-84).

``vllm_fl/distributed/kv_transfer/flagcx_connector.py`` lines 74, 80, 81:

* line 74 — ``FLAGCX_PATH`` pointing at a real directory that is not yet on
  ``sys.path`` gets appended at import time;
* lines 80-81 — a failing ``plugin.interservice.flagcx_wrapper`` import
  re-raises as an ``ImportError`` with the FLAGCX_PATH guidance message.

The flagcx connector is re-executed from its source file under a unique
module name against the shared
``tests.unit_tests.coverage.distributed._flagcx_stubs.FlagcxStubEnv``
boundary (the ``test_flagcx_transfer_plan.py`` pattern), so its import-time
environment handling runs unmodified.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[4]
_CONNECTOR_SOURCE = _ROOT / "vllm_fl/distributed/kv_transfer/flagcx_connector.py"

pytest.importorskip("msgspec")
pytest.importorskip("numpy")
pytest.importorskip("zmq")


@pytest.fixture()
def load_connector(monkeypatch):
    """Yield a loader executing the connector source under a unique name.

    Uses the shared ``_flagcx_stubs.FlagcxStubEnv`` boundary (installed via
    ``monkeypatch.setitem`` so nothing leaks), minus the wrapper stub: the
    import-time wrapper import is what the tests vary.  ``sys.path`` is
    snapshotted so the ``FLAGCX_PATH`` append is rolled back afterwards.
    """
    from tests.unit_tests.coverage.distributed._flagcx_stubs import (
        FlagcxStubEnv,
    )

    subject = "_test_ops_midtail_flagcx_connector"
    monkeypatch.setattr(sys, "path", sys.path.copy())
    monkeypatch.delenv("FLAGCX_PATH", raising=False)

    def load(install_wrapper=True, library=None):
        env = FlagcxStubEnv(monkeypatch, flagcx_library=library)
        env.install()
        if not install_wrapper:
            # Nullify the wrapper stub the env just installed: a None entry
            # makes ``from plugin.interservice.flagcx_wrapper import ...``
            # raise ImportError, which is the code path under test.
            monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
        spec = importlib.util.spec_from_file_location(subject, _CONNECTOR_SOURCE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield load
    sys.modules.pop(subject, None)


class TestFlagcxConnectorImportPaths:
    """flagcx_connector lines 72-84: sys.path append and wrapper ImportError."""

    def test_flagcx_path_dir_appended_to_sys_path(
        self, load_connector, monkeypatch, tmp_path
    ):
        """A valid FLAGCX_PATH directory is appended to sys.path.

        Independent oracle: a fresh directory (guaranteed absent from
        sys.path) is used; after the import it must be the last sys.path
        entry.
        """
        flagcx_root = tmp_path / "flagcx-root"
        flagcx_root.mkdir()
        monkeypatch.setenv("FLAGCX_PATH", str(flagcx_root))
        load_connector()
        assert str(flagcx_root) in sys.path
        assert sys.path[-1] == str(flagcx_root)

    def test_flagcx_path_missing_dir_not_appended(self, load_connector, monkeypatch):
        """A nonexistent FLAGCX_PATH directory leaves sys.path untouched."""
        monkeypatch.setenv("FLAGCX_PATH", "/definitely/not/a/real/dir/xyz")
        before = list(sys.path)
        load_connector()
        assert sys.path == before

    def test_wrapper_import_failure_raises_guidance(self, load_connector, monkeypatch):
        """A missing FlagCX wrapper re-raises ImportError with guidance.

        Independent oracle: the error message must name FLAGCX_PATH and the
        expected wrapper module path, and the original ImportError must be
        chained as ``__cause__``.
        """
        monkeypatch.setitem(sys.modules, "plugin.interservice.flagcx_wrapper", None)
        with pytest.raises(
            ImportError, match="Cannot import FlagCX wrapper"
        ) as excinfo:
            load_connector(install_wrapper=False)
        assert "FLAGCX_PATH" in str(excinfo.value)
        assert "plugin/interservice/flagcx_wrapper.py" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ImportError)
