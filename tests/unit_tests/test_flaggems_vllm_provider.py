# Copyright 2026 FlagOS Contributors

import sys
from types import ModuleType


def test_fused_experts_uses_flaggems_vllm_without_changing_arguments(monkeypatch):
    from vllm_fl.quantization.w8a8 import moe_experts

    calls = []
    module = ModuleType("flaggems_vllm")

    def invoke(**kwargs):
        calls.append(kwargs)
        return kwargs["hidden_states"]

    module.fused_experts_impl = invoke
    monkeypatch.setitem(sys.modules, "flaggems_vllm", module)
    value = object()
    assert (
        moe_experts._flaggems_fused_experts_impl(hidden_states=value, inplace=False)
        is value
    )
    assert calls == [{"hidden_states": value, "inplace": False}]


def test_enflame_setup_selects_installer_pin(monkeypatch):
    import importlib.util
    import shlex
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "installer_under_test", root / ".github/scripts/install_flaggems_vllm.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    commands = [
        shlex.split(line)
        for line in (root / ".github/scripts/enflame/setup.sh").read_text().splitlines()
        if line.startswith("python .github/scripts/install_flaggems_vllm.py")
    ]
    assert commands == []
    setup = (root / ".github/scripts/enflame/setup.sh").read_text()
    assert "python -m pip install --no-build-isolation --no-deps -e ." in setup


def test_enflame_installer_pin_when_explicitly_invoked(monkeypatch):
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "installer_under_test", root / ".github/scripts/install_flaggems_vllm.py"
    )
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    installed = []
    checked = []
    monkeypatch.setattr(installer, "package_version", lambda name: "test-version")

    def release_is_installed(commit, source_archive, archive_sha256):
        checked.append((commit, source_archive, archive_sha256))
        return bool(installed)

    monkeypatch.setattr(installer, "release_is_installed", release_is_installed)
    monkeypatch.setattr(installer, "pip_install", lambda *args: installed.append(args))
    installer.main(["--platform", "enflame"])
    assert checked == [(installer.ENFLAME_COMMIT, None, None)] * 2
    assert installed == [
        (
            "--no-deps",
            "--no-build-isolation",
            "--force-reinstall",
            f"flaggems-vllm @ git+{installer.REPOSITORY}@{installer.ENFLAME_COMMIT}",
        )
    ]


def test_default_installer_keeps_shared_pin(monkeypatch):
    import runpy
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    namespace = runpy.run_path(str(root / ".github/scripts/install_flaggems_vllm.py"))
    main = namespace["main"]
    checked = []
    monkeypatch.setitem(
        main.__globals__, "package_version", lambda name: "test-version"
    )
    monkeypatch.setitem(
        main.__globals__,
        "release_is_installed",
        lambda commit, source_archive, archive_sha256: (
            checked.append((commit, source_archive, archive_sha256)) or True
        ),
    )
    main([])
    assert checked == [(namespace["COMMIT"], None, None)] * 2
