# Copyright (c) 2026 BAAI. All rights reserved.
"""CPU-only tests for platform-specific dependency provenance selection."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def installer():
    path = ROOT / '.github/scripts/install_flaggems_vllm.py'
    spec = importlib.util.spec_from_file_location('ci_provider_installer', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('args,selected', [([], 'COMMIT'), (['--platform', 'enflame'], 'ENFLAME_COMMIT'), (['--platform', 'nvidia'], 'COMMIT')])
def test_platform_pin_and_runtime_preservation(installer, monkeypatch, args, selected):
    commit = getattr(installer, selected)
    calls = []
    checks = []
    monkeypatch.setattr(installer, 'package_version', lambda name: 'installed')
    def installed(value):
        checks.append(value)
        return len(checks) > 1
    monkeypatch.setattr(installer, 'release_is_installed', installed)
    monkeypatch.setattr(installer, 'pip_install', lambda *a: calls.append(a))
    installer.main(args)
    assert checks == [commit, commit]
    assert len(calls) == 1
    assert calls[0][-1] == f'flaggems-vllm @ git+{installer.REPOSITORY}@{commit}'
    assert '--no-deps' in calls[0]
    assert '--no-build-isolation' in calls[0]


@pytest.mark.parametrize('valid', [True, False])
def test_provenance_checks_selected_commit(installer, monkeypatch, valid):
    source = {'url': installer.REPOSITORY, 'vcs_info': {'commit_id': installer.ENFLAME_COMMIT if valid else installer.COMMIT}}
    monkeypatch.setattr(installer, 'distribution', lambda name: SimpleNamespace(read_text=lambda path: json.dumps(source)))
    assert installer.release_is_installed(installer.ENFLAME_COMMIT) is valid


def test_runtime_mutation_rejected(installer, monkeypatch):
    state = {'changed': False}
    monkeypatch.setattr(installer, 'package_version', lambda name: 'after' if state['changed'] and name == 'torch' else 'before')
    monkeypatch.setattr(installer, 'release_is_installed', lambda commit: state['changed'])
    monkeypatch.setattr(installer, 'pip_install', lambda *a: state.update(changed=True))
    with pytest.raises(RuntimeError, match='runtime dependencies changed'):
        installer.main(['--platform', 'enflame'])
