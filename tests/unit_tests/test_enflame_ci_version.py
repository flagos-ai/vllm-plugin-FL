"""Execute the CI package gate without installing or importing device packages."""
import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[2]
VERSIONS = {
    'torch': '2.11.0+cpu',
    'torch-gcu': '2.11.0+3.8.20260713',
    'flag-gems': '5.4.0',
    'flagtree': '0.7.0+enflame3.6',
    'triton-gcu': '3.6.0+1.0.20260722',
    'openai': '3.28.0',
    'pandas': '3.0.6',
}


def run_gate(vllm_version, **overrides):
    script = (ROOT / '.github/scripts/enflame/setup.sh').read_text()
    gate = script.split("python - <<'PY'\n", 1)[1].split('\nPY', 1)[0]
    versions = {**VERSIONS, 'vllm': vllm_version, **overrides}
    with patch('importlib.metadata.version', side_effect=versions.__getitem__):
        exec(compile(gate, 'enflame-ci-version-gate', 'exec'), {})


def test_current_empty_release_passes():
    run_gate('0.32.0+empty')


@pytest.mark.parametrize('version', ['0.28.0+empty', '0.32.0', '0.32.1+empty'])
def test_incompatible_release_rejected(version):
    with pytest.raises(RuntimeError, match='Expected vLLM 0.32.0 empty build'):
        run_gate(version)


def test_vendor_runtime_drift_rejected():
    with pytest.raises(RuntimeError, match='Expected torch '):
        run_gate('0.32.0+empty', torch='2.12.0')


@pytest.mark.parametrize('override', [None, '0.33.0'])
def test_build_entrypoint_forwards_enflame_version(tmp_path, override):
    docker = tmp_path / 'docker'
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\n')
    docker.chmod(0o755)
    environment = os.environ.copy()
    environment['PATH'] = f'{tmp_path}:{environment["PATH"]}'
    environment.pop('ENFLAME_VLLM_VERSION', None)
    if override is not None:
        environment['ENFLAME_VLLM_VERSION'] = override
    result = subprocess.run(
        ['bash', str(ROOT / 'docker/build.sh'), '--platform', 'enflame',
         '--target', 'ci', '--image-name', 'local/vllm-plugin-fl',
         '--image-tag', 'test-enflame-version'],
        env=environment, check=True, text=True, capture_output=True,
    )
    version = override or '0.32.0'
    assert f'VLLM_VERSION={version}\n' in result.stdout
    assert str(ROOT / 'docker/enflame') in result.stdout
