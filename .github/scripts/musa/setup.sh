#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Keep vendor Torch/SDK; default baseline or explicit SHA-bound private dependencies.
set -euo pipefail

: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${MUSA_CI_EMPTY_WHEEL:?Set a prebuilt vLLM 0.28.0+empty wheel path}"
: "${MUSA_CI_EMPTY_WHEEL_SHA256:?Set the wheel SHA256}"
: "${GEMS_VENDOR:?GEMS_VENDOR is required}"
: "${VLLM_PLUGINS:?VLLM_PLUGINS is required}"
: "${MTHREADS_VISIBLE_DEVICES:?Set the allocated physical GPU mask}"
: "${MUSA_VISIBLE_DEVICES:?Set the tested logical GPU mask}"
: "${GITHUB_PATH:?GITHUB_PATH is required}"
: "${GITHUB_ENV:?GITHUB_ENV is required}"

git config --global --add safe.directory "$(pwd)"
dependency_profile="${MUSA_CI_DEPENDENCY_PROFILE:-baseline}"
case "$dependency_profile" in baseline|latest) ;; *) echo "Unknown MUSA dependency profile" >&2; exit 1 ;; esac
export VLLM_VENDOR=musa
task_dir=$(mktemp -d "${RUNNER_TEMP}/musa-install.XXXXXXXX")
mkdir "$task_dir/wheels"
python .github/scripts/musa/verify_empty_wheel.py \
  "$MUSA_CI_EMPTY_WHEEL" "$MUSA_CI_EMPTY_WHEEL_SHA256" "$task_dir/wheels" \
  > "$task_dir/empty-wheel.json"
python -I -B -m venv --system-site-packages --without-pip "$task_dir/venv"
task_python="$task_dir/venv/bin/python"
# Empty builds are ordinary Python wheels. The runtime vendor remains MUSA,
# but setup.py supports VLLM_VENDOR only for its optional CUDA extension build.
# Let setuptools-scm derive the package version from the actual CI checkout.
build_env=(env -u VLLM_VENDOR)
while IFS= read -r variable; do
  case "$variable" in
    SETUPTOOLS_SCM_PRETEND_VERSION*) build_env+=(-u "$variable") ;;
  esac
done < <(compgen -e)
"${build_env[@]}" "$task_python" -I -B -m pip --isolated wheel --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$task_dir/wheels" . > "$task_dir/build.stdout.raw" 2> "$task_dir/build.stderr.raw"
"${build_env[@]}" "$task_python" -I -B -m pip --isolated wheel --no-index --no-deps --no-build-isolation --no-cache-dir   --wheel-dir "$task_dir/wheels" ./tools/musa-gate-observer   > "$task_dir/observer-build.stdout.raw" 2> "$task_dir/observer-build.stderr.raw"
"$task_python" -I -B .github/scripts/musa/dependency_profile.py stage \
  --task-dir "$task_dir" --profile "$dependency_profile" \
  --manifest "${MUSA_CI_DEPENDENCY_MANIFEST:-}" \
  --manifest-sha256 "${MUSA_CI_DEPENDENCY_MANIFEST_SHA256:-}" \
  > "$task_dir/dependency-stage.stdout.raw" 2> "$task_dir/dependency-stage.stderr.raw"
"$task_python" -I -B - "$task_dir" <<'PY'
import email, hashlib, json, pathlib, subprocess, sys, zipfile
root = pathlib.Path(sys.argv[1])
distributions = {}
for wheel in (root / 'wheels').glob('*.whl'):
    with zipfile.ZipFile(wheel) as z:
        names = [n for n in z.namelist() if len(pathlib.PurePosixPath(n).parts) == 2 and n.endswith('.dist-info/METADATA')]
        assert len(names) == 1, wheel
        metadata = email.message_from_bytes(z.read(names[0]))
    name = metadata['Name'].lower().replace('-', '_')
    assert name not in distributions, name
    distributions[name] = str(wheel)
profile = json.loads((root / 'dependency-profile-staged.json').read_text())
expected = {'vllm', 'vllm_plugin_fl', 'musa_gate_observer'}
if profile['profile'] == 'latest':
    expected |= {'flag_gems', 'flagtree', 'pyyaml'}
assert set(distributions) == expected, distributions
order = ['vllm', 'vllm_plugin_fl', 'musa_gate_observer']
if profile['profile'] == 'latest':
    order += ['flag_gems', 'flagtree', 'pyyaml']
argv = [sys.executable, '-I', '-B', '-m', 'pip', '--isolated', 'install', '--ignore-installed',
        '--no-index', '--no-deps', '--no-cache-dir', '--no-compile',
        *[distributions[name] for name in order]]
(root / 'install.argv.json').write_text(json.dumps(argv, indent=2) + '\n')
with (root / 'install.stdout.raw').open('xb') as out, (root / 'install.stderr.raw').open('xb') as err:
    result = subprocess.run(argv, stdout=out, stderr=err, timeout=300)
(root / 'install.exit').write_text(str(result.returncode) + '\n')
assert result.returncode == 0, result.returncode
rows = [{'path': p, 'sha256': hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()}
        for p in distributions.values()]
(root / 'installed-wheels.json').write_text(json.dumps(rows, indent=2) + '\n')
PY
"$task_python" -I -B .github/scripts/musa/dependency_profile.py installed \
  --task-dir "$task_dir" --profile "$dependency_profile" \
  --manifest "${MUSA_CI_DEPENDENCY_MANIFEST:-}" \
  --manifest-sha256 "${MUSA_CI_DEPENDENCY_MANIFEST_SHA256:-}" \
  > "$task_dir/dependency-installed.stdout.raw" 2> "$task_dir/dependency-installed.stderr.raw"
"$task_python" -I -B - "$task_dir" <<'PY' > "$task_dir/imports.json"
import base64, csv, hashlib, importlib, importlib.metadata as md, io, json, pathlib, sys, zipfile
root = pathlib.Path(sys.argv[1]); site = root / 'venv/lib/python3.10/site-packages'
# Use the running interpreter's site directory, rather than assume Python 3.10 in CI.
import sysconfig
site = pathlib.Path(sysconfig.get_path('purelib')).resolve()
profile = json.loads((root / 'dependency-profile-staged.json').read_text())['profile']
rows = []
names = ['vllm', 'vllm_fl', 'musa_gate_observer', 'torch', 'torch_musa', 'flag_gems', 'triton', 'pytest', 'openai', 'httpx']
if profile == 'latest':
    names.append('yaml')
private = {'vllm', 'vllm_fl', 'musa_gate_observer'}
vendor = {'torch', 'torch_musa'}
if profile == 'latest':
    private |= {'flag_gems', 'triton', 'yaml'}
else:
    vendor |= {'flag_gems', 'triton'}
for name in names:
    module = importlib.import_module(name); source = pathlib.Path(module.__file__).resolve(strict=True)
    if name in private:
        assert source.is_relative_to(site), (name, source, site)
    elif name in vendor:
        assert not source.is_relative_to(site), (name, 'vendor package unexpectedly replaced')
    rows.append({'name': name, 'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                 'version': getattr(module, '__version__', None)})
import torch
from vllm.platforms import current_platform
assert torch.musa.is_available() and torch.musa.device_count() == 4
assert current_platform.device_type == 'musa', current_platform.device_type
assert md.version('vllm') == '0.28.0+empty', md.version('vllm')
# Read the newly installed observer payload and ordinary wheel RECORD directly.
# Importing the observer's empty __init__ does not activate its server plugin.
dist = md.distribution('musa-gate-observer')
dist_root = pathlib.Path(dist.locate_file('')).resolve(strict=True)
assert dist_root == site, (dist_root, site)
record_paths = [f for f in dist.files if str(f).endswith('.dist-info/RECORD')]
assert len(record_paths) == 1
record_path = pathlib.Path(dist.locate_file(record_paths[0])).resolve(strict=True)
record_raw = record_path.read_bytes()
record_rows = list(csv.reader(io.StringIO(record_raw.decode('utf-8'))))
assert all(len(r) == 3 for r in record_rows)
record_map = {r[0]: r[1:] for r in record_rows}
assert len(record_map) == len(record_rows)
wheel_rows = json.loads((root / 'installed-wheels.json').read_text())
observer_wheels = []
for row in wheel_rows:
    with zipfile.ZipFile(row['path']) as z:
        metadata_names = [n for n in z.namelist() if len(pathlib.PurePosixPath(n).parts) == 2 and n.endswith('.dist-info/METADATA')]
        assert len(metadata_names) == 1
        import email
        metadata = email.message_from_bytes(z.read(metadata_names[0]))
        if metadata['Name'] == 'musa-gate-observer':
            assert metadata['Version'] == dist.version == '0.1.0'
            observer_wheels.append(row)
            payload = []
            for name in ['__init__.py', 'trace.py', 'server.py', 'pytest_plugin.py', 'summarize.py']:
                member = 'musa_gate_observer/' + name
                body = z.read(member)
                actual = pathlib.Path(dist.locate_file(member)).resolve(strict=True)
                assert actual.is_relative_to(site) and actual.read_bytes() == body, member
                digest = hashlib.sha256(body).digest()
                encoded = base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')
                assert record_map[member] == ['sha256=' + encoded, str(len(body))], member
                payload.append({'path': str(actual), 'bytes': len(body), 'sha256': digest.hex()})
assert len(observer_wheels) == 1
direct_paths = [f for f in dist.files if str(f).endswith('.dist-info/direct_url.json')]
assert len(direct_paths) == 1
direct_path = pathlib.Path(dist.locate_file(direct_paths[0])).resolve(strict=True)
direct_url = json.loads(direct_path.read_bytes())
assert 'dir_info' not in direct_url and direct_url.get('archive_info'), direct_url
expected_wheel = observer_wheels[0]
assert direct_url['url'] == pathlib.Path(expected_wheel['path']).resolve(strict=True).as_uri(), direct_url
archive_info = direct_url['archive_info']
assert isinstance(archive_info, dict)
assert archive_info.get('hashes') == {'sha256': expected_wheel['sha256']}, archive_info
if 'hash' in archive_info:
    assert archive_info['hash'] == 'sha256=' + expected_wheel['sha256'], archive_info
observer_install = {'distribution': dist.metadata['Name'], 'version': dist.version,
                    'metadata_path': str(dist._path), 'wheel': observer_wheels[0],
                    'record': {'path': str(record_path), 'bytes': len(record_raw),
                               'sha256': hashlib.sha256(record_raw).hexdigest()},
                    'direct_url': direct_url, 'payload': payload, 'editable': False}
print(json.dumps({'dependency_profile': profile, 'imports': rows, 'device_count': torch.musa.device_count(),
                  'platform': current_platform.device_type, 'observer_install': observer_install}, indent=2))
PY
"$task_python" -I -B .github/scripts/musa/cache_check.py > "$task_dir/cache.json"
printf '%s\n' "$task_dir/venv/bin" >> "$GITHUB_PATH"
printf 'MUSA_CI_PYTHON=%s\nMUSA_CI_INSTALL_EVIDENCE=%s\n' "$task_python" "$task_dir" >> "$GITHUB_ENV"
cat "$task_dir/imports.json"
cat "$task_dir/cache.json"
