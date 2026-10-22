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
printf 'MUSA_CI_INSTALL_EVIDENCE=%s\n' "$task_dir" >> "$GITHUB_ENV"
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
"$task_python" -I -B .github/scripts/musa/dependency_profile.py stage \
  --task-dir "$task_dir" --profile "$dependency_profile" \
  --manifest "${MUSA_CI_DEPENDENCY_MANIFEST:-}" \
  --manifest-sha256 "${MUSA_CI_DEPENDENCY_MANIFEST_SHA256:-}" \
  > "$task_dir/dependency-stage.stdout.raw" 2> "$task_dir/dependency-stage.stderr.raw"
"$task_python" -I -B .github/scripts/musa/gate_observer_wheel.py stage --task-dir "$task_dir" \
  > "$task_dir/gate-observer-stage.stdout.raw" 2> "$task_dir/gate-observer-stage.stderr.raw"
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
"$task_python" -I -B .github/scripts/musa/gate_observer_wheel.py installed --task-dir "$task_dir" \
  > "$task_dir/gate-observer-installed.stdout.raw" 2> "$task_dir/gate-observer-installed.stderr.raw"
# Install only ordinary pytest tooling, not the test extra's vllm[audio] runtime.
# Retain existing tool versions where present and verify core metadata before/after.
"$task_python" -I -B - "$task_dir" <<'PY_TEST_TOOLS'
import email.parser, hashlib, importlib.metadata as md, importlib.util, json, os, pathlib, re, subprocess, sys, zipfile
root = pathlib.Path(sys.argv[1]).resolve(strict=True)
assert pathlib.Path(sys.prefix).resolve(strict=True) == root / 'venv'
assert sys.flags.isolated
wheelhouse = root / 'test-wheels'; wheelhouse.mkdir()
temporary = root / 'test-tmp'; temporary.mkdir()
allowed = {'pytest', 'pytest-cov', 'pytest-json-report', 'pytest-timeout', 'coverage', 'pytest-metadata',
           'iniconfig', 'packaging', 'pluggy', 'pygments', 'exceptiongroup', 'tomli', 'typing-extensions',
           'colorama', 'py', 'importlib-metadata', 'zipp'}
protected = ('vllm', 'vllm-plugin-fl', 'torch', 'torch-musa',
             'flag-gems', 'flagtree', 'pyyaml', 'triton')
blocked_roots = {'vllm', 'vllm_fl', 'torch', 'torch_musa', 'flag_gems',
                 'flagtree', 'triton', 'yaml', '_yaml'}
report = {'source_index': 'https://pypi.org/simple', 'ordinary_wheels_only': True,
          'allowed_distributions': sorted(allowed), 'commands': [], 'wheels': [], 'passed': False}
def normalized(name):
    return re.sub(r'[-_.]+', '-', name).lower()
def snapshot():
    result = {}
    for name in protected:
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            result[name] = None
            continue
        path = pathlib.Path(dist._path).resolve(strict=True)
        result[name] = {'version': dist.version, 'metadata_directory': str(path),
                        'files': {filename: hashlib.sha256((path / filename).read_bytes()).hexdigest()
                                  for filename in ('METADATA', 'RECORD') if (path / filename).is_file()}}
    return result
pip_env = {key: value for key, value in os.environ.items()
           if not key.upper().startswith('PIP_') and key.upper() not in {'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'}}
pip_env.update({'PIP_CONFIG_FILE': os.devnull, 'TMPDIR': str(temporary), 'TEMP': str(temporary), 'TMP': str(temporary)})
def run(label, argv, seconds):
    entry = {'label': label, 'argv': argv, 'cwd': str(root)}; report['commands'].append(entry)
    stdout = root / (label + '.stdout.raw'); stderr = root / (label + '.stderr.raw')
    entry.update({'stdout': str(stdout), 'stderr': str(stderr)})
    with stdout.open('xb') as out, stderr.open('xb') as err:
        result = subprocess.run(argv, cwd=root, env=pip_env, stdout=out, stderr=err, timeout=seconds)
    entry['actual_exit'] = result.returncode
    if result.returncode != 0:
        raise RuntimeError(label + ' exited ' + str(result.returncode))
    return stdout
try:
    report['protected_before'] = snapshot()
    pytest_version = md.version('pytest')
    constraints = []
    for name in sorted(allowed):
        try:
            version = md.version(name)
        except md.PackageNotFoundError:
            continue
        if not re.fullmatch(r'[A-Za-z0-9.+!_-]+', version):
            raise ValueError('Invalid installed test-tool version')
        constraints.append(name + '==' + version)
    constraint_path = root / 'test-tools.constraints.txt'
    constraint_path.write_text('\n'.join(constraints) + '\n')
    report['existing_tool_constraints'] = constraints
    pip = [sys.executable, '-I', '-B', '-m', 'pip', '--isolated', '--disable-pip-version-check', '--no-input']
    run('test-tools-download', pip + ['download', '--index-url', report['source_index'], '--only-binary=:all:',
        '--no-cache-dir', '--constraint', str(constraint_path), '--dest', str(wheelhouse),
        'pytest-cov', 'pytest-json-report', 'pytest-timeout'], 600)
    seen = set()
    for wheel in sorted(wheelhouse.iterdir()):
        if not wheel.is_file() or wheel.suffix != '.whl':
            raise ValueError('Only ordinary test-tool wheels are allowed')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate wheel member')
            for member in names:
                parts = pathlib.PurePosixPath(member).parts
                if not parts or member.startswith('/') or '\\' in member or '..' in parts:
                    raise ValueError('Unsafe test-tool wheel member')
                package_root = parts[2] if len(parts) >= 3 and parts[0].endswith('.data') and parts[1] in {'purelib', 'platlib'} else parts[0]
                if package_root in blocked_roots:
                    raise ValueError('Test tool cannot overwrite runtime packages')
            metadata_names = [name for name in names if len(pathlib.PurePosixPath(name).parts) == 2 and name.endswith('.dist-info/METADATA')]
            if len(metadata_names) != 1:
                raise ValueError('One test-tool METADATA is required')
            metadata = email.parser.BytesParser().parsebytes(archive.read(metadata_names[0]))
            name = normalized(metadata['Name'])
            if name not in allowed or name in seen:
                raise ValueError('Unexpected or duplicate test-tool dependency')
            seen.add(name)
        report['wheels'].append({'name': name, 'version': metadata['Version'], 'path': str(wheel),
            'bytes': wheel.stat().st_size, 'sha256': hashlib.sha256(wheel.read_bytes()).hexdigest()})
    if not {'pytest', 'pytest-cov', 'pytest-json-report', 'pytest-timeout'} <= seen:
        raise ValueError('Required pytest tool wheels are incomplete')
    run('test-tools-install', pip + ['install', '--ignore-installed', '--no-index', '--no-deps', '--no-cache-dir',
        '--no-compile', *[row['path'] for row in report['wheels']]], 300)
    report['protected_after'] = snapshot()
    if report['protected_after'] != report['protected_before'] or md.version('pytest') != pytest_version:
        raise RuntimeError('Test tools changed the protected runtime or pytest baseline')
    report['installed'] = []
    for row in report['wheels']:
        dist = md.distribution(row['name']); location = pathlib.Path(dist._path).resolve(strict=True)
        if dist.version != row['version'] or not location.is_relative_to(pathlib.Path(sys.prefix).resolve(strict=True)):
            raise RuntimeError('Test tool is not installed from its ordinary wheel in this venv')
        report['installed'].append({'name': row['name'], 'version': dist.version, 'metadata_directory': str(location)})
    report['plugin_modules'] = []
    for name in ('pytest', 'pytest_cov', 'pytest_jsonreport', 'pytest_timeout'):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None or not pathlib.Path(spec.origin).resolve(strict=True).is_relative_to(pathlib.Path(sys.prefix).resolve(strict=True)):
            raise RuntimeError('Test plugin source is outside the task venv')
        report['plugin_modules'].append({'name': name, 'path': spec.origin})
    help_output = run('pytest-help', [sys.executable, '-I', '-B', '-m', 'pytest', '--help'], 60)
    help_text = help_output.read_text(errors='replace')
    report['required_options'] = {option: option in help_text for option in ('--cov=', '--cov-report=', '--json-report', '--json-report-file=', '--timeout=')}
    if not all(report['required_options'].values()):
        raise RuntimeError('Original pytest coverage/JSON/timeout options are unavailable')
    report['passed'] = True
except Exception as error:
    report['error_type'] = type(error).__name__; report['error'] = str(error)
    raise
finally:
    (root / 'test-tools.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
PY_TEST_TOOLS
"$task_python" -I -B .github/scripts/install_flaggems_vllm.py \
  > "$task_dir/flaggems-vllm-install.stdout.raw" 2> "$task_dir/flaggems-vllm-install.stderr.raw"
"$task_python" -I -B .github/scripts/musa/dependency_profile.py installed \
  --task-dir "$task_dir" --profile "$dependency_profile" \
  --manifest "${MUSA_CI_DEPENDENCY_MANIFEST:-}" \
  --manifest-sha256 "${MUSA_CI_DEPENDENCY_MANIFEST_SHA256:-}" \
  > "$task_dir/dependency-installed.stdout.raw" 2> "$task_dir/dependency-installed.stderr.raw"
"$task_python" -I -B - "$task_dir" <<'PY' > "$task_dir/imports.json"
import hashlib, importlib, importlib.metadata as md, json, pathlib, sys
root = pathlib.Path(sys.argv[1]); site = root / 'venv/lib/python3.10/site-packages'
# Use the running interpreter's site directory, rather than assume Python 3.10 in CI.
import sysconfig
site = pathlib.Path(sysconfig.get_path('purelib')).resolve()
profile = json.loads((root / 'dependency-profile-staged.json').read_text())['profile']
rows = []
names = ['vllm', 'vllm_fl', 'flaggems_vllm', 'torch', 'torch_musa', 'flag_gems', 'triton', 'pytest', 'openai', 'httpx']
if profile == 'latest':
    names.append('yaml')
private = {'vllm', 'vllm_fl', 'flaggems_vllm'}
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
print(json.dumps({'dependency_profile': profile, 'imports': rows, 'device_count': torch.musa.device_count(),
                  'platform': current_platform.device_type}, indent=2))
PY
"$task_python" -I -B .github/scripts/musa/cache_check.py > "$task_dir/cache.json"
printf '%s\n' "$task_dir/venv/bin" >> "$GITHUB_PATH"
printf 'MUSA_CI_PYTHON=%s\n' "$task_python" >> "$GITHUB_ENV"
cat "$task_dir/imports.json"
cat "$task_dir/cache.json"
