"""Prepare/serve a private runtime from verified installed sources and overlays.

Never writes installed packages or another checkout. The source bundle includes
no weights, device executables, Triton/Inductor/IR caches or credentials. A
source-verified tuning database is copied to a private cache, not edited here.
"""
import argparse
import hashlib
import json
import os
import shutil
import site
import sys
from pathlib import Path, PurePosixPath


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tree_hashes(folder):
    return {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*')
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def package_root():
    return Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'verify', 'serve'])
    parser.add_argument('--runtime', required=True, type=Path)
    parser.add_argument('--model', type=Path)
    args = parser.parse_args()
    root = package_root()
    manifest = json.loads((root / 'manifest.json').read_text())
    for rel, expected in manifest['payload_SHA256'].items():
        assert digest(root / rel) == expected, ('bundle payload changed', rel)
    runtime = args.runtime.resolve()
    if args.action == 'prepare':
        assert not runtime.exists(), 'Choose a new independent runtime directory'
        # No library import before private source paths are established. Avoid
        # invoking model/platform initializers during source preparation.
        sites = [Path(p) for p in site.getsitepackages()]
        bases = {}
        for name in ['vllm', 'flag_gems']:
            matches = [p / name for p in sites if (p / name / '__init__.py').exists()]
            assert len(matches) == 1, (name, matches)
            base = matches[0]
            for rel, expected in manifest['base_SHA256'][name].items():
                assert digest(base / rel) == expected, ('Installed source differs', name, rel)
            bases[name] = base
        assert shutil.disk_usage(runtime.parent).free > 512 * 1024**2
        runtime.mkdir()
        for name, subdir in [('vllm', 'vllm_source'), ('flag_gems', 'gems_source')]:
            shutil.copytree(bases[name], runtime / subdir / name,
                            ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
            for p in (root / 'overlays' / name).rglob('*'):
                if p.is_file():
                    target = runtime / subdir / name / p.relative_to(root / 'overlays' / name)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
        shutil.copytree(root / 'plugin', runtime / 'plugin')
        for name in ['triton', 'cache', 'inductor', 'flaggems', 'q_operand_ir', 'config', 'tmp']:
            (runtime / name).mkdir()
        for p in (root / 'tuning').rglob('*'):
            if p.is_file():
                dest = runtime / 'flaggems' / p.relative_to(root / 'tuning')
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, dest)
        state = {'source': manifest['candidate'], 'bundle_version': manifest['version'],
                 'bundle_manifest_SHA256': digest(root / 'manifest.json'),
                 'initial_device_binary_IR_graph_caches_empty': True,
                 'tuning_database_seed': manifest['tuning_seed'],
                 'installed_source_writes': False}
        (runtime / 'prepared.json').write_text(json.dumps(state, indent=2))
    state = json.loads((runtime / 'prepared.json').read_text())
    assert state['bundle_manifest_SHA256'] == digest(root / 'manifest.json')
    for name, subdir in [('vllm', 'vllm_source'), ('flag_gems', 'gems_source')]:
        actual = tree_hashes(runtime / subdir / name)
        assert actual == manifest['expected_source_SHA256'][name], ('private source mismatch', name)
    assert tree_hashes(runtime / 'plugin' / 'vllm_fl') == manifest['expected_source_SHA256']['vllm_fl']
    if args.action in ['prepare', 'verify']:
        print(json.dumps({'runtime': str(runtime), 'all_source_hashes_exact': True,
                          'source': manifest['candidate'], 'action': args.action}))
        return
    assert args.model is not None and args.model.is_dir(), 'Supply the local model directory'
    # Keep the socket path below sockaddr_un's 107-character limit without
    # writing to a global /tmp directory. Require a short workspace path.
    assert len(str(runtime / 'tmp')) + 1 + 36 <= 107, 'Use a shorter independent runtime path'
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('TIANSHU_'):
            env.pop(key)
    env.update(VLLM_PLUGINS='fl', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=':'.join(str(runtime / p) for p in ['vllm_source', 'gems_source', 'plugin']),
               TMPDIR=str(runtime / 'tmp'), XDG_CACHE_HOME=str(runtime / 'cache'),
               TRITON_CACHE_DIR=str(runtime / 'triton'), TORCHINDUCTOR_CACHE_DIR=str(runtime / 'inductor'),
               FLAGGEMS_CACHE_DIR=str(runtime / 'flaggems'), XDG_CONFIG_HOME=str(runtime / 'config'),
               TIANSHU_CANDIDATE_RECEIPT=str(runtime), TIANSHU_Q_OPERAND_CACHE=str(runtime / 'q_operand_ir'))
    command = [str(args.model.resolve()) if item == '<MODEL>' else item for item in manifest['serve_argv']]
    (runtime / 'launch.json').write_text(json.dumps({'argv': command, 'source': state['source'],
                                                  'runtime': str(runtime), 'diagnostic_observers': False}, indent=2))
    os.execvpe(command[0], command, env)


if __name__ == '__main__':
    main()
