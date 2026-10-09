#!/usr/bin/env python3
"""Validate the installed cache APIs and actual mount flags before MUSA JIT."""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def check_cache(module, variable, actual):
    configured = os.environ[variable]
    actual = Path(actual)
    if str(actual) != configured or not actual.is_absolute():
        raise RuntimeError(f'{variable}: installed API differs from configuration')
    actual.mkdir(parents=True, exist_ok=True)
    if actual.resolve() != actual:
        raise RuntimeError(f'{variable}: symlink cache path')
    flags = os.statvfs(actual).f_flag
    if flags & os.ST_NOEXEC:
        raise RuntimeError(f'{variable}: effective cache mount is noexec')
    with tempfile.NamedTemporaryFile(prefix='musa-ci-', dir=actual) as probe:
        probe.write(b'musa-ci-cache-probe')
        probe.flush()
        if Path(probe.name).read_bytes() != b'musa-ci-cache-probe':
            raise RuntimeError('Cache read/write probe failed')
    path = Path(module.__file__).resolve(strict=True)
    return {'variable': variable, 'configured': configured, 'effective': str(actual),
            'module': str(path), 'module_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'statvfs_flags': flags, 'write_read_remove_passed': True}


def main():
    from triton import knobs
    import flag_gems.utils.code_cache as gems
    import torch._inductor.runtime.cache_dir_utils as inductor
    rows = [check_cache(knobs, 'TRITON_CACHE_DIR', knobs.cache.dir),
            check_cache(gems, 'FLAGGEMS_CACHE_DIR', gems.cache_dir_path()),
            check_cache(inductor, 'TORCHINDUCTOR_CACHE_DIR', inductor.cache_dir())]
    generated = Path(gems.code_cache_dir())
    if generated != Path(os.environ['FLAGGEMS_CACHE_DIR']) / 'code_cache' or generated.resolve() != generated:
        raise RuntimeError('Unexpected FlagGems generated-code cache')
    mountinfo = Path('/proc/self/mountinfo').read_text()
    print(json.dumps({'caches': rows, 'mountinfo': mountinfo,
                      'tmp_noexec': bool(os.statvfs('/tmp').f_flag & os.ST_NOEXEC)}, indent=2))


if __name__ == '__main__':
    main()
