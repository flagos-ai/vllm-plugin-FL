#!/usr/bin/env python3
"""Copy one pinned empty wheel without following links or using network access."""
import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import stat
import zipfile


def copy_wheel(source, expected_sha, destination):
    source = Path(source)
    destination = Path(destination)
    if not source.is_absolute() or source.suffix != '.whl':
        raise ValueError('MUSA_CI_EMPTY_WHEEL must name an absolute wheel path')
    with os.fdopen(os.open(source, os.O_RDONLY | os.O_NOFOLLOW), 'rb') as reader:
        before = os.fstat(reader.fileno())
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size < 128 * 1024**2:
            raise ValueError('Unexpected wheel file type or size')
        body = reader.read()
        after = os.fstat(reader.fileno())
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
    if identity(before) != identity(after) or len(body) != before.st_size:
        raise ValueError('Wheel changed during read')
    actual_sha = hashlib.sha256(body).hexdigest()
    if actual_sha != expected_sha:
        raise ValueError('Empty wheel SHA256 mismatch')
    import io
    with zipfile.ZipFile(io.BytesIO(body)) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate ZIP members')
        metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
        if len(metadata) != 1:
            raise ValueError('Expected exactly one wheel distribution')
        message = email.message_from_bytes(archive.read(metadata[0]))
        if message['Name'].lower().replace('-', '_') != 'vllm' or message['Version'] != '0.28.0+empty':
            raise ValueError('Expected vllm 0.28.0+empty')
        if any(n.endswith(('.so', '.pyd')) for n in names):
            raise ValueError('Empty wheel contains native libraries')
    target = destination / source.name
    with target.open('xb') as writer:
        writer.write(body)
    return {'source': str(source), 'copy': str(target), 'bytes': len(body),
            'sha256': actual_sha, 'name': message['Name'], 'version': message['Version']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source')
    parser.add_argument('sha256')
    parser.add_argument('destination')
    args = parser.parse_args()
    print(json.dumps(copy_wheel(args.source, args.sha256, args.destination), indent=2))
