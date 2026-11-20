"""Prepare a NEW isolated shca runtime with its peer-memory discovery path fixed.

Experimental, exact-artifact workaround pending a vendor source rebuild. Changes
only one equal-length ELF string; does not force GDR, modify drivers, or replace
the input runtime. CPU-only preparation is NOT evidence of working GPU RDMA.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args()
    source = a.source.resolve(strict=True)
    output = a.output.resolve()
    libraries = ("librccl-net-shca.so", "libibverbs.so.1", "libshca-rdmav34.so",
                 "libucp.so.0", "libucs.so.0", "libucm.so.0", "libuct.so.0")
    expected = "20a0a2a10a6e6a6a55212990634f6de8d79cc0553a315d6a48ff110b114694d0"
    old = b"/sys/kernel/mm/memory_peers/amdkfd/version\0"
    new = b"/sys/kernel/mm/memory_peers/gpukfd/version\0"
    if output.exists() or output == source:
        p.error("output must be a new isolated directory")
    for name in libraries:
        if not (source / name).is_file():
            p.error(f"missing dependency: {name}")
    peer = Path(new[:-1].decode())
    if not peer.is_file() or peer.read_text().strip() != "1.0":
        p.error("the actual gpukfd peer-memory client version 1.0 is required")
    original = (source / libraries[0]).read_bytes()
    if hashlib.sha256(original).hexdigest() != expected:
        p.error("unqualified vendor binary; refuse to patch")
    if len(old) != len(new) or original.count(old) != 1:
        p.error("discovery literal must be unique and equal length")
    patched = original.replace(old, new, 1)
    changes = [i for i, (x, y) in enumerate(zip(original, patched)) if x != y]
    assert len(original) == len(patched) and len(changes) == 3
    metadata = dict(source=str(source), output=str(output),
                    original_sha256=expected,
                    patched_sha256=hashlib.sha256(patched).hexdigest(),
                    changed_offsets=changes, peer_version=peer.read_text().strip(),
                    scope="exact ELF discovery-string fix only; GPU tests required")
    print(json.dumps(metadata, indent=2), flush=True)
    if not a.dry_run:
        output.mkdir()
        for name in libraries:
            shutil.copy2(source / name, output / name)
        (output / libraries[0]).write_bytes(patched)
        assert (output / libraries[0]).read_bytes() == patched
        (output / "run_metadata.txt").write_text(json.dumps(metadata, indent=2) + "\n")
        print("PREPARE_PASS", output, flush=True)
