#!/usr/bin/env python3
"""Stage and verify the existing pinned ordinary Gate observer in the CI venv."""

import argparse
import sys
import sysconfig
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dependency_profile import (
    installed_wheel,
    read,
    require,
    save,
    save_json,
    wheel_review,
)

SOURCE = Path("/opt/musa/wheelhouse/musa_gate_observer-0.1.0-py3-none-any.whl")
PIN = {
    "bytes": 25288,
    "sha256": "f14557735efec09cb9ae1332c89435e40057e72ceb12e5b84b905b3f936cc5a2",
}
EXPECTED = {"name": "musa-gate-observer", "version": "0.1.0"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("stage", "installed"))
    parser.add_argument("--task-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.task_dir
    require(
        root.is_absolute() and root.resolve(strict=True) == root, "Canonical task root"
    )
    prefix = root / "venv"
    site = Path(sysconfig.get_path("purelib")).resolve(strict=True)
    require(
        Path(sys.prefix).resolve(strict=True) == prefix
        and site.is_relative_to(prefix)
        and sys.flags.isolated
        and not sys.flags.no_site,
        "Private normal-site CI interpreter",
    )
    copied = root / "wheels" / SOURCE.name
    source = SOURCE if args.phase == "stage" else copied
    raw, fingerprint = read(source, 128 * 1024, PIN)
    reviewed = wheel_review(raw, fingerprint, EXPECTED)
    if args.phase == "stage":
        reviewed.update(save(copied, raw))
        result = {
            "source": fingerprint,
            "wheel": reviewed,
            "scope": "Pinned ordinary-wheel static verification",
            "native_or_model_acceptance": False,
        }
    else:
        result = {
            "installed": installed_wheel(reviewed, site, prefix),
            "scope": "Actual private source/RECORD/direct_url verification",
            "native_or_model_acceptance": False,
        }
    save_json(root / ("gate-observer-" + args.phase + ".json"), result)
    return 0


if __name__ == "__main__":
    if not __debug__:
        raise RuntimeError(
            "Run without Python -O; wheel and private-origin guards are required"
        )
    raise SystemExit(main())
