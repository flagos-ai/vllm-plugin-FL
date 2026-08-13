#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-FL project
#
# Small helper to work around a CANN 8.5.x bug in extract_host_stub.py:
# compile_commands.json may contain object paths with a leading "./" while
# the object paths passed to extract_host_stub.py are normalized. The lookup
# of enable_ascendc_time_stamp then raises KeyError.
#
# This script is invoked by setup.py immediately before building AscendC
# kernels and is asked to restore the original file afterwards.

import argparse
import os
import shutil
import sys
from pathlib import Path


PATCH_MARKER = "# vllm-fl: normalize path before enable_ascendc_time_stamp lookup"


def _get_ascend_home() -> str:
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if ascend_home and os.path.isdir(ascend_home):
        return ascend_home
    default = "/usr/local/Ascend/ascend-toolkit/latest"
    if os.path.isdir(default):
        return default
    raise RuntimeError(
        "ASCEND_HOME_PATH is not set and the default CANN path does not exist."
    )


def find_extract_host_stub() -> Path:
    """Locate the CANN extract_host_stub.py that matches the current env."""
    ascend_home = _get_ascend_home()
    # Architecture names used by CANN toolkit installations.
    arch_dirs = ["tools", "compiler", "aarch64-linux", "x86_64-linux"]
    for arch in arch_dirs:
        candidate = Path(ascend_home) / arch / "tikcpp" / "ascendc_kernel_cmake" / "legacy_modules" / "util" / "extract_host_stub.py"
        if candidate.is_file():
            return candidate.resolve()
        # Some installations use ascendc_devkit layout.
        candidate2 = Path(ascend_home) / arch / "ascendc_devkit" / "tikcpp" / "samples" / "cmake" / "util" / "extract_host_stub.py"
        if candidate2.is_file():
            return candidate2.resolve()
    raise RuntimeError(
        f"Could not find extract_host_stub.py under ASCEND_HOME_PATH={ascend_home}. "
        "Please check your CANN installation."
    )


def _original_lookup_block_get() -> str:
    """CANN 8.5.0 style: uses dict.get(...)."""
    return """    for func_group in func_sign_groups:
        enbale_flag = enable_ascendc_time_stamp.get(func_group.filepath)
        if enbale_flag is True:"""


def _original_lookup_block_subscript() -> str:
    """CANN 8.5.1 style: uses direct dict subscription."""
    return """    for func_group in func_sign_groups:
        enbale_flag = enable_ascendc_time_stamp[func_group.filepath]
        if enbale_flag is True:"""


def _patched_lookup_block_get() -> str:
    """Replacement for the .get(...) style block."""
    return """    for func_group in func_sign_groups:
        {marker}
        enbale_flag = enable_ascendc_time_stamp.get(func_group.filepath)
        if enbale_flag is None:
            enbale_flag = enable_ascendc_time_stamp.get(os.path.normpath(func_group.filepath))
        if enbale_flag is None:
            enbale_flag = enable_ascendc_time_stamp.get(os.path.abspath(func_group.filepath))
        if enbale_flag is True:""".format(marker=PATCH_MARKER)


def _patched_lookup_block_subscript() -> str:
    """Replacement for the direct subscription style block."""
    return """    for func_group in func_sign_groups:
        {marker}
        _filepath = func_group.filepath
        if _filepath in enable_ascendc_time_stamp:
            enbale_flag = enable_ascendc_time_stamp[_filepath]
        elif os.path.normpath(_filepath) in enable_ascendc_time_stamp:
            enbale_flag = enable_ascendc_time_stamp[os.path.normpath(_filepath)]
        elif os.path.abspath(_filepath) in enable_ascendc_time_stamp:
            enbale_flag = enable_ascendc_time_stamp[os.path.abspath(_filepath)]
        else:
            enbale_flag = False
        if enbale_flag is True:""".format(marker=PATCH_MARKER)


def apply_patch(backup_path: Path | None = None) -> Path:
    """Apply the workaround patch and return the path to the backup file."""
    target = find_extract_host_stub()
    original_text = target.read_text(encoding="utf-8")

    if PATCH_MARKER in original_text:
        print(f"extract_host_stub.py already patched ({target}); skipping apply.")
        # No backup needed because we did not change anything.
        return Path("__already_patched__")

    # CANN 9.0+ stores keys as absolute_obj, so the 8.5.x path-normalization
    # workaround is no longer required.
    if "enable_ascendc_time_stamp[absolute_obj]" in original_text:
        print(f"extract_host_stub.py uses absolute_obj keys ({target}); skipping apply.")
        return Path("__already_patched__")

    old_get = _original_lookup_block_get()
    old_sub = _original_lookup_block_subscript()
    if old_get in original_text:
        old_block, new_block = old_get, _patched_lookup_block_get()
    elif old_sub in original_text:
        old_block, new_block = old_sub, _patched_lookup_block_subscript()
    else:
        raise RuntimeError(
            f"Could not find the expected lookup block in {target}. "
            "The CANN version may have changed; please review the patch script."
        )

    if backup_path is None:
        backup_path = target.with_suffix(target.suffix + ".vllm_fl_bak")
    else:
        backup_path = Path(backup_path)

    shutil.copy2(target, backup_path)

    patched_text = original_text.replace(old_block, new_block, 1)
    target.write_text(patched_text, encoding="utf-8")
    print(f"Patched {target}")
    print(f"Backup saved to {backup_path}")
    return backup_path


def restore_patch(backup_path: Path | None = None) -> None:
    """Restore the original file from backup and remove the backup."""
    target = find_extract_host_stub()

    if backup_path is None:
        backup_path = target.with_suffix(target.suffix + ".vllm_fl_bak")
    else:
        backup_path = Path(backup_path)

    if not backup_path.is_file():
        if PATCH_MARKER in target.read_text(encoding="utf-8"):
            print(
                f"Warning: backup {backup_path} missing, cannot restore {target}",
                file=sys.stderr,
            )
        return

    shutil.copy2(backup_path, target)
    backup_path.unlink()
    print(f"Restored {target} and removed backup {backup_path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply/restore a small workaround patch for CANN extract_host_stub.py"
    )
    parser.add_argument(
        "action",
        choices=["apply", "restore"],
        help="Apply the patch or restore the original file.",
    )
    parser.add_argument(
        "--backup",
        type=str,
        default=None,
        help="Path to use for the backup file. Defaults to a file next to the target.",
    )
    args = parser.parse_args()

    backup_path = Path(args.backup) if args.backup else None

    if args.action == "apply":
        apply_patch(backup_path)
    else:
        restore_patch(backup_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
