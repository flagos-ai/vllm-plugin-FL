#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Read-only runner inventory without Torch imports or accelerator allocation.

Only sanitized resource numbers, PCI IDs, runtime names, and identity digests
are emitted. Process IDs, process arguments, account names, environment values,
Docker configuration paths, and raw vendor query output are never retained.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import socket
import stat
import subprocess
import time
from pathlib import Path

QUERY_CAP = 1024 * 1024
PROCESS_CAP = 100000
FD_CAP = 500000


def query(argv):
    try:
        result = subprocess.run(argv, capture_output=True, timeout=20, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, {"available": False, "error_type": type(error).__name__}
    status = {"available": result.returncode == 0, "exit_code": result.returncode}
    if (
        result.returncode != 0
        or len(result.stdout) > QUERY_CAP
        or len(result.stderr) > QUERY_CAP
    ):
        status["within_output_cap"] = False
        return None, status
    return result.stdout.decode("utf-8", "replace"), status


def daemon_id_digest(text):
    """Hash the JSON daemon ID without retaining its raw identity in reports."""
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9:_.-]{1,256}", value):
        return None
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def parse_gmi(text):
    """Parse only overview rows; ignore the vendor's process table entirely."""
    driver_match = re.search(
        r"Driver Version\s*:\s*([0-9]+(?:\.[0-9]+){1,3}(?:-[A-Za-z0-9_.-]+)?)", text
    )
    rows = []
    pattern = re.compile(
        r"^\s*([0-9]+)\s+[^|\n]+\|\s*([0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])"
        r"\s*\|\s*(?:[0-9]+(?:\.[0-9]+)?%|N/A)\s+([0-9]+)MiB\(([0-9]+)MiB\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        ordinal, pci, used, total = match.groups()
        domain, tail = pci.lower().split(":", 1)
        if int(domain, 16) > 0xFFFF:
            continue
        canonical = f"{int(domain, 16):04x}:" + tail
        values = {
            "physical_id": int(ordinal),
            "pci": canonical,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
        }
        if 0 <= values["memory_used_mib"] <= values["memory_total_mib"]:
            rows.append(values)
    if len({row["pci"] for row in rows}) != len(rows) or len(
        {row["physical_id"] for row in rows}
    ) != len(rows):
        rows = []
    return {
        "driver_version": driver_match.group(1) if driver_match else None,
        "devices": rows,
    }


def device_nodes(device_root=Path("/dev"), sys_root=Path("/sys")):
    rows, errors = [], 0
    for path in sorted(device_root.glob("mtgpu.*")):
        if not re.fullmatch(r"mtgpu\.[0-9]+", path.name):
            continue
        try:
            info = path.lstat()
            if not stat.S_ISCHR(info.st_mode):
                raise ValueError("Not a character device")
            pci = (
                (
                    sys_root
                    / "dev/char"
                    / f"{os.major(info.st_rdev)}:{os.minor(info.st_rdev)}"
                    / "device"
                )
                .resolve(strict=True)
                .name
            )
            if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci):
                raise ValueError("Noncanonical PCI identity")
            rows.append(
                {
                    "node_id": int(path.name.split(".")[1]),
                    "pci": pci,
                    "major": os.major(info.st_rdev),
                    "minor": os.minor(info.st_rdev),
                }
            )
        except (OSError, ValueError):
            errors += 1
    return rows, errors


def fd_counts(nodes, proc_root=Path("/proc")):
    """Count only links to observed character-device nodes, with explicit caps."""
    expected = {f"/dev/mtgpu.{row['node_id']}": row["node_id"] for row in nodes}
    counts = {row["node_id"]: 0 for row in nodes}
    process_count = fd_count = inaccessible = vanished = 0
    capped = False
    for process in proc_root.iterdir():
        if not process.name.isdecimal():
            continue
        process_count += 1
        if process_count > PROCESS_CAP:
            capped = True
            break
        try:
            for descriptor in (process / "fd").iterdir():
                fd_count += 1
                if fd_count > FD_CAP:
                    capped = True
                    break
                try:
                    target = os.readlink(descriptor)
                except FileNotFoundError:
                    vanished += 1
                    continue
                except OSError:
                    inaccessible += 1
                    continue
                if target in expected:
                    counts[expected[target]] += 1
            if capped:
                break
        except FileNotFoundError:
            vanished += 1
        except OSError:
            inaccessible += 1
    return {
        "open_fd_counts": counts,
        "scanned_process_count": process_count,
        "scanned_fd_count": fd_count,
        "inaccessible_entry_count": inaccessible,
        "vanished_entry_count": vanished,
        "limit_reached": capped,
        "scan_complete": not capped and inaccessible == 0,
    }


def candidate_devices(nodes, gmi, fd_report):
    """Propose only empty-memory nodes with a complete, zero-holder FD scan."""
    if not fd_report["scan_complete"]:
        return []
    node_map = {row["pci"]: row for row in nodes}
    candidates = []
    for row in gmi["devices"]:
        node = node_map.get(row["pci"])
        if (
            node
            and row["memory_used_mib"] <= 2
            and fd_report["open_fd_counts"].get(node["node_id"]) == 0
        ):
            candidates.append(
                {
                    "physical_id": row["physical_id"],
                    "pci": row["pci"],
                    "node_id": node["node_id"],
                }
            )
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    nodes, node_errors = device_nodes()
    fd_report = fd_counts(nodes)
    gmi_text, gmi_status = query(["mthreads-gmi"])
    gmi = (
        parse_gmi(gmi_text)
        if gmi_text is not None
        else {"driver_version": None, "devices": []}
    )
    runtime_text, runtime_status = query(
        ["docker", "info", "--format", "{{json .Runtimes}}"]
    )
    runtimes = []
    if runtime_text is not None:
        try:
            value = json.loads(runtime_text)
            if isinstance(value, dict):
                runtimes = sorted(
                    name for name in value if re.fullmatch(r"[A-Za-z0-9_.-]+", name)
                )
        except (ValueError, TypeError):
            runtime_status["parse_ok"] = False
    identity_text, identity_status = query(
        ["docker", "info", "--format", "{{json .ID}}"]
    )
    identity_digest = None
    if identity_text is not None:
        identity_digest = daemon_id_digest(identity_text)
        identity_status["parse_ok"] = identity_digest is not None
    result = {
        "schema_version": 1,
        "probe_time_utc_seconds": time.time(),
        "hostname_sha256": hashlib.sha256(
            socket.gethostname().encode("utf-8")
        ).hexdigest(),
        "scope": "Read-only runner inventory; no accelerator allocation or signals",
        "nodes": nodes,
        "node_error_count": node_errors,
        "fd_scan": fd_report,
        "vendor_query_status": gmi_status,
        "vendor_resources": gmi,
        "docker_query_status": runtime_status,
        "docker_runtime_names": runtimes,
        "docker_daemon_query_status": identity_status,
        "docker_daemon_id_sha256": identity_digest,
        "candidate_idle_devices": candidate_devices(nodes, gmi, fd_report),
        "allocation_reserved": False,
        "GPU_Gate_passed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
