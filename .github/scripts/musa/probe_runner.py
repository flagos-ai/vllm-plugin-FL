#!/usr/bin/env python3
# Copyright (c) 2026 BAAI. All rights reserved.
"""Read the selected device overview before the CI interpreter initializes MUSA."""

import re
import subprocess

QUERY_CAP = 1024 * 1024


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


def parse_gmi(text):
    """Parse only overview rows; ignore the vendor's process table entirely."""
    driver_match = re.search(
        r"Driver Version\s*:\s*([0-9]+(?:\.[0-9]+){1,3}(?:-[A-Za-z0-9_.-]+)?)", text
    )
    rows = []
    pattern = re.compile(
        r"^\s*([0-9]+)\s+[^|\n]+\|\s*([0-9a-fA-F]{4,8}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])"
        r"\s*\|\s*([0-9]+(?:\.[0-9]+)?%|N/A)\s+([0-9]+)MiB\(([0-9]+)MiB\)",
        re.MULTILINE,
    )
    for match in pattern.finditer(text):
        ordinal, pci, utilization, used, total = match.groups()
        domain, tail = pci.lower().split(":", 1)
        if int(domain, 16) > 0xFFFF:
            continue
        canonical = f"{int(domain, 16):04x}:" + tail
        values = {
            "physical_id": int(ordinal),
            "pci": canonical,
            "memory_used_mib": int(used),
            "memory_total_mib": int(total),
            "utilization_percent": None
            if utilization == "N/A"
            else float(utilization[:-1]),
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


def require_idle_devices(resources, expected_pci):
    """Require four unique selected PCI devices with known zero usage."""
    if (
        not isinstance(expected_pci, (list, tuple))
        or len(expected_pci) != 4
        or any(
            not isinstance(pci, str)
            or not re.fullmatch(r"0000:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", pci)
            for pci in expected_pci
        )
        or len(set(expected_pci)) != 4
    ):
        raise ValueError("MUSA requires four unique canonical selected PCI IDs")
    devices = resources.get("devices") if isinstance(resources, dict) else None
    if not isinstance(devices, list):
        raise ValueError("MUSA resource overview is unavailable")
    selected = {}
    for row in devices:
        if not isinstance(row, dict):
            raise ValueError("Invalid MUSA resource overview")
        pci = row.get("pci")
        if pci not in expected_pci:
            continue
        if pci in selected:
            raise ValueError("Duplicate selected MUSA PCI device")
        selected[pci] = row
    if len(selected) != 4:
        raise ValueError("Incomplete selected MUSA resource overview")
    rows = [selected[pci] for pci in expected_pci]
    if (
        any(type(row.get("physical_id")) is not int for row in rows)
        or len({row["physical_id"] for row in rows}) != 4
    ):
        raise ValueError("Selected MUSA device IDs are not unique")
    for row in rows:
        total = row.get("memory_total_mib")
        used = row.get("memory_used_mib")
        utilization = row.get("utilization_percent")
        if (
            type(total) is not int
            or total <= 0
            or type(used) is not int
            or used != 0
            or type(utilization) not in (int, float)
            or utilization != 0
        ):
            raise ValueError("Selected MUSA device is occupied or usage is unknown")
    return rows
