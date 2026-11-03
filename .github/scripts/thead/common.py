# Copyright 2026 FlagOS Contributors
"""Common CI receipt and strict JUnit handling; importing this needs no accelerator."""

import json
import os
import signal
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf8")


def process_identity(pid):
    try:
        fields = (
            (Path("/proc") / str(pid) / "stat").read_text().split(") ", 1)[1].split()
        )
    except FileNotFoundError:
        return None
    return {
        "pid": pid,
        "state": fields[0],
        "pgid": int(fields[2]),
        "sid": int(fields[3]),
        "start_ticks": int(fields[19]),
    }


def bind_process(process):
    identity = process_identity(process.pid)
    if (
        identity is None
        or identity["pgid"] != process.pid
        or identity["sid"] != process.pid
    ):
        raise RuntimeError("CI child did not create its own process session")
    process.thead_identity = identity
    return identity


def session_members(identity, rows):
    """Pure identity check used immediately before signaling the owned group."""
    leader = next((row for row in rows if row["pid"] == identity["pid"]), None)
    if leader is not None and any(
        leader[key] != identity[key] for key in ("pgid", "sid", "start_ticks")
    ):
        raise RuntimeError("Child PID was reused; refuse to signal a different session")
    members = [
        row
        for row in rows
        if row["sid"] == identity["sid"]
        and row["start_ticks"] >= identity["start_ticks"]
        and row["state"] != "Z"
    ]
    if any(row["pgid"] != identity["pgid"] for row in members):
        raise RuntimeError(
            "Own session has a live child in another process group; cleanup incomplete"
        )
    return members


def live_session_members(identity):
    rows = []
    for directory in Path("/proc").iterdir():
        if directory.name.isdigit():
            row = process_identity(int(directory.name))
            if row is not None:
                rows.append(row)
    return session_members(identity, rows)


def stop_process_group(process):
    """Signal only the fresh PGID/SID captured for this actual Popen child."""
    identity = process.thead_identity
    for sig, wait in ((signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        if not live_session_members(identity):
            process.poll()
            return
        try:
            os.killpg(identity["pgid"], sig)
        except ProcessLookupError:
            process.poll()
            return
        end = time.monotonic() + wait
        while time.monotonic() < end:
            if not live_session_members(identity):
                process.poll()
                return
            time.sleep(0.1)
    raise RuntimeError("Own process session did not finish cleanup")


def run_command(
    argv, directory, *, timeout, env=None, cwd=None, cap=32 * 1024**2, monitor=None
):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    receipt = {
        "argv": argv,
        "process_started": False,
        "cleanup_complete": True,
        "exit": None,
        "timeout": False,
        "cap_exceeded": False,
        "errors": [],
    }
    process = None
    start = time.monotonic()
    try:
        with (
            (directory / "stdout.log").open("xb") as stdout,
            (directory / "stderr.log").open("xb") as stderr,
        ):
            process = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                env=env,
                cwd=cwd,
                start_new_session=True,
            )
            receipt.update(
                process_started=True, cleanup_complete=False, pid=process.pid
            )
            receipt["identity"] = bind_process(process)
            while process.poll() is None:
                if monitor is not None:
                    monitor()
                if time.monotonic() - start > timeout:
                    receipt["timeout"] = True
                    break
                if any(
                    (directory / name).stat().st_size > cap
                    for name in ("stdout.log", "stderr.log")
                ):
                    receipt["cap_exceeded"] = True
                    break
                time.sleep(0.1)
    except Exception as exc:
        receipt["errors"].append(repr(exc))
    finally:
        if process is not None:
            try:
                stop_process_group(process)
                receipt["cleanup_complete"] = True
            except Exception as exc:
                receipt["errors"].append(repr(exc))
            receipt["exit"] = process.poll()
        receipt["seconds"] = time.monotonic() - start
        receipt["cap_exceeded"] |= any(
            (directory / name).exists() and (directory / name).stat().st_size > cap
            for name in ("stdout.log", "stderr.log")
        )
        receipt["clean"] = bool(
            receipt["process_started"]
            and receipt["cleanup_complete"]
            and receipt["exit"] == 0
            and not receipt["timeout"]
            and not receipt["cap_exceeded"]
            and not receipt["errors"]
        )
        save_json(directory / "receipt.json", receipt)
    return receipt


def validate_junit(path, expected):
    root = ET.parse(path).getroot()
    cases = root.findall(".//testcase")
    identities = {(case.get("classname"), case.get("name")) for case in cases}
    bad = [
        case
        for case in cases
        if any(case.find(tag) is not None for tag in ("failure", "error", "skipped"))
    ]
    if len(cases) != expected or len(identities) != expected or bad:
        raise RuntimeError(
            f"Expected {expected} unique passes with zero skip/fail/error; "
            f"got {len(cases)} cases, {len(identities)} unique, {len(bad)} nonpasses"
        )
    return {"passed": len(cases), "failed": 0, "skipped": 0}
