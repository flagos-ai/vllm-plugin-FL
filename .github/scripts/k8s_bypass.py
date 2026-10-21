#!/usr/bin/env python3
"""Run a command inside a self-orchestrated GPU pod, bypassing the pool's
broken container hook (k8s-novolume/index.js loses its promise chain after
the readiness wait and exits 0 without writing the protocol response file).

Everything here uses only the runner pod's own service account, which the
investigation (PR #562) verified allows pods create/get/list/delete,
pods/log and pods/exec in arc-runners.  Flow: create pod -> wait Ready ->
stream the workspace in as a tar over exec -> run the command with output
relayed until completion -> always delete the pod.
"""

from __future__ import annotations

import argparse
import base64
import http.client
import io
import json
import os
import re
import ssl
import struct
import sys
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"
JOB_CONTAINER = "job"


def load_sa():
    with open(f"{SA_DIR}/namespace") as f:
        ns = f.read().strip()
    with open(f"{SA_DIR}/token") as f:
        tok = f.read().strip()
    ctx = ssl.create_default_context(cafile=f"{SA_DIR}/ca.crt")
    host = os.environ["KUBERNETES_SERVICE_HOST"]
    port = os.environ["KUBERNETES_SERVICE_PORT_HTTPS"]
    return ns, tok, ctx, host, int(port)


class K8s:
    def __init__(self):
        self.ns, self.tok, self.ctx, self.host, self.port = load_sa()
        # The pod egress proxy cannot route to the in-cluster API; go direct.
        urllib.request.install_opener(
            urllib.request.build_opener(urllib.request.ProxyHandler({}))
        )

    def api(self, method, path, body=None):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"https://{self.host}:{self.port}{path}",
            data=data,
            method=method,
            headers={
                "Authorization": "Bearer " + self.tok,
                "Content-Type": "application/json",
            },
        )
        try:
            r = urllib.request.urlopen(req, context=self.ctx, timeout=30)
            return r.status, r.read().decode(errors="replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode(errors="replace")

    def exec_ws(self, pod, args, stdin_payload=None, read_budget=300, quiet=30):
        """Exec via the v4 channel protocol.  Returns (stdout, stderr, status_json)."""
        path = (
            f"/api/v1/namespaces/{self.ns}/pods/{pod}/exec?container={JOB_CONTAINER}"
            + "".join("&command=" + urllib.parse.quote(c, safe="") for c in args)
            + "&stdout=true&stderr=true&stdin="
            + ("true" if stdin_payload is not None else "false")
        )
        conn = http.client.HTTPSConnection(
            self.host, self.port, context=self.ctx, timeout=30
        )
        key = base64.b64encode(os.urandom(16)).decode()
        conn.request(
            "GET",
            path,
            headers={
                "Authorization": "Bearer " + self.tok,
                "Connection": "Upgrade",
                "Upgrade": "websocket",
                "Sec-WebSocket-Key": key,
                "Sec-WebSocket-Version": "13",
                "Sec-WebSocket-Protocol": "v5.channel.k8s.io,v4.channel.k8s.io,v3.channel.k8s.io,"
                "v2.channel.k8s.io,channel.k8s.io",
            },
        )
        r = conn.getresponse()
        if r.status != 101:
            body = r.read(300).decode(errors="replace")
            raise RuntimeError(f"exec upgrade failed: {r.status} {body}")
        sock = conn.sock
        if stdin_payload is not None:
            self._cframe(sock, b"\x00" + stdin_payload)
        # Do NOT send a close frame before reading: the server then drops all output.
        out = {1: b"", 2: b"", 3: b""}
        buf = b""
        deadline = time.time() + read_budget
        sock.settimeout(quiet)
        while time.time() < deadline:
            try:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk
            except TimeoutError:
                continue
            except Exception:
                break
            while len(buf) >= 2:
                op = buf[0] & 0x0F
                ln = buf[1] & 0x7F
                off = 2
                if ln == 126:
                    if len(buf) < 4:
                        break
                    ln = int.from_bytes(buf[2:4], "big")
                    off = 4
                elif ln == 127:
                    if len(buf) < 10:
                        break
                    ln = int.from_bytes(buf[2:10], "big")
                    off = 10
                if len(buf) < off + ln:
                    break
                payload = buf[off : off + ln]
                buf = buf[off + ln :]
                if op == 0x2 and payload:
                    out[payload[0]] = out.get(payload[0], b"") + payload[1:]
                elif op == 0x8:
                    buf = b""
                    break
        conn.close()
        return (
            out[1].decode(errors="replace"),
            out[2].decode(errors="replace"),
            out[3].decode(errors="replace"),
        )

    @staticmethod
    def _cframe(sock, payload):
        mask = os.urandom(4)
        ln = len(payload)
        if ln < 126:
            hdr = bytes([0x82, 0x80 | ln])
        elif ln < 65536:
            hdr = bytes([0x82, 0x80 | 126]) + struct.pack(">H", ln)
        else:
            hdr = bytes([0x82, 0x80 | 127]) + struct.pack(">Q", ln)
        rep = (mask * (ln // 4 + 1))[:ln]
        masked = (int.from_bytes(payload, "big") ^ int.from_bytes(rep, "big")).to_bytes(
            ln, "big"
        )
        sock.sendall(hdr + mask + masked)


def pod_spec(name, image, gpus, model_hostpath, pin_nodes, deadline_s=None):
    container = {
        "name": JOB_CONTAINER,
        "image": image,
        "command": ["sleep", "7200"],
        "workingDir": "/__w",
        "volumeMounts": [{"name": "work", "mountPath": "/__w"}],
    }
    if gpus:
        container["resources"] = {"limits": {"nvidia.com/gpu": gpus}}
    volumes = [{"name": "work", "emptyDir": {}}]
    if model_hostpath:
        # The node's model store, read-only: e2e consumes models the pool
        # owners pre-stage (same contract as container pools), pointed at
        # by device overrides in tests/platforms/<platform>.yaml.
        container["volumeMounts"].append(
            {"name": "model", "readOnly": True, "mountPath": "/flagcicd/model"}
        )
        volumes.append({"name": "model", "hostPath": {"path": model_hostpath}})
    pod = {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {"name": name, "namespace": None, "labels": {"app": "fl-bypass"}},
        "spec": {
            "containers": [container],
            "volumes": volumes,
            "restartPolicy": "Never",
            # Mirror the CI container_options' --ipc=host: the default 64MiB
            # pod /dev/shm is too small for vLLM's shared-memory requirements.
            "hostIPC": True,
            # Reap the pod even if the runner process is killed (job timeout
            # / cancel skips the finally-delete below) - otherwise a
            # `sleep 7200` pod holds its GPUs for up to two hours and
            # starves every later pinned pod.  Counts from scheduling, so
            # queue time does not eat into it.
            "activeDeadlineSeconds": deadline_s,
        },
    }
    if pin_nodes:
        # Pool nodes are not homogeneous: pods scheduled on prod-009 died
        # (phase=Failed within ~20s of ContainerCreating, no container
        # statuses; root cause not yet captured - the events dump on the
        # failure paths is there to catch it).  Pin mount-carrying pods to
        # nodes observed green: an Unschedulable pod queues visibly for
        # the ready budget instead of dying after scheduling.
        pod["spec"]["affinity"] = {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "kubernetes.io/hostname",
                                    "operator": "In",
                                    "values": pin_nodes,
                                }
                            ]
                        }
                    ]
                }
            }
        }
    return pod


def _write_back(name: str, lines: list[str]) -> None:
    """Write one copy-back file locally, preserving its basename only."""
    import pathlib

    out = pathlib.Path(pathlib.PurePosixPath(name).name)
    out.write_text("\n".join(lines) + "\n")
    print(f"[bypass] copy-back: {out} ({out.stat().st_size} bytes)", file=sys.stderr)


def exec_exit(status_json: str) -> int | None:
    """Extract the in-pod command's exit code from the WS status frame.

    Returns None when no status frame arrived (read budget exhausted or the
    channel closed prematurely) - callers must treat that as a failure, not 0.
    """
    try:
        j = json.loads(status_json)
    except ValueError:
        j = {}
    if j.get("status") == "Success":
        return 0
    m = re.search(r"exit code (\d+)", status_json)
    return int(m.group(1)) if m else 1


def tar_dir(path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for root, _dirs, files in os.walk(path):
            for f in files:
                full = os.path.join(root, f)
                tf.add(full, arcname=os.path.relpath(full, path))
    return buf.getvalue()


TERMINAL_WAIT_REASONS = (
    "ErrImagePull",
    "ImagePullBackOff",
    "CrashLoopBackOff",
    "CreateContainerError",
)


def pod_snapshot(status: dict, spec: dict) -> str:
    """One-line, human-readable digest of why a pod is or is not Ready."""
    conds = {c.get("type"): c for c in status.get("conditions", [])}
    sched = conds.get("PodScheduled", {})
    waiting = [
        (c.get("state") or {}).get("waiting", {}).get("reason")
        for c in status.get("containerStatuses", [])
    ]
    parts = [
        f"phase={status.get('phase', '?')}",
        f"scheduled={sched.get('status', '?')}"
        + (f" ({sched.get('reason')})" if sched.get("reason") else ""),
        f"node={spec.get('nodeName') or '-'}",
    ]
    if any(waiting):
        parts.append("waiting=" + ",".join(r for r in waiting if r))
    return " ".join(parts)


def dump_pod_events(k: K8s, pod: str) -> None:
    """Print the pod's k8s events next to a failure.

    Mount/scheduling failures (FailedMount and friends) are recorded as
    events, not in pod.status - and the pod is deleted immediately after
    wait_ready gives up, so this is the only moment they can be captured.
    """
    code, txt = k.api(
        "GET",
        f"/api/v1/namespaces/{k.ns}/events?fieldSelector=involvedObject.name={pod}",
    )
    if code != 200:
        print(
            f"[bypass] event dump unavailable (HTTP {code} - likely RBAC)",
            file=sys.stderr,
        )
        return
    for ev in json.loads(txt).get("items", []):
        print(
            f"[bypass] event: {ev.get('reason')}: "
            f"{(ev.get('message') or '').strip()[:300]}",
            file=sys.stderr,
        )


def container_failure_detail(status: dict) -> str:
    """Best-effort reason dump for a pod that died before Ready.

    The pod is deleted right after wait_ready gives up on it, so this is
    the only place its container states and false conditions can be
    captured into the job log.
    """
    lines = []
    for c in status.get("containerStatuses", []) or []:
        for kind, st in (c.get("state") or {}).items():
            desc = f"{c.get('name')}: {st.get('reason') or kind}"
            if st.get("message"):
                desc += f": {st['message']}"
            lines.append(desc)
    for c in status.get("conditions", []) or []:
        if c.get("status") == "False" and c.get("reason"):
            lines.append(f"cond {c.get('type')}: {c.get('reason')}")
    return "; ".join(lines) or "no container statuses reported"


def wait_ready(k: K8s, pod: str, budget_s: float) -> tuple[bool, float]:
    """Wait for the pod to become Ready, queueing like a runner would.

    A pod that is Unschedulable or still pulling its image is WAITING,
    not broken - it keeps the full budget (the cluster is queueing it;
    a normal runner would keep the job queued too).  Terminal container
    states fail fast.  Every state transition is printed so a timeout
    is self-explaining in the job log.
    """
    t0 = time.time()
    last = None
    while time.time() - t0 < budget_s:
        code, txt = k.api("GET", f"/api/v1/namespaces/{k.ns}/pods/{pod}")
        if code == 200:
            obj = json.loads(txt)
            status, spec = obj.get("status", {}), obj.get("spec", {})
            snap = pod_snapshot(status, spec)
            if snap != last:
                print(f"[bypass] pod status: {snap}", flush=True)
                last = snap
            phase = status.get("phase")
            if phase in ("Failed", "Succeeded"):
                print(
                    f"FATAL pod reached phase={phase} before Ready: "
                    f"{container_failure_detail(status)}",
                    file=sys.stderr,
                )
                dump_pod_events(k, pod)
                return False, time.time() - t0
            conds = {
                c.get("type"): c.get("status") for c in status.get("conditions", [])
            }
            if conds.get("Ready") == "True":
                return True, time.time() - t0
            waiting = [
                (c.get("state") or {}).get("waiting", {})
                for c in status.get("containerStatuses", [])
            ]
            if any(w.get("reason") in TERMINAL_WAIT_REASONS for w in waiting):
                print(f"FATAL container state: {waiting}", file=sys.stderr)
                dump_pod_events(k, pod)
                return False, time.time() - t0
        time.sleep(2)
    return False, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--image", default="harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-cuda-ci"
    )
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument(
        "--command", required=True, help="shell command to run inside the pod"
    )
    ap.add_argument(
        "--workspace",
        default=os.environ.get("GITHUB_WORKSPACE"),
        help="local directory streamed into the pod at /__w",
    )
    ap.add_argument(
        "--model-hostpath",
        default="/mnt/airs-business/airs/sharefs/"
        "d76ef6fc-7ce3-4da0-95fc-bfcc95895aa2/ff697f3c-5782-4479-9e17-de01ec823d57/MODEL",
    )
    ap.add_argument(
        "--pin-nodes",
        default="",
        help="comma-separated kubernetes.io/hostname values the pod must "
        "schedule on; use for pods whose model hostPath exists only on "
        "some pool nodes",
    )
    ap.add_argument("--pod-prefix", default="fl-bypass")
    ap.add_argument(
        "--copy-back",
        default="",
        help="comma-separated glob patterns of text files produced inside "
        "the pod (relative to /__w) to write back to the local workspace",
    )
    ap.add_argument(
        "--ready-timeout",
        type=int,
        default=1800,
        help="total budget for the pod to become Ready; an Unschedulable "
        "(queueing) pod uses the whole budget before failing",
    )
    ap.add_argument("--read-budget", type=int, default=2700)
    args = ap.parse_args()

    k = K8s()
    pod = f"{args.pod_prefix}-{os.environ.get('GITHUB_RUN_ID', 'local')}-{int(time.time()) % 100000}"
    spec = pod_spec(
        pod,
        args.image,
        args.gpus,
        args.model_hostpath,
        [n.strip() for n in args.pin_nodes.split(",") if n.strip()],
        deadline_s=args.ready_timeout + args.read_budget + 600,
    )
    spec["metadata"]["namespace"] = k.ns
    code, txt = k.api("POST", f"/api/v1/namespaces/{k.ns}/pods", spec)
    if code != 201:
        print(f"FATAL create pod: {code} {txt[:300]}", file=sys.stderr)
        return 2
    print(f"[bypass] pod {pod} created", flush=True)
    try:
        ready, waited = wait_ready(k, pod, args.ready_timeout)
        if not ready:
            print(
                f"FATAL pod not Ready within {args.ready_timeout}s "
                f"(last status above; Unschedulable means the cluster is "
                f"queueing the pod for resources)",
                file=sys.stderr,
            )
            # Mount/admission failures (FailedMount and friends) surface
            # here and ONLY here - the pod never reached a terminal phase,
            # so the fast-fail paths in wait_ready never fired.
            dump_pod_events(k, pod)
            return 2
        print(f"[bypass] pod Ready after {waited:.1f}s", flush=True)

        if args.workspace and os.path.isdir(args.workspace):
            t0 = time.time()
            payload = tar_dir(args.workspace)
            so, se, st = k.exec_ws(
                pod,
                ["sh", "-c", "tar xf - -C /__w 2>/dev/null; find /__w -type f | wc -l"],
                stdin_payload=payload,
                read_budget=600,
            )
            print(
                f"[bypass] workspace streamed ({len(payload)} bytes, "
                f"{time.time() - t0:.1f}s), files: {so.strip()}",
                flush=True,
            )
            if (rc := exec_exit(st)) != 0:
                print(f"FATAL workspace extract failed: {st[:200]}", file=sys.stderr)
                return 2

        t0 = time.time()
        so, se, st = k.exec_ws(
            pod, ["sh", "-c", args.command], read_budget=args.read_budget
        )
        print(so, end="")
        if se:
            print(se, file=sys.stderr, end="")
        rc = exec_exit(st)
        print(
            f"\n[bypass] command finished in {time.time() - t0:.1f}s, "
            f"exec status: {st[:200]}, exit code: {rc}",
            file=sys.stderr,
            flush=True,
        )
        if rc is None:
            print(
                "FATAL no exit status received (read budget exhausted or channel "
                "closed prematurely); treating as failure",
                file=sys.stderr,
            )
            return 1
        if args.copy_back:
            patterns = " ".join(p.strip() for p in args.copy_back.split(","))
            so, se, _ = k.exec_ws(
                pod,
                [
                    "sh",
                    "-c",
                    f'for f in {patterns}; do [ -f "$f" ] && printf "\\n===FILE:%s\\n" "$f" && cat "$f"; done',
                ],
                read_budget=180,
            )
            name, lines = None, []
            for line in so.splitlines() if so else []:
                if line.startswith("===FILE:"):
                    if name is not None:
                        _write_back(name, lines)
                    name, lines = line[len("===FILE:") :], []
                elif name is not None:
                    lines.append(line)
            if name is not None:
                _write_back(name, lines)
        return rc
    finally:
        code, _ = k.api("DELETE", f"/api/v1/namespaces/{k.ns}/pods/{pod}")
        print(f"[bypass] pod deleted ({code})", file=sys.stderr, flush=True)


if __name__ == "__main__":
    sys.exit(main())
