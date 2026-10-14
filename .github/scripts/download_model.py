#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DEBUG PROBE (PR #588): fetch a model from HuggingFace into a local path.

Downloads a full snapshot of ``--repo`` (default Qwen/Qwen3-0.6B, the model
metax benchmark smoke needs and that vanished from the node's /data/models)
into ``--dest``. Idempotent: skips when ``<dest>/config.json`` already
exists.

The metax runner sits behind an HTTP proxy and huggingface.co is not
always reachable through it, so failures fall through an endpoint/proxy
chain by re-exec'ing this script:

  1. default endpoint, inherited proxy env
  2. HF_ENDPOINT=https://hf-mirror.com, inherited proxy env
  3. HF_ENDPOINT=https://hf-mirror.com, proxy env cleared

Usage::

    python .github/scripts/download_model.py \
        --repo Qwen/Qwen3-0.6B --dest /data/models/Qwen/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import fcntl
import os
import sys
from pathlib import Path

MIRROR = "https://hf-mirror.com"
PROXY_KEYS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


def next_env(attempt: int) -> dict[str, str] | None:
    """Environment for attempt ``attempt + 1``, or None when out of chain."""
    env = dict(os.environ)
    if attempt == 0:
        env["HF_ENDPOINT"] = MIRROR
    elif attempt == 1:
        env["HF_ENDPOINT"] = MIRROR
        for key in PROXY_KEYS:
            env.pop(key, None)
    else:
        return None
    env["_DL_ATTEMPT"] = str(attempt + 1)
    return env


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--dest", default="/data/models/Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    dest = Path(args.dest)

    # DEBUG PROBE (PR #588): flock so concurrent sweep instances sharing one
    # machine's filesystem do not race the check-then-download; instances on
    # different machines hold independent locks and never conflict.
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest.parent / ".download_model.lock", "w") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        return _run(args, dest, lock_file)


def _run(args: argparse.Namespace, dest: Path, lock_file) -> int:
    if (dest / "config.json").exists():
        print(f"[download-model] {dest} already populated, skipping")
        return 0

    attempt = int(os.environ.get("_DL_ATTEMPT", "0"))
    print(
        f"[download-model] attempt {attempt + 1}: repo={args.repo} dest={dest} "
        f"endpoint={os.environ.get('HF_ENDPOINT', 'default')} "
        f"proxy={'set' if os.environ.get('HTTPS_PROXY') else 'unset'}"
    )

    from huggingface_hub import snapshot_download

    try:
        path = snapshot_download(repo_id=args.repo, local_dir=str(dest))
    except Exception as exc:  # noqa: BLE001 - any failure moves to next attempt
        print(f"[download-model] attempt {attempt + 1} failed: {exc}", file=sys.stderr)
        env = next_env(attempt)
        if env is None:
            print("[download-model] all attempts failed", file=sys.stderr)
            return 1
        # Release the flock before re-exec: the successor process acquires the
        # same lock itself and would otherwise deadlock against this one.
        lock_file.close()
        os.execve(
            sys.executable,
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--repo",
                args.repo,
                "--dest",
                str(dest),
            ],
            env,
        )
        return 1  # unreachable when execve succeeds

    print(f"[download-model] downloaded to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
