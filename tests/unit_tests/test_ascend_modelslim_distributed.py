# SPDX-License-Identifier: Apache-2.0
"""Launch real Ascend regressions through run.py's serial multi-GPU pass."""

import os
import signal
import socket
import subprocess
import sys
from contextlib import suppress
from pathlib import Path

import pytest


@pytest.mark.gpu
@pytest.mark.multi_gpu
@pytest.mark.parametrize(
    "target",
    [
        "functional_tests/distributed/test_ascend_modelslim_agrs.py",
        (
            "functional_tests/models/test_deepseek_v4_causal_lm.py"
            "::test_pp_hc_head_checkpoint_loading"
        ),
    ],
    ids=["agrs", "pp"],
)
def test_modelslim_distributed_contracts(tmp_path, target):
    torch_npu = pytest.importorskip("torch_npu")
    import torch

    if not torch_npu.npu.is_available() or torch.npu.device_count() < 2:
        pytest.skip("Requires two Ascend devices")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("The launcher requires a non-distributed parent")
    tests = Path(__file__).resolve().parents[1]
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc-per-node=2",
        "--master-addr=127.0.0.1",
        f"--master-port={port}",
        "-m",
        "pytest",
        "--noconftest",
        "-q",
        "-s",
        str(tests / target),
    ]
    log = tmp_path / "modelslim-distributed.log"
    timed_out = False
    with log.open("w") as output:
        process = subprocess.Popen(
            command, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            try:
                process.wait(timeout=600)
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            # torchrun owns pytest/HCCL descendants; kill the private group,
            # even if the direct parent has already exited on an error.
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # The finally block force-kills the group after the grace period.
                pass
            finally:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert not timed_out, f"Distributed timeout after 600s: {log}\n{log.read_text()}"
    assert process.returncode == 0, f"{log}\n{log.read_text()}"
