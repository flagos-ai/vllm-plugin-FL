"""Common PD router harness: retain P tasks and log request lifecycle only."""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

import uvicorn
from fastapi.responses import JSONResponse

source = Path(sys.argv.pop(1))
spec = importlib.util.spec_from_file_location("pd_router", source)
router = importlib.util.module_from_spec(spec)
spec.loader.exec_module(router)
pending = {}
original = router.send_to_prefill


def event(phase, request_id, **details):
    print(
        json.dumps(
            dict(phase=phase, request_id=request_id, time=time.time(), **details)
        ),
        flush=True,
    )


async def traced_prefill(client, endpoint, data, request_id):
    # Retain a strong reference while the response stream is running. The
    # original router detaches this task without retaining it.
    pending[request_id] = (asyncio.current_task(), time.monotonic())
    event("prefill_start", request_id)
    try:
        await original(client, endpoint, data, request_id)
        event("prefill_return", request_id)
    except BaseException as exc:
        event("prefill_exception", request_id, error=repr(exc))
        raise
    finally:
        pending.pop(request_id, None)


router.send_to_prefill = traced_prefill


@router.app.get("/pd_health")
async def health():
    return JSONResponse(
        {"ready": router.app.state.ready.is_set()},
        status_code=200 if router.app.state.ready.is_set() else 503,
    )


@router.app.get("/pd_pending")
async def inspect():
    return [
        {
            "request_id": rid,
            "elapsed": time.monotonic() - start,
            "stack": [f"{f.f_code.co_filename}:{f.f_lineno}" for f in task.get_stack()],
        }
        for rid, (task, start) in pending.items()
    ]


router.global_args = router.parse_args()
uvicorn.run(router.app, host=router.global_args.host, port=router.global_args.port)
