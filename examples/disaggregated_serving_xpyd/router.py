# SPDX-License-Identifier: Apache-2.0
# 1P1D disaggregated serving proxy for FlagcxConnector
#
# Usage:
#   python3 router.py \
#     --host 0.0.0.0 --port 8000 \
#     --prefill http://<prefill_host>:<vllm_port> <FLAGCX_BOOTSTRAP_PORT> \
#     --decode  http://<decode_host>:<vllm_port>
#
# FLAGCX_BOOTSTRAP_PORT is the ZMQ side-channel BASE port (default 8998).

import argparse
import asyncio
import itertools
import os
import urllib.parse
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

global_args = None


async def wait_for_health(prefill_clients, decode_clients, ready):
    for client_info in prefill_clients:
        while True:
            try:
                response = await client_info["client"].get("/health")
                response.raise_for_status()
                break
            except Exception as exc:
                print(f"Waiting for prefill {client_info['url']}/health: {exc}")
                await asyncio.sleep(1)
        print(f"Prefill {client_info['url']} is healthy.")
    for client_info in decode_clients:
        while True:
            try:
                response = await client_info["client"].get("/health")
                response.raise_for_status()
                break
            except Exception as exc:
                print(f"Waiting for decode {client_info['url']}/health: {exc}")
                await asyncio.sleep(1)
        print(f"Decode {client_info['url']} is healthy.")
    ready.set()
    print("All prefill and decode instances are ready.")


@asynccontextmanager
async def lifespan(app):
    app.state.prefill_clients = []
    app.state.decode_clients = []
    app.state.ready = asyncio.Event()

    for url, side_channel_port in global_args.prefill:
        parsed_url = urllib.parse.urlparse(url)
        app.state.prefill_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "url": url,
                "remote_host": parsed_url.hostname,
                "side_channel_port": side_channel_port,
            }
        )

    for url in global_args.decode:
        app.state.decode_clients.append(
            {
                "client": httpx.AsyncClient(
                    timeout=None,
                    base_url=url,
                    limits=httpx.Limits(
                        max_connections=None, max_keepalive_connections=None
                    ),
                ),
                "url": url,
            }
        )

    asyncio.create_task(
        wait_for_health(
            app.state.prefill_clients, app.state.decode_clients, app.state.ready
        )
    )
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    print(
        f"Got {len(app.state.prefill_clients)} prefill clients and {len(app.state.decode_clients)} decode clients."
    )

    yield

    for c in app.state.prefill_clients:
        await c["client"].aclose()
    for c in app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


async def send_to_prefill(prefill_client, endpoint, req_data, request_id):
    data = req_data.copy()
    data["kv_transfer_params"] = {"do_remote_decode": True, "do_remote_prefill": False}
    data["stream"] = False
    data["max_tokens"] = 1
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 1
    data.pop("stream_options", None)
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = await prefill_client["client"].post(
            endpoint, json=data, headers=headers
        )
        response.raise_for_status()
        await response.aclose()
    except Exception as exc:
        print(f"Prefill request {request_id} error: {exc}")


async def stream_from_decode(
    prefill_client, decode_client, endpoint, req_data, request_id
):
    data = req_data.copy()
    data["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": prefill_client["remote_host"],
        "remote_port": prefill_client["side_channel_port"],
    }
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    async with decode_client["client"].stream(
        "POST", endpoint, json=data, headers=headers
    ) as response:
        response.raise_for_status()
        async for chunk in response.aiter_bytes():
            yield chunk


async def _handle_completions(api, request):
    if not app.state.ready.is_set():
        raise HTTPException(status_code=503, detail="Service Unavailable")
    try:
        req_data = await request.json()
        request_id = str(uuid.uuid4())
        prefill_client = app.state.prefill_clients[next(app.state.prefill_iterator)]
        decode_client = app.state.decode_clients[next(app.state.decode_iterator)]
        asyncio.create_task(send_to_prefill(prefill_client, api, req_data, request_id))

        async def generate():
            async for chunk in stream_from_decode(
                prefill_client, decode_client, api, req_data, request_id
            ):
                yield chunk

        return StreamingResponse(generate(), media_type="application/json")
    except Exception as e:
        import sys
        import traceback

        print(f"Error in proxy [{api}]: {e}")
        print("".join(traceback.format_exception(*sys.exc_info())))
        raise


@app.post("/v1/completions")
async def handle_completions(request: Request):
    return await _handle_completions("/v1/completions", request)


@app.post("/v1/chat/completions")
async def handle_chat_completions(request: Request):
    return await _handle_completions("/v1/chat/completions", request)


def parse_args():
    parser = argparse.ArgumentParser(
        description="1P1D proxy for FlagcxConnector (ZMQ side-channel)"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument(
        "--prefill",
        nargs="+",
        action="append",
        dest="prefill_raw",
        metavar=("URL", "ZMQ_PORT"),
        help="Prefill URL and ZMQ side-channel base port (= FLAGCX_BOOTSTRAP_PORT, default 8998)",
    )
    parser.add_argument(
        "--decode",
        nargs=1,
        action="append",
        dest="decode_raw",
        metavar="URL",
        help="Decode vllm URL",
    )

    args = parser.parse_args()
    args.prefill = []
    for item in args.prefill_raw or []:
        url = item[0]
        port = int(item[1]) if len(item) >= 2 else 8998
        args.prefill.append((url, port))
    args.decode = [item[0] for item in (args.decode_raw or [])]

    if not args.prefill:
        parser.error("At least one --prefill URL is required.")
    if not args.decode:
        parser.error("At least one --decode URL is required.")
    return args


if __name__ == "__main__":
    global_args = parse_args()
    import uvicorn

    uvicorn.run(app, host=global_args.host, port=global_args.port)
