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
import logging
import os
import threading
import time
import urllib.parse
import uuid
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

global_args = None

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [router] %(levelname)s %(message)s"
)
logger = logging.getLogger("flagcx.router")

# Interval (seconds) between periodic router metric log lines. 0 disables it.
STAT_LOG_INTERVAL = float(os.environ.get("FLAGCX_ROUTER_STAT_INTERVAL", "10"))


class RouterStats:
    """Per-interval router-side latency/throughput observations.

    Mirrors the connector-side ``FlagCXKVConnectorStats`` shape: writers
    append raw observations under a lock, and ``clone_and_reset`` hands the
    logger a snapshot so each interval is reported exactly once. Timings are
    seconds and reported in ms, matching the KV-transfer metric convention.

    The three latencies bracket the P/D handoff:
      prefill_latency  request sent to P -> P responded (KV ready to pull)
      ttft             request sent to D -> first streamed byte from D
      e2e_latency      request sent to D -> decode stream closed
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.reset()

    def reset(self):
        self.data: dict[str, list[float | int]] = {
            "prefill_latency": [],
            "ttft": [],
            "e2e_latency": [],
            "decode_bytes": [],
            "num_failed_prefills": [],
            "num_failed_decodes": [],
        }

    def record_prefill(self, duration_s: float):
        with self._lock:
            self.data["prefill_latency"].append(duration_s)

    def record_failed_prefill(self):
        with self._lock:
            self.data["num_failed_prefills"].append(1)

    def record_request(self, ttft_s: float, e2e_s: float, num_bytes: int):
        with self._lock:
            self.data["ttft"].append(ttft_s)
            self.data["e2e_latency"].append(e2e_s)
            self.data["decode_bytes"].append(num_bytes)

    def record_failed_decode(self):
        with self._lock:
            self.data["num_failed_decodes"].append(1)

    def clone_and_reset(self) -> dict[str, list[float | int]]:
        with self._lock:
            snapshot = {k: list(v) for k, v in self.data.items()}
            self.reset()
        return snapshot

    @staticmethod
    def is_empty(data: dict[str, list[float | int]]) -> bool:
        return not any(data.values())

    @staticmethod
    def reduce(data: dict[str, list[float | int]]) -> dict[str, int | float]:
        """Reduce one interval's observations to representative values."""
        num_failed_prefills = len(data["num_failed_prefills"])
        num_failed_decodes = len(data["num_failed_decodes"])
        num_requests = len(data["e2e_latency"])

        def _ms(values, pct=None):
            if not values:
                return 0
            ordered = sorted(values)
            if pct is None:
                return round(sum(ordered) / len(ordered) * 1e3, 3)
            # Nearest-rank percentile; avoids a numpy dependency in the proxy.
            idx = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
            return round(ordered[idx] * 1e3, 3)

        total_mb = sum(data["decode_bytes"]) / 2**20
        total_time_s = sum(data["e2e_latency"])
        # Concurrent requests overlap, so this is per-request average
        # throughput, not aggregate router egress.
        throughput_mb_s = round(total_mb / total_time_s, 3) if total_time_s > 0 else 0.0

        return {
            "Num requests": num_requests,
            "Avg prefill time (ms)": _ms(data["prefill_latency"]),
            "P90 prefill time (ms)": _ms(data["prefill_latency"], 90),
            "Avg TTFT (ms)": _ms(data["ttft"]),
            "P90 TTFT (ms)": _ms(data["ttft"], 90),
            "Avg E2E time (ms)": _ms(data["e2e_latency"]),
            "P90 E2E time (ms)": _ms(data["e2e_latency"], 90),
            "Avg MB per request": (
                round(total_mb / num_requests, 4) if num_requests else 0
            ),
            "Throughput (MB/s)": throughput_mb_s,
            "Num failed prefills": num_failed_prefills,
            "Num failed decodes": num_failed_decodes,
        }


stats = RouterStats()


async def _stat_logger_loop():
    """Periodically log one line per interval, like KVConnectorLogging.log."""
    while True:
        await asyncio.sleep(STAT_LOG_INTERVAL)
        snapshot = stats.clone_and_reset()
        if RouterStats.is_empty(snapshot):
            continue
        metrics = RouterStats.reduce(snapshot)
        logger.info(
            "Router metrics: %s", ", ".join(f"{k}={v}" for k, v in metrics.items())
        )


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
    stat_logger_task = (
        asyncio.create_task(_stat_logger_loop()) if STAT_LOG_INTERVAL > 0 else None
    )
    app.state.prefill_iterator = itertools.cycle(range(len(app.state.prefill_clients)))
    app.state.decode_iterator = itertools.cycle(range(len(app.state.decode_clients)))
    print(
        f"Got {len(app.state.prefill_clients)} prefill clients and {len(app.state.decode_clients)} decode clients."
    )

    yield

    if stat_logger_task is not None:
        stat_logger_task.cancel()
    for c in app.state.prefill_clients:
        await c["client"].aclose()
    for c in app.state.decode_clients:
        await c["client"].aclose()


app = FastAPI(lifespan=lifespan)


async def send_to_prefill(prefill_client, endpoint, req_data, request_id):
    data = req_data.copy()
    data["kv_transfer_params"] = {
        "do_remote_decode": True,
        "do_remote_prefill": False,
        "transfer_id": f"fgx-{request_id}",
    }
    data["stream"] = False
    data["max_tokens"] = 1
    if "max_completion_tokens" in data:
        data["max_completion_tokens"] = 1
    data.pop("stream_options", None)
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start_time = time.perf_counter()
    try:
        response = await prefill_client["client"].post(
            endpoint, json=data, headers=headers
        )
        response.raise_for_status()
        await response.aclose()
        duration = time.perf_counter() - start_time
        stats.record_prefill(duration)
        logger.debug(
            "Prefill %s on %s done, took %.3fs",
            request_id,
            prefill_client["url"],
            duration,
        )
    except Exception as exc:
        stats.record_failed_prefill()
        logger.error(
            "Prefill request %s on %s failed after %.3fs: %s",
            request_id,
            prefill_client["url"],
            time.perf_counter() - start_time,
            exc,
        )


async def stream_from_decode(
    prefill_client, decode_client, endpoint, req_data, request_id
):
    data = req_data.copy()
    data["kv_transfer_params"] = {
        "do_remote_prefill": True,
        "do_remote_decode": False,
        "remote_host": prefill_client["remote_host"],
        "remote_port": prefill_client["side_channel_port"],
        "transfer_id": f"fgx-{request_id}",
    }
    headers = {"X-Request-Id": request_id}
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    start_time = time.perf_counter()
    first_byte_time = None
    num_bytes = 0
    try:
        async with decode_client["client"].stream(
            "POST", endpoint, json=data, headers=headers
        ) as response:
            response.raise_for_status()
            async for chunk in response.aiter_bytes():
                if first_byte_time is None:
                    first_byte_time = time.perf_counter()
                num_bytes += len(chunk)
                yield chunk
    except Exception as exc:
        stats.record_failed_decode()
        logger.error(
            "Decode request %s on %s failed after %.3fs: %s",
            request_id,
            decode_client["url"],
            time.perf_counter() - start_time,
            exc,
        )
        raise
    end_time = time.perf_counter()
    # No byte ever arrived (empty stream): attribute the whole wait to TTFT.
    ttft = (first_byte_time or end_time) - start_time
    stats.record_request(ttft_s=ttft, e2e_s=end_time - start_time, num_bytes=num_bytes)
    logger.debug(
        "Decode %s on %s done: ttft=%.3fs e2e=%.3fs bytes=%d",
        request_id,
        decode_client["url"],
        ttft,
        end_time - start_time,
        num_bytes,
    )


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
