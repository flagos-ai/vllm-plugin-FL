"""PD failures must close the peer stream instead of leaving it waiting."""

import asyncio
import importlib.util
import itertools
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def router():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/disaggregated_serving_xpyd/router.py"
    )
    spec = importlib.util.spec_from_file_location("pd_router_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.app.state.ready = SimpleNamespace(is_set=lambda: True)
    module.app.state.prefill_clients = [{}]
    module.app.state.decode_clients = [{}]
    module.app.state.prefill_iterator = itertools.cycle([0])
    module.app.state.decode_iterator = itertools.cycle([0])
    return module


async def response(router):
    async def body():
        return {"model": "test", "prompt": [1, 2], "max_tokens": 4}

    return await router._handle_completions(
        "/v1/completions", SimpleNamespace(json=body)
    )


def test_prefill_failure_cancels_decode_wait(router, monkeypatch):
    async def run():
        decoding, closed = asyncio.Event(), asyncio.Event()

        async def prefill(*args):
            await decoding.wait()
            raise httpx.ReadError("upstream connection closed")

        async def decode(*args):
            try:
                decoding.set()
                await asyncio.Future()
                yield b"never"
            finally:
                closed.set()

        monkeypatch.setattr(router, "send_to_prefill", prefill)
        monkeypatch.setattr(router, "stream_from_decode", decode)

        async def consume():
            r = await response(router)
            return [chunk async for chunk in r.body_iterator]

        with pytest.raises(Exception) as error:
            await asyncio.wait_for(consume(), 1)
        assert any(isinstance(e, httpx.ReadError) for e in error.value.exceptions)
        assert closed.is_set()

    asyncio.run(run())


@pytest.mark.parametrize("failure", ["decode", "client"])
def test_stream_failure_cancels_prefill(router, monkeypatch, failure):
    async def run():
        started, cancelled, decoding = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def prefill(*args):
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        async def decode(*args):
            await started.wait()
            decoding.set()
            if failure == "decode":
                raise httpx.ReadError("decode disconnected")
            await asyncio.Future()
            yield b"never"

        monkeypatch.setattr(router, "send_to_prefill", prefill)
        monkeypatch.setattr(router, "stream_from_decode", decode)

        async def consume():
            r = await response(router)
            return [chunk async for chunk in r.body_iterator]

        task = asyncio.create_task(consume())
        await asyncio.wait_for(decoding.wait(), 1)
        if failure == "client":
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            with pytest.raises(Exception) as error:
                await task
            assert any(isinstance(e, httpx.ReadError) for e in error.value.exceptions)
        assert cancelled.is_set()

    asyncio.run(run())


def test_success_preserves_stream_and_overlaps_prefill(router, monkeypatch):
    async def run():
        decoding = asyncio.Event()

        async def prefill(*args):
            await decoding.wait()

        async def decode(*args):
            decoding.set()
            for chunk in (b"first", b"second"):
                yield chunk
                await asyncio.sleep(0)

        monkeypatch.setattr(router, "send_to_prefill", prefill)
        monkeypatch.setattr(router, "stream_from_decode", decode)
        r = await response(router)
        assert [chunk async for chunk in r.body_iterator] == [b"first", b"second"]

    asyncio.run(run())


def test_http_prefill_error_is_propagated(router):
    async def run():
        def respond(request):
            return httpx.Response(500, json={"error": "failed"})

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(respond), base_url="http://prefill"
        ) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await router.send_to_prefill(
                    {"client": client},
                    "/v1/completions",
                    {"prompt": [1, 2], "max_tokens": 4},
                    "test",
                )

    asyncio.run(run())
