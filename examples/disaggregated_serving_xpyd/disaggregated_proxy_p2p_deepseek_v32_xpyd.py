import argparse
import copy
import importlib.util
import json
import os
import socket
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, jsonify, make_response, request


DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 10001
DEFAULT_DISCOVERY_HOST = "0.0.0.0"
DEFAULT_DISCOVERY_PORT = 30002
DEFAULT_INSTANCE_TTL_SECONDS = 5
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=6 * 60 * 60)

app = Quart(__name__)

count = 0
prefill_instances: dict[str, tuple[str, float]] = {}
decode_instances: dict[str, tuple[str, float]] = {}
prefill_cv = threading.Condition()
decode_cv = threading.Condition()
instance_ttl_seconds = DEFAULT_INSTANCE_TTL_SECONDS


def _load_deepseek_v32_encoding_module():
    # Prefer installed package import (works regardless of script location).
    try:
        import vllm.tokenizers.deepseek_v32_encoding as _m
        return _m
    except ImportError:
        pass

    # Fallback: locate the file relative to the project root.
    # The script may live in any subdirectory (e.g. shell/, reproduction/),
    # so walk upward until we find the vllm package directory.
    script_dir = Path(__file__).resolve().parent
    for base in [script_dir, script_dir.parent, script_dir.parent.parent]:
        module_path = base / "vllm" / "tokenizers" / "deepseek_v32_encoding.py"
        if module_path.exists():
            break
    else:
        raise ImportError(
            "Unable to locate deepseek_v32_encoding.py. "
            "Install the vllm package or place the script under the vllm project root."
        )

    spec = importlib.util.spec_from_file_location(
        "_router_nccl_deepseek_v32_encoding",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load DeepSeek V3.2 encoder from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_deepseek_v32_encoding = _load_deepseek_v32_encoding_module()
encode_messages = _deepseek_v32_encoding.encode_messages
parse_message_from_completion_text = (
    _deepseek_v32_encoding.parse_message_from_completion_text
)


def random_uuid() -> str:
    return uuid.uuid4().hex


def _prune_expired(instances: dict[str, tuple[str, float]]) -> None:
    now = time.time()
    expired = [key for key, (_, deadline) in instances.items() if deadline <= now]
    for key in expired:
        zmq_addr, deadline = instances.pop(key)
        print(f"remove instance http={key} zmq={zmq_addr} deadline={deadline}")


def _register_instance(role: str, http_address: str, zmq_address: str) -> None:
    global prefill_instances
    global decode_instances

    bucket = prefill_instances if role == "P" else decode_instances
    cv = prefill_cv if role == "P" else decode_cv
    deadline = time.time() + instance_ttl_seconds

    with cv:
        node = bucket.get(http_address)
        bucket[http_address] = (zmq_address, deadline)
        _prune_expired(bucket)
        if node is None:
            print(f"add instance role={role} http={http_address} zmq={zmq_address}")


def _listen_for_register(poller: zmq.Poller, router_socket: Any) -> None:
    while True:
        socks = dict(poller.poll())
        if router_socket not in socks:
            continue

        remote_address, message = router_socket.recv_multipart()
        data = msgpack.loads(message)
        role = data.get("type")
        http_address = data.get("http_address")
        zmq_address = data.get("zmq_address")

        if role not in {"P", "D"} or not http_address or not zmq_address:
            print(
                f"unexpected register remote={remote_address!r} data={data!r}",
            )
            continue

        _register_instance(role, http_address, zmq_address)


def start_service_discovery(hostname: str, port: int) -> threading.Thread:
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("discovery port cannot be 0")

    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")

    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)

    thread = threading.Thread(
        target=_listen_for_register,
        args=(poller, router_socket),
        daemon=True,
    )
    thread.start()
    print(f"service discovery listening on tcp://{hostname}:{port}")
    return thread


def _choose_instance(
    instances: dict[str, tuple[str, float]],
    cv: threading.Condition,
    idx: int,
) -> tuple[str, str] | None:
    with cv:
        _prune_expired(instances)
        if not instances:
            return None
        items = list(instances.items())
        http_addr, (zmq_addr, _) = items[idx % len(items)]
        return http_addr, zmq_addr


async def _post_request(
    url: str,
    data: dict[str, Any],
    request_id: str,
    auth_header: str | None,
    *,
    stream: bool = False,
):
    headers = {"X-Request-Id": request_id}
    if auth_header:
        headers["Authorization"] = auth_header
    elif os.environ.get("OPENAI_API_KEY"):
        headers["Authorization"] = f"Bearer {os.environ['OPENAI_API_KEY']}"

    if not stream:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
            async with session.post(url=url, json=data, headers=headers) as response:
                body = await response.read()
                return {
                    "ok": response.status == 200,
                    "status": response.status,
                    "body": body,
                    "content_type": response.headers.get(
                        "Content-Type",
                        "application/json",
                    ),
                }

    session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
    try:
        response = await session.post(url=url, json=data, headers=headers)
    except Exception:
        await session.close()
        raise

    content_type = response.headers.get(
        "Content-Type",
        "application/json",
    )
    if response.status != 200:
        try:
            body = await response.read()
            return {
                "ok": False,
                "status": response.status,
                "body": body,
                "content_type": content_type,
            }
        finally:
            response.close()
            await session.close()

    async def _stream():
        try:
            async for chunk in response.content.iter_chunked(8192):
                yield chunk
        finally:
            response.close()
            await session.close()

    return {
        "ok": True,
        "status": response.status,
        "stream": _stream(),
        "content_type": content_type,
    }


def _build_request_id(prefill_zmq_addr: str, decode_zmq_addr: str) -> str:
    return (
        f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
        f"{decode_zmq_addr}_{random_uuid()}"
    )


def _is_chat_request_path(path: str) -> bool:
    return path == "/v1/chat/completions"


def _error_response(message: str, status: int = 400):
    return jsonify({"error": message}), status


def _normalize_message_content(message: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(message)
    content = normalized.get("content")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                raise ValueError(
                    "Only text content is supported in /v1/chat/completions bridge"
                )
            text_parts.append(part.get("text", ""))
        normalized["content"] = "".join(text_parts)
    return normalized


def _get_thinking_mode(request_data: dict[str, Any]) -> str:
    if request_data.get("thinking") or request_data.get("enable_thinking"):
        return "thinking"

    reasoning_effort = request_data.get("reasoning_effort")
    include_reasoning = request_data.get("include_reasoning", True)
    if reasoning_effort not in (None, "none") and include_reasoning:
        return "thinking"

    return "chat"


def _build_chat_prompt(request_data: dict[str, Any]) -> tuple[str, str]:
    messages = request_data.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        raise ValueError("`messages` must be a non-empty list")

    normalized_messages = [_normalize_message_content(msg) for msg in messages]
    thinking_mode = _get_thinking_mode(request_data)

    system_metadata: dict[str, Any] = {}
    if request_data.get("tools"):
        system_metadata["tools"] = copy.deepcopy(request_data["tools"])
    if request_data.get("response_format"):
        system_metadata["response_format"] = copy.deepcopy(
            request_data["response_format"]
        )
    if system_metadata:
        normalized_messages.insert(0, {"role": "system", **system_metadata})

    drop_thinking = normalized_messages[-1].get("role") in {"user", "developer"}
    prompt = encode_messages(
        normalized_messages,
        thinking_mode=thinking_mode,
        drop_thinking=drop_thinking,
    )
    return prompt, thinking_mode


def _build_completion_request_from_chat(
    request_data: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if request_data.get("stream"):
        raise ValueError(
            "`stream=true` is not supported for bridged /v1/chat/completions"
        )

    prompt, thinking_mode = _build_chat_prompt(request_data)

    completion_request = copy.deepcopy(request_data)
    for key in (
        "messages",
        "tools",
        "tool_choice",
        "response_format",
        "stream_options",
        "reasoning_effort",
        "include_reasoning",
        "parallel_tool_calls",
        "user",
        "chat_template",
        "chat_template_kwargs",
        "add_generation_prompt",
        "continue_final_message",
        "add_special_tokens",
        "documents",
        "thinking",
        "enable_thinking",
    ):
        completion_request.pop(key, None)

    max_completion_tokens = completion_request.pop("max_completion_tokens", None)
    if max_completion_tokens is not None and completion_request.get("max_tokens") is None:
        completion_request["max_tokens"] = max_completion_tokens

    completion_request["prompt"] = prompt
    completion_request["stream"] = False
    return completion_request, thinking_mode


def _materialize_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    materialized: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        function = tool_call.get("function", {})
        materialized.append(
            {
                "id": f"call_{random_uuid()}",
                "type": tool_call.get("type", "function"),
                "function": {
                    "name": function.get("name"),
                    "arguments": function.get("arguments", ""),
                },
            }
        )
    return materialized


def _convert_completion_to_chat_response(
    completion_payload: dict[str, Any],
    thinking_mode: str,
    include_reasoning: bool,
) -> dict[str, Any]:
    choices: list[dict[str, Any]] = []
    for choice in completion_payload.get("choices", []):
        text = choice.get("text", "")
        try:
            parsed_message = parse_message_from_completion_text(text, thinking_mode)
        except Exception:
            parsed_message = {
                "role": "assistant",
                "content": text,
                "reasoning": "",
                "tool_calls": [],
            }

        tool_calls = _materialize_tool_calls(parsed_message.get("tool_calls", []))
        content = parsed_message.get("content")
        message: dict[str, Any] = {
            "role": "assistant",
            "content": content if content or not tool_calls else None,
        }
        if include_reasoning and parsed_message.get("reasoning"):
            message["reasoning"] = parsed_message["reasoning"]
        if tool_calls:
            message["tool_calls"] = tool_calls

        finish_reason = choice.get("finish_reason")
        if tool_calls and finish_reason in (None, "stop"):
            finish_reason = "tool_calls"

        choices.append(
            {
                "index": choice.get("index", 0),
                "message": message,
                "logprobs": None,
                "finish_reason": finish_reason or "stop",
                "stop_reason": choice.get("stop_reason"),
                "token_ids": choice.get("token_ids"),
            }
        )

    response_id = completion_payload.get("id", f"chatcmpl-{random_uuid()}")
    if isinstance(response_id, str) and response_id.startswith("cmpl-"):
        response_id = "chat" + response_id

    return {
        "id": response_id,
        "object": "chat.completion",
        "created": completion_payload.get("created", int(time.time())),
        "model": completion_payload.get("model"),
        "choices": choices,
        "usage": completion_payload.get("usage"),
        "service_tier": completion_payload.get("service_tier"),
        "system_fingerprint": completion_payload.get("system_fingerprint"),
        "prompt_logprobs": completion_payload.get("prompt_logprobs"),
        "prompt_token_ids": completion_payload.get("prompt_token_ids"),
        "kv_transfer_params": completion_payload.get("kv_transfer_params"),
    }


@app.route("/health", methods=["GET"])
async def health():
    with prefill_cv:
        _prune_expired(prefill_instances)
        prefill_count = len(prefill_instances)
    with decode_cv:
        _prune_expired(decode_instances)
        decode_count = len(decode_instances)
    return jsonify(
        {
            "status": "ok",
            "prefill_instances": prefill_count,
            "decode_instances": decode_count,
        }
    )


@app.route("/debug/instances", methods=["GET"])
async def debug_instances():
    with prefill_cv:
        _prune_expired(prefill_instances)
        prefills = dict(prefill_instances)
    with decode_cv:
        _prune_expired(decode_instances)
        decodes = dict(decode_instances)
    return jsonify({"prefill": prefills, "decode": decodes})


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    global count

    original_request_data = await request.get_json()
    auth_header = request.headers.get("Authorization")
    is_chat_request = _is_chat_request_path(request.path)
    include_reasoning = bool(original_request_data.get("include_reasoning", True))
    thinking_mode = "chat"

    if is_chat_request:
        try:
            request_data, thinking_mode = _build_completion_request_from_chat(
                original_request_data
            )
        except ValueError as exc:
            return _error_response(str(exc), 400)
    else:
        request_data = original_request_data
    upstream_path = "/v1/completions" if is_chat_request else request.path

    pair_index = count
    count += 1

    prefill = _choose_instance(prefill_instances, prefill_cv, pair_index)
    decode = _choose_instance(decode_instances, decode_cv, pair_index)

    if prefill is None:
        return (
            jsonify({"error": "no registered prefill instances"}),
            503,
        )
    if decode is None:
        return (
            jsonify({"error": "no registered decode instances"}),
            503,
        )

    prefill_addr, prefill_zmq_addr = prefill
    decode_addr, decode_zmq_addr = decode

    print(
        "route request "
        f"[HTTP:{prefill_addr}, ZMQ:{prefill_zmq_addr}] -> "
        f"[HTTP:{decode_addr}, ZMQ:{decode_zmq_addr}]"
    )

    prefill_request = dict(request_data)
    prefill_request["max_tokens"] = 1
    prefill_request["stream"] = False
    prefill_request.pop("stream_options", None)
    if "max_completion_tokens" in prefill_request:
        prefill_request["max_completion_tokens"] = 1
    should_stream = bool(request_data.get("stream"))

    request_id = _build_request_id(prefill_zmq_addr, decode_zmq_addr)

    prefill_result = await _post_request(
        f"http://{prefill_addr}{upstream_path}",
        prefill_request,
        request_id,
        auth_header,
        stream=False,
    )
    if not prefill_result["ok"]:
        return make_response(
            prefill_result["body"],
            prefill_result["status"],
            {"Content-Type": prefill_result["content_type"]},
        )

    decode_result = await _post_request(
        f"http://{decode_addr}{upstream_path}",
        request_data,
        request_id,
        auth_header,
        stream=should_stream,
    )
    if not decode_result["ok"]:
        return make_response(
            decode_result["body"],
            decode_result["status"],
            {"Content-Type": decode_result["content_type"]},
        )

    if is_chat_request:
        try:
            completion_payload = json.loads(decode_result["body"])
        except json.JSONDecodeError:
            return _error_response("decode instance returned non-JSON completion body", 502)

        chat_payload = _convert_completion_to_chat_response(
            completion_payload,
            thinking_mode,
            include_reasoning,
        )
        return await make_response(
            json.dumps(chat_payload, ensure_ascii=False),
            decode_result["status"],
            {"Content-Type": "application/json"},
        )

    if should_stream:
        response = await make_response(
            decode_result["stream"],
            decode_result["status"],
            {"Content-Type": decode_result["content_type"]},
        )
        response.timeout = None
        return response

    return await make_response(
        decode_result["body"],
        decode_result["status"],
        {"Content-Type": decode_result["content_type"]},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=DEFAULT_HTTP_HOST)
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument("--discovery-host", default=DEFAULT_DISCOVERY_HOST)
    parser.add_argument("--discovery-port", type=int, default=DEFAULT_DISCOVERY_PORT)
    parser.add_argument("--instance-ttl-seconds", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    instance_ttl_seconds = args.instance_ttl_seconds
    discovery_thread = start_service_discovery(
        args.discovery_host,
        args.discovery_port,
    )
    app.run(host=args.host, port=args.http_port)
    discovery_thread.join()
