# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import requests
from fastapi.responses import ORJSONResponse
from starlette.responses import JSONResponse

pytest.importorskip("orjson")

spec = importlib.util.spec_from_file_location(
    "bge_m3_benchmark",
    Path(__file__).resolve().parents[4] / "examples/bge_m3_joint/benchmark.py",
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


def payload(mask, length=3):
    vector = np.linspace(-1, 1, 1024, dtype=np.float32).tolist()
    return {
        "dense_vecs": [vector] if mask & 1 else None,
        "lexical_weights": [{"123": 0.125, "456": 2.5}] if mask & 2 else None,
        "colbert_vecs": [[vector] * (length - 1)] if mask & 4 else None,
    }


@pytest.mark.parametrize("mask", range(8))
def test_wire_equivalence(mask):
    data = payload(mask)
    envelope = {"request_id": "中文", "created_at": 1, "data": data}
    standard = JSONResponse(envelope).body
    fast = ORJSONResponse(envelope).body
    assert json.loads(standard) == json.loads(fast) == envelope
    for body in (standard, fast):
        response = requests.Response()
        response.status_code = 200
        response._content = body
        for parser in ("stdlib", "orjson"):
            result = benchmark.decode_response(response, parser)
            assert result == data
            benchmark.validate_output(result, mask, 3)


@pytest.mark.parametrize("field", benchmark.FIELDS)
def test_nonfinite_or_invalid_output_rejected(field):
    data = payload(7)
    if field == "lexical_weights":
        data[field][0]["123"] = float("nan")
    elif field == "dense_vecs":
        data[field][0][0] = float("nan")
    else:
        data[field][0][0][0] = float("nan")
    with pytest.raises(ValueError):
        benchmark.validate_output(data, 7, 3)


def test_wrong_shape_or_enabled_field_rejected():
    data = payload(7)
    data["colbert_vecs"][0].pop()
    with pytest.raises(ValueError):
        benchmark.validate_output(data, 7, 3)
    with pytest.raises(ValueError):
        benchmark.validate_output(payload(7), 1, 3)


def test_http_errors_not_parsed_as_results():
    response = requests.Response()
    response.status_code = 500
    response._content = b'{"error":"failed"}'
    with pytest.raises(requests.HTTPError):
        benchmark.decode_response(response, "orjson")


def test_actual_http_peak_excludes_client_parse():
    rows = [
        {"start_ns": 1, "body_received_ns": 3, "end_ns": 10},
        {"start_ns": 3, "body_received_ns": 5, "end_ns": 12},
    ]
    assert benchmark.http_peak(rows) == 1
