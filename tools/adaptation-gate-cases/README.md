# Adaptation Gate Cases

| Model | mode |single long text case | single long image case| text cases | image cases |mixed text and image cases | total cases
|---|---:|---:|---:|---:|---:|---:|---:|
| `Qwen3.6-27B` | eager  | 1 | 1 | 8 | 8 | 4+4 | 26 |
| `Qwen3.6-27B` | graph  | 1 | 1 | 8 | 8 | 4+4 | 26 |
| `Qwen3.6-35B-A3B` | eager | 1 | 1 | 8 | 8 |  4+4 | 26 |
| `Qwen3.6-35B-A3B` | graph | 1 | 1 | 8 | 8 |  4+4 | 26 |

The matrix counts pytest scenarios; each concurrent scenario sends eight
requests.

This directory is a small manual gate for accelerator adaptation and vLLM
plugin upgrades. It tests two Qwen models in eager and graph modes.

Each mode runs the following scenarios:

| Test file | Scenarios |
|---|---|
| `test_text.py` | Single text case; 8 concurrent text cases |
| `test_image.py` | Single image case; 8 concurrent image cases |
| `test_mix_text_image.py` | 8 concurrent mixed text-image cases |

## Files

- `run_serve_eager.sh`: starts vLLM with `--enforce-eager`.
- `run_serve_graph.sh`: starts vLLM without `--enforce-eager`.
- `run_serve.sh`: compatibility alias for `run_serve_graph.sh`.
- `run_test.sh`: waits for the service and runs all three pytest files.
- `test_text.py`: one detailed LLM introduction of at least 256 characters and
  eight concurrent short requests.
- `test_image.py`: one OCR request and eight concurrent image requests.
- `test_mix_text_image.py`: eight concurrent requests, four text and four image.
- `case_utils.py`: requests, concurrency, quality checks, and JSON reports.
- `generate_images.py`: regenerates the eight deterministic PNG fixtures.

The image fixtures cover quadrant colors, shape ordering, shape counting, OCR
of `hello vllm`, nested shapes, shape colors, and colored-shape counting. The
single-image case performs exact OCR. The eight-way case uses every image and
checks text presence for the OCR fixture while keeping exact checks for the
other visual tasks.

## Run

Start an eager service in terminal 1:

```bash
cd /vllm-workspace/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen3.6-27B PORT=8000 ./run_serve_eager.sh
```

Wait for the service and run every test in terminal 2:

```bash
cd /vllm-workspace/vllm-plugin-FL/tools/adaptation-gate-cases
MODEL_PATH=/models/Qwen3.6-27B PORT=8000 ./run_test.sh
```

Then stop the eager service and repeat with graph mode on a different port:

```bash
MODEL_PATH=/models/Qwen3.6-27B PORT=8001 ./run_serve_graph.sh
MODEL_PATH=/models/Qwen3.6-27B PORT=8001 ./run_test.sh
```

`run_test.sh` executes all of these commands even if an earlier file fails:

```bash
pytest -sv test_text.py
pytest -sv test_image.py
pytest -sv test_mix_text_image.py
```

The test client resolves the request model from `SERVED_MODEL_NAME` first and
falls back to `MODEL_PATH`; it never uses an implicit model name. It does not
detect or depend on the server execution mode. Every model must still be tested
once with eager execution and once with graph execution; stop the current
service before switching modes:

- `/models/Qwen3.6-27B`
- `/models/Qwen3.6-35B-A3B`
- `eager`
- `graph`

The serve scripts require `MODEL_PATH` and `PORT`. Their served model name uses
`SERVED_MODEL_NAME` when set and otherwise falls back to `MODEL_PATH`.
`run_test.sh` requires `PORT` plus either `SERVED_MODEL_NAME` or `MODEL_PATH`.
Useful environment overrides are `TENSOR_PARALLEL_SIZE`, `MAX_MODEL_LEN`, and
`SERVER_PID_FILE`. `run_test.sh` also supports `BASE_URL`, `SERVICE_TIMEOUT`,
`REQUEST_TIMEOUT`, `SERVER_PID_FILE`, and `RESULTS_DIR`.

You may use a custom command instead of `run_serve.sh`. It must expose an
OpenAI-compatible endpoint, serve the model as `qwen`, and allow local images
from this directory. For example:

```bash
vllm serve /models/Qwen3.6-27B \
    --served-model-name /models/Qwen3.6-27B \
    --host 127.0.0.1 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --max-model-len 32768 \
    --allowed-local-media-path "$PWD/images" \
    --trust-remote-code \
    --enforce-eager
```

Use `--enforce-eager` for the eager run. Remove it for the graph run, then
run `MODEL_PATH=/path/to/model PORT=8000 ./run_test.sh` against each service.
Both runs are mandatory even though the test client does not need to know the
execution mode.

Each pytest file or individual case can also be run directly. `PORT` and at
least one of `SERVED_MODEL_NAME` or `MODEL_PATH` are required. The result label
uses `MODEL_PATH` when available and otherwise uses `SERVED_MODEL_NAME`:

```ini
MODEL_PATH=/models/Qwen3.6-27B PORT=8000 pytest -sv test_text.py
```

```ini
MODEL_PATH=/models/Qwen3.6-27B PORT=8000 \
    pytest -sv test_image.py::test_image_single
```

## Results and quality checks

Each pytest case writes one JSON file under
`results/MODEL/port-PORT/SCENARIO.json`. Use different ports for eager and
graph services when both result sets must be retained; reusing a port
overwrites that model and port's previous result. A two-model, two-port matrix
contains 20 JSON files.
Every file contains the exact inputs, API outputs, timing, per-response checks,
and a pass/fail summary. On failure, a concise request-level error summary is
also printed to the terminal.

Responses must contain the expected semantic answer and must not contain empty
output, `!!!`, mojibake, control characters, suspicious character runs, or
repeated words or phrases. The detailed LLM introduction must also contain at
least 256 characters. Number words and digits are treated as equivalent.

Regenerate images with:

```bash
python3 generate_images.py
```
