# Validate the K3 installation

After installing the pinned dependencies, run the integration contracts in a
matching MetaX runtime with an idle visible GPU:

```sh
VLLM_FL_CONFIG="$PWD/examples/kimi_k3_metax/dispatch.yaml" \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
python -m pytest --import-mode=importlib --confcutdir=tests/k3 -q tests/k3
```

These checks cover policy, dispatch, registration, PP ownership and native API
compatibility. They do not replace numerical, Graph or model-quality validation.

Use fresh private caches for a model smoke covering short and long text,
images and mixed requests. Check final answers, finite logprobs and request
accounting. Startup success alone does not establish correctness.

## Four-node service gate

Use the K3 example to start a dedicated test service. Run
`tools/adaptation-gate-cases/run_test.sh` once with the example's decode Graph
configuration and once with eager execution. For eager, use a separate copy of
the example configuration with `enforce-eager: true`, compilation mode 0 and
`cudagraph_mode: NONE`. Keep the model, topology, admission policy and client
inputs identical. For the direct-answer gate set the K3 chat template's
`thinking` default to false. Mount the gate images at the same path on the
client and server and allow that directory with `allowed-local-media-path`.

```bash
MODEL_PATH=/models/Kimi-K3 SERVED_MODEL_NAME=kimi-k3 PORT=8000 \
BASE_URL="http://${MASTER_ADDR}:8000/v1" RESULTS_DIR="$RESULT_DIR" \
bash tools/adaptation-gate-cases/run_test.sh
```

Check all five scenario files (26 unique responses per mode), their content
checks and server-side token increments. At shutdown, inspect the per-worker
runner cleanup messages and the executor's graceful-exit messages across all
four nodes. A container exit code alone also permits supervisors that killed
remaining workers. Retain failures and distinguish component tests from
full-model runs.

K3 requires the pinned vLLM source and MetaX runtime. The retained KDA precision
and stride-safe MLA fallbacks are part of the accepted numerical contract.
Full quality scores and serving measurements belong to their recorded version;
a cleanup does not rerun or update those results. Detailed run evidence is
maintained with the PR and project reports.

## Results for plugin revision `d66a186`

The TP4 compatibility gate used MetaX Torch 2.8 / MACA 3.7 and the installed
plugin at `d66a18634b9efd2d81c6a184917f6b24e4d3a4b6`. The image used FGV
`f775b37f3066e2935742d26c7592a5dd55499159` for these Qwen models. Each row
covers single text, single image, eight text, eight image and eight mixed
requests, in that order.

| Model | Mode | Content checks by scenario | Unique completions |
| --- | --- | --- | ---: |
| Qwen3.6-27B | eager | 1/1, 1/1, 8/8, 8/8, 8/8 | 26 |
| Qwen3.6-27B | Graph | 1/1, 1/1, 8/8, 8/8, 8/8 | 26 |
| Qwen3.6-35B-A3B | eager | 1/1, 1/1, 8/8, 8/8, 8/8 | 26 |
| Qwen3.6-35B-A3B | Graph | 1/1, 1/1, 8/8, 8/8, 8/8 | 26 |
| Kimi K3 | eager | 1/1, 1/1, 8/8, 8/8, 8/8 | 26 |
| Kimi K3 | decode Graph | 1/1, 0/1, 8/8, 8/8, 8/8 | 26 |

For the Qwen rows, serve each checkpoint with TP4, model length 32768,
`--disable-custom-all-reduce --trust-remote-code`, and
`--allowed-local-media-path` pointing to the gate images. Add
`--enforce-eager` for eager mode. Set the client model, served name and endpoint
to match, and run `tools/adaptation-gate-cases/run_test.sh` once per row.

K3 used the pinned Torch 2.10 / MACA 3.8 stack and TP8 × PP4. Its 118 integration
tests passed. The model gate counted 6,350 prompt tokens per mode, 728 output
tokens in eager and 745 in Graph. All 32 workers shut down gracefully in each
run. In Graph, single-image OCR case `image-05` returned `VLLM` with finish
reason `stop` and omitted `HELLO`. Both word regions influenced the vision
processor tensor. The cause of the omission during generation is unresolved.
The GPQA and ERQA results in the example refer to the previously evaluated
model revision; these gate requests did not rerun either quality benchmark.
