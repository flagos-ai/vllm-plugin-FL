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
