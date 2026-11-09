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

K3 requires the pinned vLLM source and MetaX runtime. The retained KDA precision
and stride-safe MLA fallbacks are part of the accepted numerical contract.
Full quality scores and serving measurements belong to their recorded version;
a cleanup does not rerun or update those results. Detailed run evidence is
maintained with the PR and project reports.
