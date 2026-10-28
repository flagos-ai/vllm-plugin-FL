# BGE-M3 joint output example

See [the model guide](../../docs/models/bge_m3.md) for the official output
contract, operator pins and platform requirements. From the repository root,
load the validated vendor runtime, then install the fast pooling serializer
in the environment that runs the server:

```bash
python -m pip install -e '.[pooling]' --no-build-isolation
export VLLM_PLUGINS=fl,bge_m3
export VLLM_FL_CONFIG="$PWD/vllm_fl/dispatch/config/bge_m3_metax.yaml"
vllm serve /models/bge-m3 --config examples/bge_m3_joint/serve.yaml
```

vLLM 0.24.0 automatically selects its existing ORJSONResponse when orjson is
installed. Restart a previously running server after installation. The measured
runtime used orjson 3.11.4. `serve.yaml` uses FP32/TP1/eager, max_num_seqs=64
and max_num_batched_tokens=32768. Load the model once and select any combination
of dense, sparse and ColBERT outputs in each request.

## Client

The client also needs orjson and requests. Parse the complete response bytes:

```python
import orjson
import requests

response = requests.post(
    "http://localhost:8000/pooling",
    json={
        "model": "bge-m3",
        "task": "plugin",
        "data": {
            "sentences": ["How does retrieval work?"],
            "return_dense": True,
            "return_sparse": True,
            "return_colbert_vecs": True,
            "max_length": 8192,
        },
    },
    timeout=600,
)
response.raise_for_status()
outputs = orjson.loads(response.content)["data"]
```

## Full-response benchmark

The server is managed separately. The benchmark additionally uses NumPy and
Transformers from the model runtime. Supply a JSON fixture
`{"texts": ["...", "..."]}`; every text must tokenize to exactly the requested
length, including special tokens. The script cycles the fixture to 128 requests,
verifies lengths before timing, and records fixture hashes.

Run the lengths separately, with different output directories:

```bash
python examples/bge_m3_joint/benchmark.py \
  --tokenizer /models/bge-m3 --input /data/input-1024.json --length 1024 \
  --concurrency 64 --requests 128 --output /data/results-1k
python examples/bge_m3_joint/benchmark.py \
  --tokenizer /models/bge-m3 --input /data/input-4096.json --length 4096 \
  --concurrency 64 --requests 128 --output /data/results-4k
```

Each invocation runs dense, sparse, ColBERT and joint in that order, each with
one warmup and one formal round. `--modes joint` selects one mode;
`--json-parser stdlib` retains the old requests.json path for controlled
comparisons. The default is orjson, and a missing dependency fails visibly.

Timing includes HTTP transfer, complete JSON parsing, and shape/finite checks.
Per-request latency excludes the client executor queue and includes server
queueing. Raw request timestamps verify actual HTTP concurrency; failures or a
concurrency peak below the requested value fail the run. There are no retries.
Save the launch configuration, dependency versions, device-memory measurements
and tuning-cache state separately. For stable-state comparisons, use equivalent
prepopulated caches and record any new tuning entries; this is different from
cold-start/out-of-box performance. Repeat independent warmup/formal pairs and
report all results, not the best one.
