# S60 integration evidence

Tested code: `262f814db849e4687bdabd5417832ed9cf83265a`, integrating
`0.5.0-dev@77f141a961d75e54c0cf209de80c99a7c7eec04b`.

| Model | Mode | Requests | Quality checks | Skips |
|---|---|---:|---:|---:|
| Qwen3.6-27B | eager | 26 | 260/260 | 0 |
| Qwen3.6-27B | graph | 26 | 260/260 | 0 |
| Qwen3.6-35B-A3B | eager | 26 | 260/260 | 0 |
| Qwen3.6-35B-A3B | graph | 26 | 260/260 | 0 |

Each group ran text single/8 concurrent, image single/8 concurrent and mixed
8 concurrent. `results/` contains the original per-request JSON; `client.log`
and `server.log` contain real execution output. `launch_matrix.py` preserves
the actual environment and commands. Its workspace/model paths are those of
the test host, not a portable installation script. Verify the persisted counts:

```sh
python docs/validation/pr675-262f814/verify_gate_counts.py
```

This verifier reads artifacts relative to itself and asserts request/check
counts. `status.json` reports each client exit code. `environment.json`,
`pip-freeze.txt`, and `dpkg-toolkit.tsv` describe the **development container**.
They do not establish the contents or provenance of the published image digest.
The local model paths do not disclose immutable upstream revisions; metadata
SHA256 hashes are included instead, explicitly not equivalent to revision IDs.

Runner selection was automatic (no V1 override). On this integration the worker
imports upstream `vllm.v1.worker.gpu.model_runner.GPUModelRunner` for V2;
`model_runner.py` messages in the server logs identify that path. Do not label
this result `ModelRunnerFL` or infer that every unset-variable configuration
chooses V2. Graph means the launch configuration and runtime graph path used
here, not independent coverage of every graph mode or RMS provider contract.

Host regression logs: 52 passed; separate Inductor regression log: 86 passed.
These are not replacements for upstream NVIDIA precision or other-platform CI.
The 35B graph server emitted compiler diagnostics during startup, subsequently
served every gate request successfully; the diagnostics are retained, not hidden.
