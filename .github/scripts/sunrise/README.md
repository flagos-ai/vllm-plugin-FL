# Sunrise vLLM 0.32 V2 Gate

The runner is derived from `codex/sunrise-vllm-0.28.0` at `b7b1258`,
`.github/scripts/sunrise/run_gate.py` (introduced by `f62fc27`). It invokes the
unchanged `tools/adaptation-gate-cases/run_test.sh` and records SHA256 hashes of
the shared clients and images. It does not change production model code.

Run each model in eager and graph modes, sequentially on the reserved devices.
For remote execution, launch the command inside a named tmux session and the
intended container, with its `/opt/sunrise-build/environment.sh` sourced and
`TANG_VISIBLE_DEVICES` explicitly set. An example inside that container is:

```bash
source /opt/sunrise-build/environment.sh
python .github/scripts/sunrise/run_gate.py \
  --repo "$PWD" \
  --model /data/models/Qwen/Qwen3.6-27B \
  --mode eager \
  --output-dir /data/codex/sunrise-032/gate/27b-eager
```

Repeat with `--mode graph`, then with
`--model /data/models/Qwen/Qwen3.6-35B-A3B` in both modes. Each output directory
must be new. The model directory must contain `config.json`, tokenizer data,
and the full weights. The standard acceptance configuration is TP2, context
8192, eight sequences, and 4 GiB KV cache per worker. Graph mode explicitly uses
compilation mode `0` (`NONE`), backend `eager`, `FULL_DECODE_ONLY`, and capture
sizes `[1, 2, 4, 8]`; both modes enable native `--cudagraph-metrics`.

To borrow a server that this task already started, add all three options:

```bash
--reuse-server-url http://127.0.0.1:18360 \
--served-model-name sunrise-gate-27b-eager \
--server-log /data/codex/sunrise-032/outputs/gate-27b-eager-preflight.log
```

The endpoint must be local and explicitly specify its port. Readiness checks
match both the served name and `/v1/models` model root. The log must identify the
actual Sunrise V2 runner and sufficient allocated KV token capacity. The runner
also reads `/server_info?config_format=json` when available and retains only the
necessary model/TP/scheduler/KV/compilation fields, never its environment data.
The endpoint is development-only in vLLM 0.32 and is normally absent; HTTP 404/405
uses the dedicated native server startup log instead, requiring explicit actual
APIServer fields and matching EngineCore configuration. Missing or conflicting
fields fail verification. `effective-configuration.json` records the observed
configuration; requested CLI values are never substituted for missing evidence.
The runner
never opens a borrowed log for writing or terminates a borrowed server. The
borrowed server must have been started with the same standard configuration and,
for graph mode, native graph metrics. An owned server and the shared test client
are cleaned up only by their task-created process sessions.

Each model/mode must pass all five shared scenarios: text single (1 request),
image single (1), text concurrency eight (8), image concurrency eight (8), and
mixed concurrency eight (4 text + 4 image). All 26 requests must pass all ten
shared quality checks, including expected semantics/order, minimum response
length, and repetition/encoding checks. These are deterministic semantic smoke
checks; no benchmark accuracy percentage or latency/throughput SLO is prescribed
by the shared Gate. Request latency and token usage remain in the raw reports.

Each model/mode also runs one supplemental context-retention request. A local
tokenizer builds a prompt of at least 4096 tokens and the returned server
`usage.prompt_tokens` must independently confirm at least 4096. The answer must
retain the requested passphrase. The original 104 shared requests remain intact;
the four supplements bring the full matrix to 108 requests.

Graph acceptance requires successful capture and native `FULL` execution rows
with positive counts emitted **after server readiness**, during real requests.
Capture messages alone, warmup-only replay, a graph-related environment variable,
or omission of `--enforce-eager` do not satisfy this check. The native V2 FULL
execution descriptor directly invokes the graph replay manager.

Validate all four output directories after the servers finish:

```bash
python .github/scripts/sunrise/verify_gate_matrix.py \
  /data/codex/sunrise-032/gate/27b-eager \
  /data/codex/sunrise-032/gate/27b-graph \
  /data/codex/sunrise-032/gate/35b-a3b-eager \
  /data/codex/sunrise-032/gate/35b-a3b-graph \
  --output /data/codex/sunrise-032/gate/matrix.json
```

Keep the server logs at their receipt paths until matrix verification completes;
the matrix validator re-reads post-ready native graph evidence. It also checks
that all four scenarios used the same shared dataset, revalidates every raw
shared report, and rejects incomplete or duplicate model/mode coverage. Each
receipt must include verified effective TP2/context8192/sequences8/KV4GiB and the
actual eager/V2/graph configuration, with its configuration evidence file. For
startup-log evidence, the validator re-parses the original log and compares the
values. Requested settings alone never satisfy matrix validation.

CPU-only acceptance contracts:

```bash
python -m pytest --noconftest -p no:cacheprovider \
  .github/scripts/sunrise/test_gate.py
```

厂商排查用的最小客户端、模型无关 TP2 collective 探针、原生 V2 模型入口及
脱敏关键日志见 [离线复现包](../../../tools/repro/sunrise_032/README.md)。
该包仅做离线检查；GPU 验证已停止，未执行 reset。