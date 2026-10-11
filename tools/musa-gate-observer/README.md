# Independent MUSA Gate observer

This is a separate ordinary test wheel in `tools/musa-gate-observer/`.
It observes original runtime calls and validates their event contract without
changing the FL plugin, vendor packages or original Gate quality rules. CPU
contract tests do not establish native, model or Gate acceptance; each installed
wheel and runtime result needs its own independently verified evidence.

Build/install this wheel only in the Gate job, using the already provisioned
offline build tools. No dependency download is implicit in the package.

```bash
python -I -B -m pip wheel --no-deps --no-build-isolation tools/musa-gate-observer -w OBSERVER_WHEELHOUSE
python -I -B -m pip install --no-index --no-deps OBSERVER_WHEEL
```

The actual wheel bytes, installed source bytes, normal package origins, candidate
wheel/commit, upstream wheel/commit and container/process identities need their
own independently verified manifest. Build commands are not installation proof.
`pyproject.toml` has one `vllm.general_plugins` entry and no automatic pytest
entry. Enable it only for a concrete Gate run:

```bash
VLLM_PLUGINS=fl,musa_gate_observer
MUSA_GATE_OBSERVER_CONFIG=/absolute/readonly/config.json
MUSA_GATE_OBSERVER_ROUTE_ID=ACTUAL_CONFIG_ROUTE
python -I -B -m pytest -sv -p musa_gate_observer.pytest_plugin ORIGINAL_TEST_TEXT_FILE
python -I -B -m pytest -sv -p musa_gate_observer.pytest_plugin ORIGINAL_TEST_IMAGE_FILE
python -I -B -m pytest -sv -p musa_gate_observer.pytest_plugin ORIGINAL_TEST_MIX_FILE
python -I -B -m musa_gate_observer.summarize --trace-dir ACTUAL_TRACE_DIR --expected-manifest /absolute/readonly/config.json --route ROUTE1 --route ROUTE2 --route ROUTE3 --route ROUTE4 --output NEW_EXCLUSIVE_SUMMARY
```

The server and client inherit the same read-only config and current route ID.
The runner supplies a known run ID, binds it to `config.run_id`, and creates a
new exclusive run root. `config.trace_directory` must equal that root's
`observer-traces` directory. The observer creates only exclusive per-process
JSONL files there. The summarizer runs once after all four routes have stopped.
Its four `--route` values must equal the complete config route set, without
duplicates. They cannot select a smaller matrix and produce matrix completion.

An explicit diagnostic single-route invocation uses `--partial-route ROUTE`
instead of all four `--route` values. The complete four-route immutable config
is still required; that trace directory must contain only the selected route.
Other-route input files, duplicate selection and mixed CLI modes fail. The
selected route still requires all 26 requests, five original scenario outputs,
one API, four workers and three original single-file clients, including complete
trace footers. A graph route additionally requires 26 genuine joins on all four
ranks. Its summary has `scope: route_only`, `full_matrix_complete: false`,
`observer_complete: false` and `graph_observation_complete: false`; only
`route_observer_complete` and, for a graph route,
`route_graph_observation_complete` may close. Eager graph applicability is
`not_applicable`. This diagnostic interface cannot complete the four-route CI
contract or manufacture traces for prior unobserved eager runs.

The config is an immutable **producer source/request manifest**, also used as
the summarizer's `--expected-manifest`. It is not an actual process attestation.
Do not populate future PID, startup, response, original result, capture or replay
values in it. The runner independently records actual route process identities
and rereads all original scenario JSON files after execution. It must compare
those records with the observer summary before claiming any runtime result.

## Explicit config contract

All paths below are actual, absolute canonical paths in the relevant container.
All fingerprints are actual whole-file bytes/SHA256, never labels or examples.
Missing, `UNBOUND`, mismatched or unsupported sources make observation fail.
An offline CPU binder may explicitly project verified read-only kit files from
a CPU `/baseline` alias to a future GPU `/task` alias. Such a config does not
prove that alias exists or is readable in a future container: each producer must
reread the full original kit at its bound actual path before activation.

* `schema_version`: `1`; `run_id`: nonempty known run ID.
* `observer_sources`: exact five package Python names (`__init__.py`, `trace.py`,
  `server.py`, `pytest_plugin.py`, `summarize.py`), each `{bytes, sha256}`.
* `observer_source_sha256`: SHA256 of `observer_sources` encoded using Python
  `json.dumps(sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)`
  and UTF-8. Each producer rereads these five installed sources.
* `source_binding_id`: actual 64-character lower-case SHA256 of the separately
  verified candidate/upstream/vendor/Gate source binding manifest.
* `trace_directory`: new owned absolute trace directory.
* `sources`: module-name mapping to exactly `{path, bytes, sha256}`. Producer
  modules must already have been loaded by the normal application/test path.
  No observer import loads a target module to satisfy this table.
* `required_role_sources`: keys `api`, `worker`, `client`, each a nonempty unique
  module-name list. Their actual fingerprints are recorded in activation.
* `required_role_hooks`: exact hook lists given below.
* `gate_kit_files`: exact nineteen original relative names, including `.gitignore`,
  eleven original text/script files and eight `images/image_01.png` through
  `image_08.png`, each `{path, bytes, sha256}`. All are actually read and matched,
  bounded at 1 MiB per file and 4 MiB total, without executing an unused test.
  `case_utils` and the three test entries in `sources` must equal their kit rows.
* `vendor_graph`: `{module: "torch_musa.musa_graph.graphs"}`. The actual source
  belongs in `sources` and the worker role source list.
* `routes`: exactly the two original Qwen3.6 model basenames, each eager and
  graph; exactly four routes. Each has `route_id` (safe file component), `model`,
  `mode`, `expected_ranks: [0,1,2,3]` and its original ordered 26 `requests`.
* Each request has globally unique `logical_id`, `original_request_id`,
  `scenario`, **whole-route zero-based** `ordinal` (exact `0..25`),
  `request_input_sha256` and exact original ten `required_checks`.
  Request input SHA hashes the complete original request dictionary using the
  same canonical JSON convention. Original image file URIs depend on the actual
  server-readable kit path; planned deployment paths cannot pass this check.

Original scenarios/counts are `text_single:1`, `text_concurrent_8:8`,
`image_single:1`, `image_concurrent_8:8`, `mixed_concurrent_8:8` per route.
Original checks are `non_empty`, `minimum_length`, `expected_semantics`,
`expected_order`, `no_bang_triplet`, `no_mojibake`, `no_control_characters`,
`no_long_character_run`, `no_repeated_word_run`, `no_repeated_phrase`.

Required API sources include `vllm.plugins`,
`vllm.entrypoints.openai.api_server`,
`vllm.entrypoints.openai.chat_completion.serving`,
`vllm.v1.engine.async_llm`, `vllm.v1.engine.input_processor`.
API hooks are `OpenAIServingChat.create_chat_completion`, `AsyncLLM.generate`,
`AsyncLLM._add_request`. The exact original async generator is returned without
proxying iteration. `_add_request` records the **actual** external and salted
engine IDs after the original enqueue returns; no ID-prefix join is accepted.

Required worker sources include `vllm.plugins`,
`vllm.v1.worker.worker_base`, `vllm_fl.worker.worker`,
`vllm_fl.worker.model_runner`, `vllm_fl.compilation.graph`,
`vllm.forward_context`, `torch_musa.musa_graph.graphs`.
Worker hooks are `runner.execute_model`, `GraphWrapper.__call__`,
`torch.musa.graph.__enter__`, `torch.musa.graph.__exit__`, `MUSAGraph.replay`,
`worker.shutdown`. This observer supports the real FL V1 `ModelRunnerFL`; another
runner is explicitly unsupported. Normal `WorkerWrapperBase.init_device`
returns before attaching these hooks, before subsequent model load/capture.

Required client sources include `case_utils`, `test_text`, `test_image`,
`test_mix_text_image`, `httpx._client`, `openai._base_client`.
Client hooks are `case_utils.run_case`, `case_utils.execute_request`,
`httpx.AsyncClient.send`. Preserve the original three separate pytest commands
per route; execute the remaining groups even if an earlier one fails. Each
activation lists its single actually collected `loaded_test_modules` value and
the whole statically verified `gate_kit_files` table. Only the collected original
test alias is replaced; the other two tests are not imported. The original
imported `run_case` alias must still refer to the original function at attachment.
Exactly one text client covers the two text scenarios, one image client covers
the two image scenarios and one mixed client covers the mixed scenario.

## Actual observation and limits

The semantic event carries run/source/route/model/mode, actual own PID,
`/proc` starttime and PID namespace, contiguous process-local sequence, event
UUID, wall/monotonic timestamps, kind/status and the complete dynamic data.
Activation is first and a zero-error telemetry footer is last.

Physical JSONL defines the eleven immutable process/source-binding fields once
in the first row's `profile`, with their canonical SHA256 in `profile_id`.
Later rows use that file-local reference. Every event UUID, sequence, timestamp,
kind, status and full data object remains recorded. The reader expands the
profile into the original eighteen-field event before applying the existing
schema, source, identity, role, sequence, closure and graph/request joins.
Legacy complete rows remain readable. Unknown or mismatched references,
redefined profiles and mixed compact/legacy files are rejected.

Each exclusively created process file has a finite **2 GiB** budget and
**256 KiB** event cap, with **64 KiB** reserved for a failure footer. The first
I/O, serialization or cap failure latches observation failure; later non-end
events are counted as dropped. Up to four bounded error samples and a total
error count remain, with one bounded stderr warning. A footer with errors or
dropped evidence has error status and cannot satisfy the original acceptance
checks. There is no event sampling, fabricated success or original trace rewrite.

Consumer limits are **128 files and 12 GiB total input**. These are finite
evidence-retention budgets, not a proof that every capture topology fits them
or a bound on RSS: the reader retains expanded events for joins, so memory can
exceed encoded input size. Files must remain regular, canonical and stable.
Missing footer, error/unsupported status, duplicate/nonfinite JSON, gaps, mixed
identities or cap failure rejects completion. Forced termination can lose a
footer and must remain incomplete.

A new ordinary observer wheel must regenerate RECORD and bind the actual five
installed source fingerprints in a fresh config. Reusing an older observer
manifest cannot validate changed source bytes. The separate optional sampling
or mul diagnostic plugin remains disabled by default; observer enablement does
not enable those diagnostics or change the original Gate requests or thresholds.

The actual vendor source observed in this task was `graphs.py`, 22300 bytes,
SHA256 `f4dbde4a5d224942cc2c6fffc174cb0bc9f34b297c58127c013c8099bfe7316c`,
at `/usr/local/lib/python3.10/dist-packages/torch_musa/musa_graph/graphs.py`.
It defines Python `graph.__enter__`, `graph.__exit__` and `MUSAGraph.replay`.
This is one measured profile, not an allowed hash for every CI image. The
observer verifies actual aliases/signatures, actual method replacement and
weakref support on a production-created graph. It creates no graph or stream for
a capability check. Native bodies, parameters, return values, stream behavior
and exceptions remain the original calls; observation errors are sticky.

Capture is recorded only after the original native context exit and the original
GraphWrapper return bind that exact weak-referenced graph to the actual
GraphEntry. Wrapper IDs and generation counters accompany object identities;
dead weak bindings are discarded, and live bindings have a finite cap of 4096.
Replay proof requires the same live graph/entry, batch descriptor and generation,
the current real execute_model scheduler IDs and a matching earlier capture in
that same worker process. Native warmup capture/replay with no scheduler request
is retained as unassociated diagnostic data and supplies no request proof.

Client contextvars attach original scenario/request identity to original SDK
`send` calls. These are **observed HTTPX send invocations**, not a guarantee of
every TCP, redirect or transport attempt. No retry setting, header, body, prompt,
sampling value, extra read or request outcome is changed. Original returned
request-row canonical SHA is compared with that row in the original scenario
JSON; the whole scenario JSON fingerprint is a separate value.

## Summary and independent acceptance

The summary requires exactly 104 original passing request rows, 20 original
scenario fingerprints, one actual API and three client processes per route, four
worker ranks per route, and unique API response/engine queue joins. Every one of
52 graph logical requests must have a capture/replay/scheduler/engine/API/client
join on each actual rank `[0,1,2,3]`. Time proximity, matching text labels and
unassociated native events cannot substitute for these joins.

`observer_complete` and `graph_observation_complete` describe the event contract
only. All `acceptance` fields (`source`, `GPU`, `ABI`, `model`, `Gate`) remain
false. Producer source acknowledgments and output fingerprints are claims until
the independent runner rereads the actual source/original JSON and verifies its
owned process attestation. A successful synthetic consumer fixture is neither
an observed graph nor a model/Gate result.

The independent aggregator must reread every raw input fingerprint, validate the
actual route/PID/starttime/ns/rank/session relationships, verify each original
JSON fingerprint, match its full `input` and every `output` row (canonical
`original_result_sha256`), and independently evaluate the unchanged original ten
checks and counts. It must retain all original exceptions, partial outputs and
cleanup failures. It must also preserve complete original process stdout/stderr
and reject observer I/O/close/unsupported diagnostic errors; the JSONL footer
cannot prove a subsequent OS close succeeded. Observer completion alone does
not authorize future actions.
