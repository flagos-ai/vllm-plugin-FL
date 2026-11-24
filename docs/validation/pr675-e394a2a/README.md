# S60 Adaptation Gate on RMS execution-mode fix

Tested plugin commit: `e394a2ac468a168fc864a9966b63e687d26dc496`.
All four groups completed with test and server exit code 0. Each used the V2
runner, TP2, and its own fresh compiler caches. Startup logs include cold
compilation and graph capture; these runs do not measure steady-state startup.

| Model | Mode | Scenarios | Requests | Passing checks |
|---|---|---:|---:|---:|
| Qwen3.6-27B | eager | 5 | 26 | 260 |
| Qwen3.6-27B | graph | 5 | 26 | 260 |
| Qwen3.6-35B-A3B | eager | 5 | 26 | 260 |
| Qwen3.6-35B-A3B | graph | 5 | 26 | 260 |

Each group contains single text, eight concurrent text requests, single image,
eight concurrent image requests, and eight mixed requests. No scenario was
skipped. The raw scenario JSON, manifest, pytest log, complete server log,
startup excerpt and environment record are included in each group directory.
Environment records retain the actual argv, selected environment variables,
installed package versions and model metadata hashes. Launch scripts record
commands and cache setup; their original absolute paths must be adjusted for
another host. `verification.json` preserves the original audit and paths.

Run the portable read-only audit with Python's standard library:

```bash
python docs/validation/pr675-e394a2a/verify_bundled_gate.py
```

It verifies inventory hashes, code SHA, exit codes, exact scenario/request
counts and all 1040 checks. It was also executed from a relocated copy under
`/tmp`; changing one quality check caused both integrity and quality failures.

Weight contents match immutable Qwen revisions
`1cda14dd71a3f47115850eb790f416f165da1cb5` (27B, 15 shards) and
`345d467b117d1fd5eda9b879a5988de9ae5f47a7` (35B-A3B, 26 shards).
These directories include local tokenizer/chat-template overrides and a 35B
configuration override. Their exact files and hashes are in `model-provenance`;
these are not unmodified upstream snapshots.

The original coordinator started redundant eager jobs after graph completion.
Those redundant jobs were stopped; the selected eager results come from the
independent run on devices 4–7. Graph used devices 0–3. This bundle selects only
the four complete groups and makes no claim that the coordinator's interrupted
aggregate job passed.

These are GCU functional/quality regressions. They do not establish NVIDIA
provider numerical parity, cross-platform CI coverage, or official upstream
operator acceptance. See the [review record](../../usage/enflame-pr675-review-status.md).

The installed `vllm-plugin-fl` distribution version in environment JSON predates
this checkout. The tested module was selected by `PYTHONPATH` pointing at the
recorded checkout, as shown in the launch scripts; package metadata alone is
not the tested plugin code identity. The manifests record that checkout's SHA.
Captured JSON/logs and third-party metadata are excluded from spelling checks
so opaque request IDs and original package descriptions remain byte intact.
