# MetaX baked FlagCX release verification

The hardware run uses plugin commit `914dd1d30f2212c8cee3184a3dc3cabaa44409a2`,
merged with `0.5.0-dev` commit `045afdc496e33cbbf803235632b32303f0452d0c`.

The published image is
`harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.32.0-metax-flagcx0130-release-914dd1d@sha256:b8579e0baa84898822e7592580c360e2d1405ba7b67d231f3d99740f6e03df4f`.
Its platform manifest and config digests are recorded separately in the summary.

- All four original TP4 adaptation Gate profiles passed, covering both checkpoints in eager and PIECEWISE modes. The total is 20 scenarios, 104 logical responses and 1,040 quality checks.
- Four native collective cases passed on each of four C550 ranks, including live-input graph replay against CPU references.
- The fresh network-isolated container installed only the plugin. FlagCX, FlagGems-vLLM, FlagGems and FlagTree came from the image.
- A second physical host pulled the published digest and verified the native core and wrapper hashes.

[Summary](summary.json), [runtime provenance](runtime.json),
[checkpoint metadata](model-snapshots.json), and [file hashes](sha256.json)
accompany the original per-scenario JSON, client logs and compressed server logs.
The selected server observations retain the Torch bootstrap fallback warning,
FlagCX initialization, actual V2 runner, graph capture and error lines.

The 35B cached model files record revision `1a5ae24e867f8d82388070d3f61590158a01d15c`.
The 27B local snapshot has unavailable repository revision metadata. Its
configuration, index and tokenizer SHA256 values identify the recorded files.

Controller and container exit statuses are recorded. Serving worker exit statuses
have no separate process-level receipt in this test. The first 27B image request
used the standard client timeout and default SDK retries during cold tuning.
