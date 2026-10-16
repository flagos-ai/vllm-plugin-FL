# MUSA ordinary-wheel image candidate

This private runtime build starts from the exact MUSA SDK 4.3.5 / Python 3.10 /
Torch and TorchMUSA 2.9 vendor RepoDigest recorded in `prepare_context.py`.
It copies six ordinary wheels into the image, installs them into
`/opt/musa/venv`, and keeps their whole SHA and installed RECORD evidence.
The context contains no checkout or host `PYTHONPATH`; `pip -e` is absent.
The active interpreter rejects inherited editable distributions, egg-links and
editable `.pth` hooks. A vendor base with any active editable package must be
corrected through an ordinary-wheel overlay before this image can pass.

## Actual inputs before building

Provide `stack-manifest.json` with schema 1, `ordinary_wheels_only: true`, the
fixed `base_image` RepoDigest/config digest, the actual 40-character
`plugin_commit`, matching `observer_commit`, five protected vendor files,
`vendor_versions`, a nonempty `source_lineage` and exactly six `wheels` rows.
Each row has `name`, `version`, `filename`, `bytes` and `sha256`; new FL and
observer rows additionally carry `source_commit` equal to `plugin_commit`.
Bind nonempty actual source-build receipts through `plugin_build_receipt` and
`observer_build_receipt` filename/byte/SHA fields. Review the actual clean checkout
and source proof before calling `make_stack_manifest.py`; that utility preserves
receipt fingerprints but does not rederive their source-commit claims.
The four prevalidated dependency pins are enforced in `verify_stack.py`.
The plugin and observer must be ordinary wheels built from that real committed
checkout; the historical CPU07 plugin/observer repacks do not stand in for the
new source commit.

FlagGems source lineage must name public base
`bfc6c49d2c4d06e5deac9784f203730c8372a84a`, three production hunks from PR #6894,
MUSA device/pin-memory fixes, MUSA NDmul singleton 512/4/3 with FlagTune expansion
disabled, and LibEntry native illegal-memory rethrow. The validated current
`musafix2` wheel is an ordinary repack of the prior source-built `musafix1`
wheel, with five changed wheel members and 5588 unchanged bodies; it is not a
new FlagGems source build or an upstream compiler fix.

[FlagGems PR #6894](https://github.com/flagos-ai/FlagGems/pull/6894) merged
on 2026-10-16 at
[`317b763204391d530d6024b52b6235de150443d8`](https://github.com/flagos-ai/FlagGems/commit/317b763204391d530d6024b52b6235de150443d8).
Its three split-KV changes are now upstream; the separate MUSA corrections
above remain outside that PR's scope. This status update leaves the published
image, embedded dependency manifest, bfc6 source lineage and wheel digests
unchanged. The fixed archive still needs its carried patch for reproduction;
a new upstream-based stack requires separately recorded builds and validation.

FlagTree lineage must name
`69148faf5cc03edb0a5737abd4b75b691132cea9`, the single header `#else` to
`#elif defined(__PROTON__)` change and explicit MUSA build profile
`TRITON_BUILD_FLAGPRISM=OFF` / `TRITON_BUILD_PROTON=OFF`. The validated
`profilefix1` wheel was built from that patched source; it is not a pristine
default-profile main wheel. Public patch manifests and reproduction instructions
are in `tools/musa_dependency_recipe`.

Create the manifest from the reviewed actual build receipts and six wheels,
then prepare an exclusive context in a private owned directory:
Place all six supplied wheel files, using their original filenames, in
`PRIVATE_WHEELHOUSE` before preparing the context. The manifest generator reads
and fingerprints the actual inputs; it does not copy those wheel files.

```bash
python3 -I -B make_stack_manifest.py --plugin-commit "$PLUGIN_COMMIT" \
  --plugin-wheel "$FL_PLUGIN_WHEEL" --observer-wheel "$GATE_OBSERVER_WHEEL" \
  --plugin-build-receipt "$FL_BUILD_RECEIPT" \
  --observer-build-receipt "$OBSERVER_BUILD_RECEIPT" \
  --dependency-wheelhouse "$PRIVATE_WHEELHOUSE" --empty-wheel "$EMPTY_VLLM_WHEEL" \
  --output "$STACK_MANIFEST"
python3 -I -B prepare_context.py --manifest "$STACK_MANIFEST" \
  --wheelhouse "$PRIVATE_WHEELHOUSE" --output "$PRIVATE_BUILD_CONTEXT"
MUSA_BUILD_CONTEXT="$PRIVATE_BUILD_CONTEXT" bash docker/build.sh \
  --platform musa --target ci --image-name harbor.baai.ac.cn/plugin/vllm-plugin-fl \
  --image-tag "v0.28.0-musa-ci-20261011-${PLUGIN_COMMIT:0:12}"
```

Use the printed Docker argv prefix plus `--tag "$UNIQUE_IMAGE_TAG"` and the
context path. Use the ordinary MUSA image registry namespace:
`harbor.baai.ac.cn/plugin/vllm-plugin-fl:v0.28.0-musa-ci-20261011-<commit12>`.
The current published tag and immutable RepoDigest are in the
[MUSA deployment guide](../../docs/usage/musa.md#ordinary-stack-image-and-ci-dispatch).
Run long remote builds in tmux and preserve stdout, stderr and exit files.
`--network=none` applies to Docker RUN instructions, and the build uses the
locally verified base with `--pull=false`. No source or wheel download occurs.
The public Dockerfile keeps the standard `dev`, `ci`, `release` target interface.
`docker/build.sh` verifies exactly ten context files against the reviewed recipe
and whole wheel pins, and rejects overriding its five bound MUSA build arguments.
Other backend contexts and build interfaces remain unchanged. MUSA dispatch uses
the previously tested `GEMS_VENDOR=mthreads` with `VLLM_VENDOR=musa`.

## CPU acceptance and publication

Inspect the completed image's OCI source/revision, stack manifest SHA and image
ID. Run `verify_stack.py installed` from a fresh container without GPU devices,
host source mounts or environment injection; preserve its output and exit 0.
The validator rechecks six whole wheels, every wheel/installed RECORD and payload,
ordinary direct URLs, private module origins, no active editable installs and
unchanged five vendor fingerprints. It performs no accelerator imports and
does not claim the new image's GPU Gate or CI passed.

Use the existing private registry authentication on the builder; inspect only
whether the target Harbor auth entry or credential helper exists. Do not print
Docker auth/config values or copy credentials into the build context. Root
publishes the exact unique image tag, captures its returned registry digest,
and binds CI/docs to that immutable digest after a successful push.

## CI handoff

The image exposes `MUSA_CI_EMPTY_WHEEL`, its exact SHA and the `latest`
dependency profile, `/opt/musa/dependency-manifest.json` and its literal SHA.
The CI configuration defaults to this published image and those five software
values; the deployment guide records their exact paths and whole-file hashes.
Hardware allocation still requires operator-supplied JSON for the executing runner.
That manifest contains exactly FlagGems/FlagTree/PyYAML and their source lineage.
CI rebuilds the checked-out FL and observer as ordinary wheels for its actual
merge/branch SHA and installs all six wheels in its fresh venv. A nested venv's
`--system-site-packages` does not inherit `/opt/musa/venv`; explicitly reinstall
the three image wheelhouse dependencies rather than relying on nested-venv paths.
CI's validator therefore needs the narrow `/opt/musa/wheelhouse/` allowed root,
in addition to its existing `/data/` interface, without weakening SHA, vendor or
RECORD checks. Sync exact image digest and manifest fields into `.github/configs/musa.yml`.

The full GPU Gate retains TP4/MCCL, FlagCX absent, eager utilization 0.6,
graph utilization 0.92 and sparse captures `[1,2,4,8,16,256,1024,4096]`,
max length 32768, batch 4096 and seq 8. The prior CPU07 four-route Gate passed
104 original requests, 1040 checks, 20 JUnit cases across 12 XML files and strict
52 graph-request / 208 rank joins. A newly committed image/CI run must still be
tracked on its own actual commit and immutable image digest.
