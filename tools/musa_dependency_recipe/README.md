# Pinned ordinary-wheel source recipe

This bundle prepares reviewed FlagGems source in new private trees. It supports `latest-bfc6` and opt-in `latest-bfc6-ndsafe512` (Gems HEAD `bfc6c49d2c4d06e5deac9784f203730c8372a84a`) and retains `historical-cf174` as the default. It does not edit installed FlagGems, Torch, TorchMUSA or FlagTree files.

The replacements carry the three production changes from [PR #6894](https://github.com/flagos-ai/FlagGems/pull/6894), replace the shared public MHA property-query argument with `q.device`, and use the active runtime accelerator in `_pin_memory` while retaining explicit CPU rejection. The shared `flash_api.py` is the PR postimage plus the device-query correction, rather than an exact PR postimage. The enflame file is retained because it belongs to the reviewed PR; this does not claim that its backend was tested on MUSA. The entire PR head is not installed. The ineffective masked-Q padding experiment is excluded. The public-source accelerator checks are tracked in [FlagGems #6973](https://github.com/flagos-ai/FlagGems/issues/6973); issue state is not a Gate prerequisite, and the issue does not identify a cause of the model graph fault.

## File-by-file modification inventory

The [deployment guide's dependency inventory](../../docs/usage/musa.md#additional-flaggems-and-flagtree-modifications)
lists all six Gems production replacements and the one Tree header change.
The inventory distinguishes the three carried PR #6894 production changes,
the two accelerator/device corrections, the MUSA NDmul safety configuration
and native-error propagation, and Tree's optional-profiler build guard.
The NDmul workaround and header guard do not establish a compiler lowering
root-cause fix. The observed musafix2 Gems artifact is an ordinary-wheel
repack; Tree profilefix1 is an explicit patched source build. Keep those
operations and their actual wheel digests separate when preparing a new
ordinary stack.

## Prepare the latest private source trees

From the repository root, run `cd tools/musa_dependency_recipe`. The four raw postimages are in `patch-inputs/`; `production.patch` and the fixed manifests are beside the script. Use Python 3.10+ and an existing private parent; both output leaves must be new. The helper uses only the standard library and imports no accelerator packages.

```bash
curl --fail --location --output FlagGems-bfc6c49.tar.gz \
  https://api.github.com/repos/flagos-ai/FlagGems/tarball/bfc6c49d2c4d06e5deac9784f203730c8372a84a
printf '%s  %s\n' \
  1c84a39d25e5cf2650e5c82218eeccb863ffa9465ad114e03b2a59809b497868 \
  FlagGems-bfc6c49.tar.gz | sha256sum --strict -c -
python3 -I -B prepare_private_source.py --profile latest-bfc6 \
  --archive FlagGems-bfc6c49.tar.gz --output "$PRIVATE_PARENT/master-source" --variant master
python3 -I -B prepare_private_source.py --profile latest-bfc6 \
  --archive FlagGems-bfc6c49.tar.gz --output "$PRIVATE_PARENT/corrected-source" --variant corrected
```

The [latest Gems manifest](source-patch-manifest-latest-bfc6.json) pins the 14,003,066-byte archive, 6606 members, 6359 regular files and 53,856,950 expanded bytes. Both trees are extracted independently. Each binding records complete original/final inventories; corrected source changes exactly four files and preserves the other 6355 byte for byte. The latest archive's four preimages match the historical profile's preimages, so the same reviewed postimages are retained; new upstream bytes are rejected rather than silently patched. On Windows, use an extended-length output path when a private parent plus archive member would exceed the OS path limit.

The historical profile remains available through `--profile historical-cf174` or by omitting `--profile`. Its unchanged [manifest](source-patch-manifest.json) pins commit `cf174ab70ab2af934158c83385458fcc9c830712`, archive SHA `366c9d2d712a6c409482c4e0d0743b9c51db71f4a09787c12542a8d6892e43ae`, 6344 regular files and 6340 unchanged corrected-source files. Use its own archive and assigned versions; the latest profile cannot accept the historical archive.

## Build ordinary Gems wheels in a CPU environment

Use a private offline Python 3.10 builder with locked setuptools 76.1.0, setuptools-scm 9.2.2, wheel 0.46.2, packaging 26.2 and tomli 2.2.1. Expose no GPU nodes and preserve the SDK/vendor and builder inputs before/after. `BUILDER_PYTHON` names that interpreter.

```bash
mkdir "$PRIVATE_PARENT/master-wheels" "$PRIVATE_PARENT/corrected-wheels"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.4.0+archive.gbfc6c49d2c4d \
  "$BUILDER_PYTHON" -I -B -m pip --isolated wheel \
  --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$PRIVATE_PARENT/master-wheels" "$PRIVATE_PARENT/master-source/FlagGems"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix1 \
  "$BUILDER_PYTHON" -I -B -m pip --isolated wheel \
  --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$PRIVATE_PARENT/corrected-wheels" "$PRIVATE_PARENT/corrected-source/FlagGems"
sha256sum "$PRIVATE_PARENT/master-wheels/"*.whl "$PRIVATE_PARENT/corrected-wheels/"*.whl
```

The explicit SCM override assigns archive versions; it is not derived from a Git checkout, and no fake Git repository is created. Record actual rebuilt wheel digests: binary reproducibility is not promised. The executed latest CPU build produced these artifacts:

| Variant | Explicit version | Bytes | SHA256 |
| --- | --- | ---: | --- |
| Master | `5.4.0+archive.gbfc6c49d2c4d` | 12,056,408 | `4e097db6f1c1785fad02bd0bd73d3f846340323de858c1dece17400254d44457` |
| Corrected | `5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix1` | 12,057,042 | `9226df1ca704fa76e10f5abc3145d7b20f6f4c1b6c8e9d69b97bcf8e0d46b9e8` |

Validate canonical ordinary ZIP members, METADATA version, `py3-none-any` tag and complete regular-file RECORD hashes/sizes. Normalize dist-info version directories for the A/B comparison: only four production files, `_version.py` and the explicit METADATA Version line may differ; each RECORD is checked independently. The historical CPU27 master/corrected artifacts remain respectively 12,013,388 bytes/`155e8adbdea513f7466dacd04eb60669b34b9d2c39f44d805b01802e336abadd` and 12,014,024 bytes/`05b5fdb2d652b5a7ed8bfc12b3dc6d11b801ea982cb75ccaf563935bec97a7b5`, with the separate `gcf174ab70ab2` assigned versions.

## Opt-in MUSA NDmul safety profile

`latest-bfc6-ndsafe512` adds a narrowly scoped workaround to the same bfc6 archive; it does not change the historical default or `latest-bfc6`. The [six-file manifest](source-patch-manifest-latest-bfc6-ndsafe512.json) retains the four reviewed corrections and adds [the two-file patch](musa-nd-safe512.patch). Only the shared MUSA `mul_generic_nd_kernel` uses BLOCK_SIZE 512, four warps and three stages, with its FlagTune expansion disabled. The kernel body, first-compile lock, other mul kernels and other backends are retained. Known MUSA illegal-memory errors propagate instead of being treated as a failed compilation and followed by another configuration. This is a safety workaround, not a compiler root-cause fix. The isolated fixed-1024 launch evidence is tracked in [FlagTree #1346](https://github.com/flagos-ai/FlagTree/issues/1346), and native-error handling is tracked in [FlagGems #7005](https://github.com/flagos-ai/FlagGems/issues/7005). These reports have separate scopes from the accelerator checks in #6973 and the split-KV corrections in PR #6894.

```bash
python3 -I -B prepare_private_source.py --profile latest-bfc6-ndsafe512 \
  --archive FlagGems-bfc6c49.tar.gz --output "$PRIVATE_PARENT/ndsafe-source" --variant corrected
mkdir "$PRIVATE_PARENT/ndsafe-wheels"
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLAG_GEMS=5.4.0+archive.gbfc6c49d2c4d.pr6894.musafix2 \
  "$BUILDER_PYTHON" -I -B -m pip --isolated wheel \
  --no-index --no-deps --no-build-isolation --no-cache-dir \
  --wheel-dir "$PRIVATE_PARENT/ndsafe-wheels" "$PRIVATE_PARENT/ndsafe-source/FlagGems"
```

This private source preparation changes exactly six original files and preserves the other 6353 byte for byte. Use the unchanged Tree69148 explicit profile below. Record actual rebuilt wheel digests; the observed CPU07 artifact came from an ordinary musafix1 wheel repack, not from this source-build command. CPU07 verified the musafix2 ordinary wheel and all seven private installed `RECORD` files; its six other wheel bodies and vendor/image files stayed unchanged. The observed Gems wheel is 12,057,216 bytes/SHA `0bdf03d1730885a502c05995ba0d16f89b0f0a6fe8584d24de76dc6e58c69e71`. The repack changed only mul.py, libentry.py, version assignments, METADATA and RECORD; its other 5588 normalized member bodies matched. These CPU/static checks do not establish native numerical, model Gate or compiler acceptance. Keep diagnostics off unless explicitly running a separate diagnostic.

Separate runtime67r2 numerical diagnostics passed all four native/public
cases on the recorded 27B and 35B geometries, using synthetic tensor values.
The public entry used block 512 with zero benchmark calls; the child checked
the complete independent CPU oracle, output sentinels and unchanged inputs.
These synthetic checks establish that numerical scope, not model Gate or
compiler root-cause acceptance.

For the current model validation, use the [runner instructions](../../docs/usage/musa-ci-gate-runner.md#cpu07-ndmul-safe512-candidate).
The source-profile name differs from the generic CI interface
`MUSA_CI_DEPENDENCY_PROFILE=latest`: supply exactly three verified ordinary
dependency wheels, vendor fingerprints and source-build/repack lineage. CI setup requires committed
FL/observer source matching `MUSA_CI_EXPECTED_COMMIT` and `GITHUB_SHA` when set;
CPU07 is an ordinary repack, not a new Git commit.
Actual CPU07 model validation completed the unchanged 35B-A3B eager68r2,
27B eager70, 27B graph80 and 35B-A3B graph81 routes. The graph runs used sparse
capture sizes `[1,2,4,8,16,256,1024,4096]`, utilization 0.92, maximum 32768
tokens, batch4096 and seq8; eager runs retained utilization 0.6. All 104/104
original logical results and 1040/1040 checks passed, with 20 passing JUnit
cases across 12 XML files and 32 normal process ends. The unchanged strict
full-matrix consumer exited 0 with 52 graph-request/208 rank replay joins and
no errors. **This exact four-route CPU07 Gate is PASS.**

The four-case runtime67r2 numerical scope above uses recorded geometries and
synthetic values. Model/observer validation does not establish a compiler
root-cause fix, broad ABI compatibility or arbitrary memory fit. Earlier
default16 replay-coverage failure, runtime76 missing-literal quality failure
and runtime74/77/78 startup model-NOT_RUN receipts remain in the
[runner history](../../docs/usage/musa-ci-gate-runner.md#earlier-cpu07-trials).
GitHub CI remains NOT_RUN and automatic discovery remains disabled.

## Explicit latest FlagTree MUSA core profile

Use public Tree commit `69148faf5cc03edb0a5737abd4b75b691132cea9` plus [the single-header patch](flagtree-musa-proton-off.patch). The [latest Tree manifest](flagtree-source-patch-manifest-latest-69148.json) pins the source/archive, exact header pre/post images, reviewed build inputs and actual wheel. The branch before six optional Proton headers changes only `#else` to `#elif defined(__PROTON__)`; MUSA registration and the FlagPrism/Proton-enabled branches retain their text. This explicitly sets `TRITON_BUILD_FLAGPRISM=OFF` and `TRITON_BUILD_PROTON=OFF`; it is not a pristine/default main wheel. The original `ba25a9551b2e5157d843affe6f4c2cada1dcfde8` build boundary and guard are tracked in [FlagTree #1344](https://github.com/flagos-ai/FlagTree/issues/1344).

For a new private checkout, set `RECIPE_DIR` to this bundle and `PRIVATE_TREE_SOURCE` to that checkout:

```bash
test "$(git -C "$PRIVATE_TREE_SOURCE" rev-parse HEAD)" = \
  69148faf5cc03edb0a5737abd4b75b691132cea9
printf '%s  %s\n' \
  5a4868d5eb23cb81c5e795412cfdb5ca334224bae2352d3a40af99e79f566765 \
  "$PRIVATE_TREE_SOURCE/third_party/mthreads/bin/RegisterTritonDialects.h" | sha256sum --strict -c -
git -c core.autocrlf=false -C "$PRIVATE_TREE_SOURCE" apply --check "$RECIPE_DIR/flagtree-musa-proton-off.patch"
git -c core.autocrlf=false -C "$PRIVATE_TREE_SOURCE" apply "$RECIPE_DIR/flagtree-musa-proton-off.patch"
printf '%s  %s\n' \
  98e3daec66874141c6628e3d8b1e39e36adb6c26f0a42a34f53de372a29d71d0 \
  "$PRIVATE_TREE_SOURCE/third_party/mthreads/bin/RegisterTritonDialects.h" | sha256sum --strict -c -
```

Prepare the manifest-pinned MUSA LLVM/local-binary/Triton dependency bundles and locked build tools offline. Export `LLVM_SYSPATH` as the verified LLVM root and `JSON_SYSPATH` as the verified JSON include root. Export private `TRITON_HOME`/`FLAGTREE_CACHE_DIR` paths containing the verified build-dependency and MUSA local-binary payloads, and prepend the Tree builder's bin directory to `PATH` for its locked build tools. Do not overwrite the runtime's installed compiler. The ordinary wheel build uses:

```bash
FLAGTREE_BACKEND=mthreads TRITON_BUILD_FLAGPRISM=OFF TRITON_BUILD_PROTON=OFF \
TRITON_OFFLINE_BUILD=ON MAX_JOBS=8 CMAKE_BUILD_PARALLEL_LEVEL=8 \
FLAGTREE_WHEEL_VERSION=0.7.0+mthreads3.6.git69148faf.profilefix1 \
  "$TREE_BUILDER_PYTHON" -I -B -m pip --isolated wheel \
  --no-index --no-deps --no-build-isolation \
  --wheel-dir "$PRIVATE_PARENT/tree-wheels" "$PRIVATE_TREE_SOURCE"
```

Preserve the explicit profile, single-header proof, source inventories, actual wheel/tag/SHA and complete RECORD checks. The executed CPU build produced `0.7.0+mthreads3.6.git69148faf.profilefix1`, `cp310-cp310-linux_x86_64`, 180,126,552 bytes/SHA `57fd36f4c9be804bb9ab0af2103316746be4c45bc14fe370b543ee0d9c3d1d7e`. Its original archive joins 6463 official Git members (6455 regular files and eight symlinks); the current receipt does not provide the older `ba25a9551b2e5157d843affe6f4c2cada1dcfde8` post-build all-other-files proof. Static origins used `PathFinder`, not target imports; CPU build success establishes neither default-profile nor ABI/GPU/model acceptance.

The unchanged [historical Tree manifest](flagtree-source-patch-manifest.json) retains base `ba25a9551b2e5157d843affe6f4c2cada1dcfde8`, version `0.7.0+mthreads3.6.gitba25a955.profilefix1`, 180,144,953-byte wheel/SHA `e11252f44e39742553ca52b267d617cae666ee9129da57fae91f76d774a72019`, and its separately verified 6461 unchanged post-build originals. That historical proof does not transfer to the later HEAD.

## Install a new ordinary runtime

Create a fresh private venv from the fixed Python 3.10 SDK/TorchMUSA interpreter. Supply actual verified empty vLLM 0.28.0, current FL plugin, current Gate observer, corrected Gems, patched MUSA Tree and PyYAML 6.0.1 ordinary wheels. The current FL source additionally retains/stages MUSA capture inputs and splits `vllm::all_reduce`; build it from the reviewed repository source, rather than silently reusing the historical FL wheel. The current observer's [format/capacity contract](../musa-gate-observer/README.md) preserves strict semantic joins. An optional seventh ordinary diagnostics wheel remains disabled by default.

```bash
"$RUNTIME_PYTHON" -I -B -m venv --system-site-packages --without-pip "$PRIVATE_PARENT/runtime/venv"
"$PRIVATE_PARENT/runtime/venv/bin/python" -I -B -m pip --isolated install \
  --no-index --no-deps --ignore-installed --no-cache-dir --no-compile \
  "$EMPTY_VLLM_WHEEL" "$FL_PLUGIN_WHEEL" "$GATE_OBSERVER_WHEEL" \
  "$CORRECTED_GEMS_WHEEL" "$MUSA_FLAGTREE_WHEEL" "$PYYAML_WHEEL"
```

Validate every installed RECORD/payload, direct-url wheel SHA, private module origin and version. Recheck SDK/Torch/TorchMUSA/vendor fingerprints unchanged. The CI interfaces are `MUSA_CI_DEPENDENCY_PROFILE=latest`, `MUSA_CI_DEPENDENCY_MANIFEST` and `MUSA_CI_DEPENDENCY_MANIFEST_SHA256`; the manifest must use the actual corrected Gems/Tree/PyYAML wheel pins and fixed vendor fingerprints. Source lineage requires preserved source/build receipts; the manifest alone does not derive it. Baseline remains the CI default.

## Validation scope

CPU06 successfully installed seven ordinary wheels; only the two latest dependency wheels replaced CPU05, and the other five wheel bodies/vendor fingerprints remained unchanged. CONFIG06 bound seven metadata rows, the original 19 Gate files and four routes × 26 requests/four ranks. The owned CPU objects were stopped and removed with evidence/environment retained. These installation receipts are CPU/static evidence; the historical CPU06 musafix1 four-route GPU runs have 87/104 quality passes and strict full-matrix FAIL, with that profile's 35B-A3B eager unresolved (see the runner results). Automatic discovery remains disabled and GitHub CI is NOT_RUN.

Earlier operator evidence used source `7599dd1` and published MUSA Tree `0.7.0+mthreads3.6`: pin-memory controls passed 44/44; carrying PR #6894 changed the same suite from 10/36 to 30/36, including all 24 direct and six graph cases. Six public MHA numerical failures remained, and Q padding did not improve them. Those results do not establish acceptance on the new HEADs. The historical cf174/`ba25a9551b2e5157d843affe6f4c2cada1dcfde8` migrated baseline remains full Gate FAIL (35/104 quality passes). Detailed runtime54r2/Tiny58/runtime60 and original stream/cleanup boundaries are preserved in [the runner validation history](../../docs/usage/musa-ci-gate-runner.md#validation-history-and-candidate-boundaries). No requests, quality thresholds or observer acceptance rules are relaxed.

The six portable profile/failure tests run with `python3 -I -B tools/musa_dependency_recipe/test_prepare_private_source.py` from the repository root. Actual source-preparation checks are separate from wheel building or accelerator execution.
