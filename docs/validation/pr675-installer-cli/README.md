# Provider installer CLI regression

Failure source: PR675 CI run `44693787363`, Kunlunxin functional job
`136146414827`, setup step. The job failed before model execution with
`install_flaggems_vllm.py: error: unrecognized arguments: --source-archive`.

The script parsed arguments twice: its outer parser accepted the archive flag,
then `main()` parsed the same process arguments without that option. Conversely,
the outer parser rejected `--platform`. The fix gives `main()` sole ownership of
both options and retains archive SHA256 verification and runtime dependency
invariance checks. Combining Enflame's distinct pin with the shared archive is
explicitly rejected instead of installing the wrong provider.

Regression: `tests/unit_tests/test_provider_installer_cli.py` runs the script's
`__main__` entry point with real argv, mocking package metadata and installation
only. It covers default and Kunlunxin environments, default/archive/Enflame modes,
archive installation's pinned URL and SHA256, and invalid archive combinations.
It does not claim that an actual provider installation or hardware functional
run passed.

Commands from repository root:

```sh
python -m pytest --noconftest -c /dev/null -q tests/unit_tests/test_provider_installer_cli.py tests/unit_tests/test_flaggems_vllm_provider.py -k 'not fused_experts'
python .github/scripts/install_flaggems_vllm.py --help
```

`installer-cli-before.log` records 5 failures and 2 passes before the installer
fix. `installer-cli-after.log` records 12 passes and 1 deselected test after the
fix and added rejection cases. The deselected fused-experts test is unrelated
to installer CLI behavior and remains included in the full applicable unit run.

Full applicable local unit regression:

```sh
python -m pytest -q tests/unit_tests --ignore=tests/unit_tests/dispatch/test_ascend_vocab_parallel_embedding.py
```

Observed result: **2579 passed, 293 skipped, 5 warnings in 96.32 s**.
The explicit single-file exclusion requires unavailable `torch_npu` in this
S60 environment; it is not counted as a passing Ascend test. Raw output is
`installer-full-unit.log`. This result does not establish remote CI success.
