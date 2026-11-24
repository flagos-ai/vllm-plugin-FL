# Supplementary NVIDIA coverage

Once the workflow is on the default branch, dispatch **Supplementary coverage**
during an available NVIDIA runner window. To validate a candidate checkout, run `bash coverage_tests/run_nvidia.sh` in the NVIDIA CI image after the
standard NVIDIA check and setup steps. The entry point runs the core unit and
functional suites, then runs the supplementary suite in separate pytest
processes using the existing runner's worker and device settings.

The workflow archives baseline and combined coverage under
`coverage-results/`, together with their scope metadata and test logs.
Both measurements use the same checkout and coverage settings. The
platform-effective report counts shared Python code and the CUDA backend.

## Known gaps

- Qwen3.8 MTP coverage is skipped on the vllm 0.24 image; restoring the 168
  statements when the image ships vllm >= 0.33 is tracked in
  [#760](https://github.com/flagos-ai/vllm-plugin-FL/issues/760).
