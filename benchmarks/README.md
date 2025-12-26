To use the benchmark_throughput_flagos feature from vllm-plugin-fl, you must first complete the following preliminary steps:

1. Start an LLM inference service compliant with the OpenAI API protocol using the --served-model-name Qwen3-Next argument, or use a different name and update the string on line 11 of benchmark_throughput_flagos.py to match your chosen --served-model-name exactly (character-for-character).

2. Run the benchmark_throughput_flagos script on the same machine where the inference service is hosted.

3. Ensure the host machine has stable global network bandwidth of at least 10 Mbps, with 100 Mbps recommended for reliable benchmarking.

4. The benchmark_throughput_flagos feature has been validated in environments using vLLM versions 0.11.0 and 0.12.0. Using higher or lower vLLM versions—or serving the model with alternative frameworks such as SGLang—may result in compatibility issues.

Once the above prerequisites are met, you can simply run the following command from the directory containing this file: 

```bash
python3 benchmark_throughput_flagos.py
```

If the command runs successfully, it will generate a directory named vllm_bench_logs in the current working directory. After the execution completes, follow these two steps to verify the benchmark ran correctly and to obtain the performance evaluation results: 

1. Check for failed requests by running the following command in the current directory:
```bash
grep "Fail" -rn vllm_bench_logs
```

All matching lines should report zero failed requests (i.e., Fail: 0).

2. Generate performance statistics by executing:
```bash
python3 benchmark_throughput_flagos_statistics.py
```
This will output the final performance evaluation results based on the collected logs.
