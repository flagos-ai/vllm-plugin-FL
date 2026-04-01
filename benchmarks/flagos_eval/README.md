# FlagOS Evaluation Suite

Evaluation toolkit for large language models with LM Eval and vLLM Benchmark.

## Quick Start

### 1. Dependencies

* **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)** 
    ```bash
    git clone https://github.com/EleutherAI/lm-evaluation-harness.git
    cd lm-evaluation-harness
    git checkout ee7e8f4fe58e13d6760c066474f0d01477317d1d
    pip install -e .
    pip install "lm_eval[hf,vllm,api]"
    pip install datasets==2.14.7
    ```

* **[vllm](https://github.com/vllm-project/vllm)**
    ```
    Version: 0.13.0
    Commit ID: 72506c98349d6bcd32b4e33eec7b5513453c1502
    ```

* **[vllm-plugin-FL](https://github.com/flagos-ai/vllm-plugin-FL)**
    ```
    Commit ID: af1e0b2adfb0061df9699b23e8837b36333cec41
    ```

* **Python 3.10+**

### 2. Run Evaluation

```bash
cd benchmarks/flagos_eval

# LM Evaluation (~35 minutes)
./run_eval.sh /path/to/model/ hf_xxxxxxxxxxxxx

# Performance Benchmark
./run_benchmark.sh /path/to/model/

# Mixed-Length Benchmark (Vendor Acceptance)
./run_mixed_benchmark.sh /path/to/model/         # default seed=42
./run_mixed_benchmark.sh /path/to/model/ 123      # custom seed
```

## Evaluation Tasks

### LM Evaluation

Tasks:
- **General**: BBH
- **Math**: GSM8K
- **Coding**: HumanEval, MBPP
- **Multilingual**: MGSM (Chinese)

Output Files:
- `results_summary.csv` - Summary
- `output/*/results*.json` - Detailed results

### Performance Benchmark

Metrics:
- **Throughput**: tokens/s, requests/s
- **Latency**: Mean, P50, P90, P99 (ms)

Output Files:
- `bench_summary.csv` - Summary
- `bench_results/*.json` - Detailed results

### Mixed-Length Benchmark

Runs throughput tests with mixed input lengths in shuffled order for vendor performance acceptance testing. Uses a seed to ensure reproducible shuffle results.

Configuration (1610 samples total):

| Input/Output | Samples |
|--------------|---------|
| 1k/1k       | 500     |
| 2k/1k       | 300     |
| 4k/1k       | 300     |
| 6k/1k       | 300     |
| 12k/1k      | 200     |
| 32k/1k      | 10      |

Output Files:
- `mixed_bench_summary.csv` - Summary
- `mixed_bench_summary.json` - Summary (JSON)
- `mixed_bench_results/*.json` - Detailed results

## Project Structure

```
flagos_eval/
├── run_eval.sh                      # LM evaluation script
├── run_benchmark.sh                 # Performance benchmark script
├── run_mixed_benchmark.sh           # Mixed-length benchmark script
├── collect_eval_results.py          # LM evaluation result collector
├── collect_benchmark_results.py     # Performance benchmark result collector
├── collect_mixed_benchmark_results.py # Mixed-length result collector
├── output/                          # LM evaluation results
├── bench_results/                   # Performance benchmark results
├── mixed_bench_results/             # Mixed-length benchmark results
├── results_summary.csv              # LM evaluation summary
├── bench_summary.csv                # Performance benchmark summary
├── mixed_bench_summary.csv          # Mixed-length benchmark summary
└── mixed_bench_summary.json         # Mixed-length benchmark summary (JSON)
```
