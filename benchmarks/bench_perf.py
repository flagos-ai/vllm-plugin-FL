#!/usr/bin/env python3
import subprocess
import os
from datetime import datetime

# ====== 配置区 ======
HOST = "0.0.0.0"
PORT = 8000
ENDPOINT = "/v1/chat/completions"
BACKEND = "openai-chat"

# 新的 7 组评测场景: (name, input_len, output_len, concurrency)
SCENARIOS = [
    ("p128d128",     128,    128,   100),
    ("p6144d128",    6144,   128,   100),
    ("p30720d128",   30720,  128,   100),
    ("p128d6144",    128,    6144,  100),
    ("p6144d6144",   6144,   6144,  100),
    ("p30720d6144",  30720,  6144,  100),
    ("p4096d2048",   4096,   2048,  64),
]

LOG_DIR = "vllm_bench_logs"
os.makedirs(LOG_DIR, exist_ok=True)

NUM_RUNS = 4  # 改为运行 4 次：run1 ~ run4
# ====================

def run_benchmark(name, input_len, output_len, concurrency, run_id):
    num_prompts = concurrency
    cmd = [
        "vllm", "bench", "serve",
        "--host", HOST,
        "--port", str(PORT),
        "--backend", BACKEND,
        "--model", "Qwen3-Next-Metax",
        "--tokenizer", "/share/Qwen3-Next-80B-A3B-Instruct",
        "--dataset-name", "random",
        "--endpoint", ENDPOINT,
        "--ignore-eos",
        "--trust-remote-code",
        "--random-input-len", str(input_len),
        "--random-output-len", str(output_len),
        "--num-prompts", str(num_prompts),
        "--max-concurrency", str(concurrency)
    ]

    log_file = os.path.join(LOG_DIR, f"{name}_run{run_id}.log")
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 🚀 Starting scenario: {name} (Run {run_id})")
    print(f"    Input: {input_len}, Output: {output_len}, Concurrency: {concurrency}")
    print(f"    Logging to: {log_file}")
    print(f"    Command: {' '.join(cmd)}\n")

    with open(log_file, "w") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)

    status = "✅ Success" if result.returncode == 0 else "❌ Failed"
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {status}: {name} Run {run_id} (exit code: {result.returncode})\n")

def main():
    print(f"🧪 Starting vLLM benchmark suite for {len(SCENARIOS)} scenarios, each repeated {NUM_RUNS} times...\n")
    for name, inp, out, conc in SCENARIOS:
        for run_id in range(1, NUM_RUNS + 1):
            run_benchmark(name, inp, out, conc, run_id)
    print("🏁 All scenarios and runs completed. Logs saved in:", LOG_DIR)

if __name__ == "__main__":
    main()
