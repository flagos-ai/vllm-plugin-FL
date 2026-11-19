"""64K×1 only: actual GDN inputs, teacher-forced logits, free generation, retrieval."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = ROOT / "correctness.json"
    if output.exists():
        raise FileExistsError(output)
    from vllm import LLM, SamplingParams

    settings = dict(
        model="/data/model",
        tensor_parallel_size=4,
        dtype="bfloat16",
        enforce_eager=True,
        enable_prefix_caching=False,
        seed=42,
        max_model_len=66560,
        max_num_batched_tokens=65536,
        max_num_seqs=64,
        kv_cache_memory_bytes=4294967296,
        gpu_memory_utilization=0.85,
        attention_config={"backend": "FLASH_ATTN", "flash_attn_version": 3},
        kernel_config={"moe_backend": "triton"},
        enable_chunked_prefill=True,
        language_model_only=True,
        generation_config="vllm",
        disable_log_stats=True,
        additional_config={"gdn_prefill_backend": "flashinfer"},
        worker_extension_cls="check_worker.CheckWorker",
    )
    ids = json.loads((ROOT.parent / "qwen36_64k_20260928/prompt_ids.json").read_text())
    assert len(ids) == 65536
    result = {
        "settings": settings,
        "scope": "Numerical/limited task screening; no broad quality equivalence claim",
        "prompt_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
        "phases": {},
        "needles": [],
    }

    def save():
        output.write_text(json.dumps(result, indent=2))

    llm = LLM(**settings)

    def generate(prompt, count, ignore_eos=True):
        assert len(prompt) == 65536
        request = llm.generate(
            [{"prompt_token_ids": prompt}],
            SamplingParams(
                temperature=0,
                seed=42,
                max_tokens=count,
                ignore_eos=ignore_eos,
                logprobs=5,
            ),
            use_tqdm=False,
        )[0]
        assert list(request.prompt_token_ids) == prompt
        o = request.outputs[0]
        if ignore_eos:
            assert len(o.token_ids) == count
        return {
            "tokens": list(o.token_ids),
            "text": o.text,
            "finish_reason": o.finish_reason,
            "logprobs": [{str(k): v.logprob for k, v in s.items()} for s in o.logprobs],
        }

    try:
        for phase, backend, count in [
            ("core_audit", "flashinfer", 1),
            ("capture", "flashinfer", 128),
            ("repeat", "flashinfer", 128),
            ("teacher", "triton", 128),
            ("free", "triton", 128),
        ]:
            identity = llm.collective_rpc("configure", args=(backend, phase))
            assert all(len(r["layers"]) == 30 for r in identity)
            generated = generate(ids, count)
            ranks = llm.collective_rpc("results")
            result["phases"][phase] = {
                "backend": backend,
                "identity": identity,
                "output": generated,
                "ranks": ranks,
            }
            save()
            print("PHASE", phase, "complete", flush=True)
        base = result["phases"]["capture"]["output"]
        result["free_tokens_equal"] = (
            base["tokens"] == result["phases"]["free"]["output"]["tokens"]
        )
        result["free_logprobs_equal"] = (
            base["logprobs"] == result["phases"]["free"]["output"]["logprobs"]
        )
        assert all(
            result["phases"][p]["output"]["tokens"] == base["tokens"]
            for p in ["repeat", "teacher"]
        )
        save()

        # Known-answer retrieval at three depths, two seeds. Each invocation
        # is one distinct 65536-token request, never a multi-request batch.
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("/data/model")
        enc = lambda s: tokenizer.encode(s, add_special_tokens=False)
        filler = enc(
            "The archive records routine work at the library. Staff review books, organize shelves, and prepare the reading room.\n"
        )
        for seed in (17, 29):
            for depth in (0.1, 0.5, 0.9):
                rng = random.Random(seed * 100 + int(depth * 10))
                key, answer = (
                    f"PROJECT-{rng.randrange(1000, 9999)}",
                    str(rng.randrange(100000, 999999)),
                )
                prefix = enc(
                    "<|im_start|>user\nRead the archive. Find the requested project's access code.\n<archive>\n"
                )
                needle = enc(f"\nThe unique access code for {key} is {answer}.\n")
                suffix = enc(
                    f"\n</archive>\nWhat is the access code for {key}? Reply with only the six digits, without any explanation.\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
                )
                count = 65536 - len(prefix) - len(needle) - len(suffix)
                haystack = (filler * (count // len(filler) + 1))[:count]
                at = int(count * depth)
                prompt = prefix + haystack[:at] + needle + haystack[at:] + suffix
                row = {
                    "seed": seed,
                    "depth": depth,
                    "expected": answer,
                    "prompt_sha256": hashlib.sha256(
                        json.dumps(prompt).encode()
                    ).hexdigest(),
                }
                result["needles"].append(row)
                order = (
                    ("flashinfer", "triton") if seed == 17 else ("triton", "flashinfer")
                )
                for backend in order:
                    llm.collective_rpc("configure", args=(backend, "retrieval"))
                    value = generate(prompt, 32, False)
                    row[backend] = {
                        "output": value,
                        "exact_answer": value["text"].strip() == answer,
                    }
                    save()
                print(
                    "NEEDLE",
                    seed,
                    depth,
                    row["flashinfer"]["exact_answer"],
                    row["triton"]["exact_answer"],
                    flush=True,
                )
        result["limited_retrieval_pass"] = all(
            r[b]["exact_answer"]
            for r in result["needles"]
            for b in ["flashinfer", "triton"]
        )
        result["completed"] = True
        save()
    finally:
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
