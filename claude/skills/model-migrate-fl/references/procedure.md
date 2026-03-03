# Migration Procedure

## Goal

Enable benchmarking **{{MODEL_DISPLAY_NAME}}** with:

```bash
vllm bench throughput --model /models/{{MODEL_DISPLAY_NAME}} --dataset-name random --input-len 128 --output-len 128 --num-prompts 2 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --load-format dummy --max-num-seqs 10 --trust-remote-code
```

## Constraints

* vLLM version: **0.13.0**
* Modify **only** `vllm-plugin-FL` (installed with `-e`)
* Do NOT modify:
  * `/usr/local/lib`
  * `/models/`
* Do NOT import code from **vLLM > 0.13.0**
* Do NOT add any **extra environment variables** during final verification
* Reuse existing plugin patterns (e.g. `qwen3_5`, `kimi_k25`, `qwen3_next`, `minicpmo`)

---

## Project Structure

```
vllm-plugin-FL/
  vllm_fl/
    __init__.py            # register() and register_model() entry points
    configs/               # HuggingFace config bridges
      qwen3_5_moe.py       # Example: Qwen3.5 MoE config
    models/                # Model implementations
      qwen3_5.py            # Example: Complex MoE + hybrid attention
      kimi_k25.py           # Example: Wrapper around DeepseekV2
      qwen3_next.py         # Example: Hybrid attention model
      minicpmo.py           # Example: Multimodal model
      fla_ops.py            # Shared linear attention operations
    ops/                   # Custom operators
    dispatch/              # Dispatch backends (flaggems, cuda, etc.)
  setup.py                # Entry points: vllm.platform_plugins, vllm.general_plugins
```

---

## Step 1: Baseline Unit Tests (before any code changes)

> **→ Tell user**: `🔍 Step 1: Running baseline unit tests before making any changes...`

Run all unit tests **before modifying any plugin code** to establish a baseline:

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

Record the results:
- Total passed / failed / skipped / errors
- List of any **pre-existing failures** (test name + error summary)

**Handling results:**
- **All pass** → clean baseline, any failure after migration is a regression we introduced
- **Some fail** → report pre-existing failures to the user. These are NOT caused by our migration. Note the exact test names so we can distinguish them from regressions later

> **→ Tell user**: Report baseline results. Example:
> ```
> ✅ Step 1 complete: Baseline unit tests recorded
>   - 42 passed, 0 failed, 3 skipped
>   - Baseline is clean — any post-migration failure is a regression
> ```
> Or:
> ```
> ⚠️ Step 1 complete: Baseline unit tests recorded (pre-existing failures found)
>   - 40 passed, 2 failed, 3 skipped
>   - Pre-existing failures:
>     - test_xxx.py::test_foo — ImportError: ...
>     - test_yyy.py::test_bar — AssertionError: ...
>   - These are NOT caused by our migration and will be excluded from regression checks
> ```

---

## Step 2: Prepare Upstream Reference Source

> **→ Tell user**: `🔍 Step 1: Cloning/updating upstream vLLM repository for reference...`

Clone or update the latest vLLM source for reference:

```bash
test -d {{upstream_folder}} && cd {{upstream_folder}} && git pull || git clone --depth 1 https://github.com/vllm-project/vllm.git {{upstream_folder}}
```

From `{{upstream_folder}}`, identify the **{{model_name}}** model files:

- Search `vllm/model_executor/models/` for files matching `{{model_name_lower}}`
- Check `vllm/transformers_utils/config.py` for config registration
- Note the class names, import paths, and `model_type`

> **→ Tell user**: Report what files were found, list the class names and model_type discovered. Example:
> ```
> ✅ Step 1 complete: Upstream reference ready
> 📋 Found upstream model files:
>   - vllm/model_executor/models/{{model_name_lower}}.py (N lines)
>   - Classes: {{ModelClassName}}ForCausalLM, {{ModelClassName}}Model, ...
>   - model_type: "{{model_type}}"
> ```

---

## Step 3: Learn How Models Are Added in `vllm-plugin-FL`

> **→ Tell user**: `🔍 Step 3: Studying existing plugin patterns to choose the best migration strategy...`

Study these existing implementations to choose the best migration strategy:

| Model | Pattern | Key characteristic |
|---|---|---|
| `qwen3_5` | Complex MoE + hybrid attention | Full model file adaptation |
| `kimi_k25` | Wrapper around DeepseekV2 | Lightweight delegation |
| `qwen3_next` | Hybrid attention model | Custom attention integration |
| `minicpmo` | Multimodal model | Vision + language composition |

Learn from them: config bridge pattern in `vllm_fl/configs/`, model registration in `vllm_fl/__init__.py`, how upstream code is adapted for 0.13.0 compatibility.

> **→ Tell user**: Report which existing model is most similar to the target model and which pattern you plan to follow. Example:
> ```
> ✅ Step 2 complete: Pattern analysis done
> 📋 The {{MODEL_DISPLAY_NAME}} architecture is most similar to [existing_model]
> ✅ Decision: Will follow the [pattern_name] pattern because [reason]
> ```

---

## Step 4: Add {{MODEL_DISPLAY_NAME}} Model to `vllm-plugin-FL`

> **→ Tell user**: `🔍 Step 4: Creating model files for {{MODEL_DISPLAY_NAME}}...`

### 3.1 Create Config file (if needed)

If the model uses a `model_type` (= `{{model_type}}`) that is **not known** to vLLM 0.13.0's transformers, create a config bridge:

- File: `{{plugin_folder}}/vllm_fl/configs/{{model_name_lower}}.py`
- Subclass `transformers.PretrainedConfig`
- Set `model_type = "{{model_type}}"`
- Copy constructor parameters from the upstream config class

> **→ Tell user**: State whether a config bridge is needed and why. Example:
> ```
> 📋 model_type "{{model_type}}" is NOT in vLLM 0.13.0 → config bridge needed
> ✅ Created: vllm_fl/configs/{{model_name_lower}}.py (config bridge with N parameters)
> ```
> Or: `📋 model_type "{{model_type}}" already exists in 0.13.0 → no config bridge needed`

### 3.2 Create Model file under `vllm_fl/models/` — Copy-then-Patch

**IMPORTANT: Use copy-then-patch, NOT read-and-rewrite.** This is the key to fast migration.

**Step A**: Copy the upstream file directly:
```bash
cp {{upstream_folder}}/vllm/model_executor/models/{{model_name_lower}}.py {{plugin_folder}}/vllm_fl/models/{{model_name_lower}}.py
```

**Step B**: Apply relevant patches from [compatibility-patches.md](compatibility-patches.md) using the Edit tool. Each patch is a targeted search-and-replace, not a full file rewrite.

**Step C**: Verify with a quick import test:
```bash
python3 -c "from vllm_fl.models.{{model_name_lower}} import {{ModelClassName}}; print('OK')"
```
If it fails, fix only the specific error and retry. Do NOT re-read the entire file.

> **→ Tell user**: Output a single summary after all patches are applied (not per-fix):
> ```
> ✅ Created: vllm_fl/models/{{model_name_lower}}.py (~N lines)
>   Copied from upstream, then applied patches: P1-P5
>   Import test: ✅ passed
> ```

---

## Step 5: Register the Model in Plugin Entry Point

> **→ Tell user**: `🔍 Step 5: Registering {{MODEL_DISPLAY_NAME}} in plugin entry point...`

In `{{plugin_folder}}/vllm_fl/__init__.py`, add to `register_model()`:

1. (If config bridge was created) Register config:
```python
from vllm.transformers_utils.config import _CONFIG_REGISTRY
from vllm_fl.configs.{{model_name_lower}} import {{ConfigClassName}}
_CONFIG_REGISTRY["{{model_type}}"] = {{ConfigClassName}}
```

2. Register model class:
```python
ModelRegistry.register_model(
    "{{ModelClassName}}",
    "vllm_fl.models.{{model_name_lower}}:{{ModelClassName}}"
)
```

Both wrapped in try/except to avoid breaking other models if one fails.

> **→ Tell user**:
> ```
> ✅ Step 5 complete: Registration added to vllm_fl/__init__.py
>   - Config: _CONFIG_REGISTRY["{{model_type}}"] = {{ConfigClassName}}
>   - Model:  ModelRegistry.register_model("{{ModelClassName}}", ...)
> ```

---

## Step 6: Regression Unit Tests (after code changes)

> **→ Tell user**: `🔍 Step 6: Running unit tests to check for regressions...`

Run all unit tests again — same command as Step 1:

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

**Compare with baseline from Step 1:**
- **New failures** (test passed in baseline but fails now) → **regression introduced by our migration → MUST FIX before proceeding**. Diagnose the failure, fix the code, and re-run until no new failures remain.
- **Same failures as baseline** → pre-existing issue, not caused by us → note and continue.
- **All pass** (same as baseline) → migration is clean.

> **→ Tell user** (clean):
> ```
> ✅ Step 6 complete: No regressions detected
>   - 42 passed, 0 failed, 3 skipped (same as baseline)
> ```
> **→ Tell user** (regression found):
> ```
> ❌ Step 6: Regression detected!
>   - New failure: test_xxx.py::test_foo — [error summary]
>   - This was NOT failing in baseline → caused by our migration
> 🔧 Fix: [what you're going to do]
> ```
> Then fix and re-run. Repeat until no new failures.

---

## Step 7: Functional Tests (after code changes)

> **→ Tell user**: `🔍 Step 7: Running functional tests...`

Run all functional tests:

```bash
pytest {{plugin_folder}}/tests/functional_tests/ -v -s --tb=short
```

**Handling results:**
- **Tests auto-skip** (model weights not at `/models/` or GPU unavailable) → this is expected in some environments. **Warn and continue**, include skipped tests in the final report.
- **Tests pass** → existing model inference/serving is not broken by our changes.
- **Tests fail** → **warn and continue** (do not block). Include failures in the final report so the user can investigate. These may be environment-specific or pre-existing.

> **→ Tell user** (all pass/skip):
> ```
> ✅ Step 7 complete: Functional tests done
>   - 5 passed, 0 failed, 3 skipped (models not available)
> ```
> **→ Tell user** (some fail):
> ```
> ⚠️ Step 7 complete: Functional tests done (some failures)
>   - 3 passed, 1 failed, 4 skipped
>   - Failed: test_offline_minicpm.py::test_basic_generation — [error summary]
>   - (This may be pre-existing or environment-specific — included in final report)
> ```

---

## Step 8: Verify — Benchmark {{MODEL_DISPLAY_NAME}}

> **→ Tell user**: `🔍 Step 8: Running benchmark verification for {{MODEL_DISPLAY_NAME}}...`

Run **exactly**:

```bash
vllm bench throughput --model /models/{{MODEL_DISPLAY_NAME}} --dataset-name random --input-len 128 --output-len 128 --num-prompts 2 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --load-format dummy --max-num-seqs 10 --trust-remote-code
```

> **→ Tell user** (on success):
> ```
> ✅ Step 8 complete: Benchmark passed!
>   - Throughput: X.XX requests/s
>   - [any other notable metrics]
> ```
> **→ Tell user** (on failure):
> ```
> ❌ Step 8 failed: Benchmark error
>   - Error: [key error message]
>   - Root cause: [your analysis]
> 🔧 Fix: [what you're going to do]
> ```
> Then fix the issue and re-run. Report each fix-and-retry cycle to the user.

---

## Step 9: Verify — Serve {{MODEL_DISPLAY_NAME}}, then request the server

> **→ Tell user**: `🔍 Step 9: Starting serve + request verification for {{MODEL_DISPLAY_NAME}}...`

### 6.1: Serve

> **→ Tell user**: `🚀 Starting vLLM server on port 8121...`

Run **exactly**:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 vllm serve /models/{{MODEL_DISPLAY_NAME}} --host 0.0.0.0 --port 8121 --tensor-parallel-size 8 --gpu-memory-utilization 0.9 --max-num-seqs 10 --load-format fastsafetensors --trust-remote-code
```

> **→ Tell user** (on success): `✅ Server started successfully, ready for requests`
> **→ Tell user** (on failure): Report the error, analyze root cause, and fix before retrying.

### 6.2: Request

> **→ Tell user**: `📡 Sending test request to server...`

Run **exactly**:

```bash
curl http://localhost:8121/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"/models/{{MODEL_DISPLAY_NAME}}","messages":[{"role":"user","content":"介绍一下 vLLM 的核心优势"}],"max_tokens":10000}'
```

> **→ Tell user** (on success):
> ```
> ✅ Step 9 complete: Serve + Request verification passed!
>   - Server started and responded correctly
>   - Model generated coherent output
> ```
> **→ Tell user** (on failure): Report error details, analyze, fix, and retry. Keep user informed of each attempt.

---

## Expected Behavior

* Plugin provides the HF → vLLM config bridge
* vLLM core remains untouched
* No additional environment variables are introduced
* No filesystem writes outside plugin
* Unit tests: no regressions (no new failures compared to baseline)
* Functional tests: pass or skip (no new failures)
* Benchmark completes successfully
* Serve completes successfully
* Request completes successfully

---

## Final Report (MANDATORY)

After ALL steps complete, output a comprehensive migration summary to the user:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Migration Complete: {{MODEL_DISPLAY_NAME}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files created/modified:
  - {{plugin_folder}}/vllm_fl/configs/xxx.py     (config bridge — N lines)
  - {{plugin_folder}}/vllm_fl/models/xxx.py      (model impl — N lines)
  - {{plugin_folder}}/vllm_fl/__init__.py         (registration added)

Compatibility fixes applied:
  1. [brief description of each fix]

Test results:
  - Unit tests (baseline):    N passed, M failed, K skipped
  - Unit tests (regression):  N passed, M failed, K skipped — [regressions: none / list]
  - Functional tests:         N passed, M failed, K skipped — [notes on skips/failures]

Verification results:
  - Benchmark: ✅ passed / ❌ failed (details)
  - Serve:     ✅ passed / ❌ failed (details)
  - Request:   ✅ passed / ❌ failed (details)

Pre-existing issues (not caused by migration):
  - [any pre-existing test failures, or "None"]

Known issues / TODOs:
  - [any remaining items, or "None"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## Summary

* **No vLLM core changes**
* **Full permissions assumed only for project directories**
* **No runtime permission hacks**
* **No extra environment variables**
* **Minimal, stable plugin-only solution**
* **Fully compatible with vLLM 0.13.0**
* **Idempotent — safe to re-run when upstream updates**
