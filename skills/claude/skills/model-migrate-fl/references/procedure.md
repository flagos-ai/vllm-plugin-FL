# Migration Procedure

## Goal

Enable benchmarking **{{MODEL_DISPLAY_NAME}}** — the target command is `bash scripts/benchmark.sh {{MODEL_DISPLAY_NAME}}` (see `scripts/benchmark.sh` for exact flags).

## Constraints

* vLLM version: **0.13.0**
* Modify **only** `vllm-plugin-FL` (installed with `-e`)
* Do NOT modify `/usr/local/lib` or `/models/`
* Do NOT import code from **vLLM > 0.13.0**
* Do NOT add any **extra environment variables** during final verification
* Reuse existing plugin patterns (e.g. `qwen3_5`, `kimi_k25`, `qwen3_next`, `minicpmo`)
* **Idempotent** — if target files already exist, overwrite with latest upstream (re-adapted). Safe to re-run.

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

## Step 1: Baseline Unit Tests

> **→ Tell user**: `🔍 Step 1: Running baseline unit tests before making any changes...`

Run all unit tests **before modifying any plugin code**:

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

Record: total passed / failed / skipped / errors, and any **pre-existing failure** names.

- **All pass** → clean baseline; any post-migration failure is a regression
- **Some fail** → note exact test names to exclude from regression comparison

> **→ Tell user**: Report baseline (count + pre-existing failures if any).

---

## Step 2: Prepare Upstream Reference Source

> **→ Tell user**: `🔍 Step 2: Cloning/updating upstream vLLM repository...`

```bash
test -d {{upstream_folder}} && cd {{upstream_folder}} && git pull || git clone --depth 1 https://github.com/vllm-project/vllm.git {{upstream_folder}}
```

From `{{upstream_folder}}`, find the model files:
- Search `vllm/model_executor/models/` for `{{model_name_lower}}`
- Check `vllm/transformers_utils/config.py` for config registration
- Note class names, import paths, and `model_type`

> **→ Tell user**: Report files found, class names, model_type.

---

## Step 3: Study Existing Plugin Patterns

> **→ Tell user**: `🔍 Step 3: Studying existing plugin patterns...`

| Model | Pattern | Key characteristic |
|---|---|---|
| `qwen3_5` | Complex MoE + hybrid attention | Full model file adaptation |
| `kimi_k25` | Wrapper around DeepseekV2 | Lightweight delegation |
| `qwen3_next` | Hybrid attention model | Custom attention integration |
| `minicpmo` | Multimodal model | Vision + language composition |

Learn: config bridge pattern, model registration in `__init__.py`, 0.13.0 adaptation techniques.

> **→ Tell user**: Which existing model is most similar, which pattern you'll follow, and why.

---

## Step 4: Add Model Files

> **→ Tell user**: `🔍 Step 4: Creating model files for {{MODEL_DISPLAY_NAME}}...`

### 4.1 Config Bridge (if needed)

If `{{model_type}}` is NOT known to vLLM 0.13.0's transformers, create:
- File: `{{plugin_folder}}/vllm_fl/configs/{{model_name_lower}}.py`
- Subclass `transformers.PretrainedConfig`, set `model_type = "{{model_type}}"`
- Copy constructor parameters from upstream config class

> **→ Tell user**: Whether config bridge is needed and why.

### 4.2 Model File — Copy-then-Patch

**IMPORTANT: Copy-then-patch, NOT read-and-rewrite.**

**A.** Copy upstream file:
```bash
cp {{upstream_folder}}/vllm/model_executor/models/{{model_name_lower}}.py {{plugin_folder}}/vllm_fl/models/{{model_name_lower}}.py
```

**B.** Apply patches from `compatibility-patches.md` using the Edit tool.

**C.** Verify import:
```bash
python3 -c "from vllm_fl.models.{{model_name_lower}} import {{ModelClassName}}; print('OK')"
```
If it fails, fix the specific error and retry. Do NOT rewrite the whole file.

> **→ Tell user**: Single summary after all patches (files created, patches applied, import test result).

---

## Step 5: Register Model

> **→ Tell user**: `🔍 Step 5: Registering {{MODEL_DISPLAY_NAME}} in plugin entry point...`

In `{{plugin_folder}}/vllm_fl/__init__.py`, add to `register_model()`:

1. (If config bridge created) Register config:
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

Both wrapped in try/except to avoid breaking other models.

> **→ Tell user**: Registration summary (config + model entries added).

---

## Step 6: Post-Migration Code Review

> **→ Tell user**: `🔍 Step 6: Reviewing migrated code for correctness...`

### 6.1 Automated checks

Run the validation script (relative to this skill's root):

```bash
python3 {{skill_root}}/scripts/validate_migration.py {{plugin_folder}} {{plugin_folder}}/vllm_fl/models/{{model_name_lower}}.py {{plugin_folder}}/vllm_fl/configs/{{model_name_lower}}.py
```

Omit the config file argument if no config bridge was created. The script checks:
- **Imports**: relative imports, missing `vllm_fl.*` modules
- **API compatibility**: known 0.13.0-missing APIs (`_mark_tower_model`, `MambaStateCopyFunc`, etc.)
- **Config consistency**: `model_type` defined, `PretrainedConfig` subclass, `__init__` present
- **Registration**: class names and import paths in `__init__.py` match actual code
- **Code cleanliness**: bare `except:`, hardcoded external paths

Exit code 0 = passed, 1 = issues found. **Fix all ISSUES before proceeding.** WARNINGS are informational.

### 6.2 Manual review (items the script cannot check)

- Inherited base class method signatures match 0.13.0 (compare with `/usr/local/lib/python*/dist-packages/vllm/`)
- Config bridge fields match upstream `config.json` defaults (no missing required fields)
- No commented-out upstream code left behind (remove or adapt)
- No references to vLLM > 0.13.0 features beyond the known API list

> **→ Tell user**: Report script output + manual review findings. Fix any issues before proceeding.

---

## Step 7: Regression Unit Tests

> **→ Tell user**: `🔍 Step 7: Running unit tests to check for regressions...`

```bash
pytest {{plugin_folder}}/tests/unit_tests/ -v --tb=short
```

Compare with Step 1 baseline:
- **New failures** → regression from our migration → **MUST FIX** before proceeding
- **Same failures as baseline** → pre-existing, not us → continue
- **All pass** → clean

> **→ Tell user**: Results + comparison with baseline. Fix and re-run on regressions.

---

## Step 8: Functional Tests

> **→ Tell user**: `🔍 Step 8: Running functional tests...`

```bash
pytest {{plugin_folder}}/tests/functional_tests/ -v -s --tb=short
```

- **Auto-skip** (missing weights / GPU) → expected, warn and continue
- **Pass** → existing inference not broken
- **Fail** → warn and continue (may be pre-existing or env-specific), include in final report

> **→ Tell user**: Results with notes on skips/failures.

---

## Step 9: Verify — Benchmark

> **→ Tell user**: `🔍 Step 9: Running benchmark...`

```bash
bash {{skill_root}}/scripts/benchmark.sh {{MODEL_DISPLAY_NAME}}
```

> **→ Tell user**: Pass/fail + throughput metrics. On failure: analyze, fix, and re-run.

---

## Step 10: Verify — Serve + Request

> **→ Tell user**: `🔍 Step 10: Starting serve + request verification...`

### 10.1 Serve
```bash
bash {{skill_root}}/scripts/serve.sh {{MODEL_DISPLAY_NAME}}
```

### 10.2 Request
```bash
bash {{skill_root}}/scripts/request.sh {{MODEL_DISPLAY_NAME}}
```

> **→ Tell user**: Pass/fail for both. On failure: analyze, fix, retry.

---

## Final Report

After ALL steps complete, output:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Migration Complete: {{MODEL_DISPLAY_NAME}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files created/modified:
  - vllm_fl/configs/xxx.py     (config bridge)
  - vllm_fl/models/xxx.py      (model impl — ~N lines)
  - vllm_fl/__init__.py         (registration added)

Compatibility fixes applied:
  1. [brief per-fix description]

Code review results:
  - [issues found and fixed, or "Clean — no issues"]

Test results:
  - Unit (baseline):    N passed, M failed, K skipped
  - Unit (regression):  N passed, M failed, K skipped — regressions: none/list
  - Functional:         N passed, M failed, K skipped — notes

Verification:
  - Benchmark: ✅ / ❌
  - Serve:     ✅ / ❌
  - Request:   ✅ / ❌

Pre-existing issues: [list or "None"]
Known issues / TODOs: [list or "None"]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
