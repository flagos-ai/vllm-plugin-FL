---
name: model-migrate-fl
description: >
  Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project
  (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new
  model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released
  model. Trigger when the user says things like "migrate X model", "add X model support",
  "port X from upstream vLLM", "make X work with the FL plugin", or simply
  "/model-migrate-fl model_name". The model_name argument uses snake_case
  (e.g. qwen3_5, kimi_k25, deepseek_v4).
  Do NOT use for models already supported by vLLM 0.13.0 core, or for
  multimodal-only components that don't need backporting.
argument-hint: model_name [upstream_folder] [plugin_folder]
user-invokable: true
compatibility: "Requires vLLM 0.13.0, Python 3.8+, GPU with CUDA, vllm-plugin-FL installed via pip install -e"
metadata:
  version: "1.2"
  author: flagos-ai
  category: workflow-automation
  tags: [model-migration, vllm, backport]
allowed-tools: "Bash(pytest:*) Bash(python3:*) Bash(git:*) Bash(vllm:*) Bash(cp:*) Bash(curl:*) Bash(bash:*) Bash(test:*) Bash(ls:*) Bash(pip:*) Bash(cd:*) Read Edit Write Glob Grep AskUserQuestion TaskCreate TaskUpdate TaskList TaskGet"
---

# FL Plugin — Model Migration Skill

## Usage

```
/model-migrate-fl <model_name> [upstream_folder] [plugin_folder]
```

| Argument | Required | Default |
|---|---|---|
| `model_name` | Yes | — |
| `upstream_folder` | No | `/tmp/vllm-upstream-ref` |
| `plugin_folder` | No | current working directory |

## Execution

### Step 1: Parse arguments and validate paths

Extract from user input:
- `{{model_name}}` = first argument (required, snake_case)
- `{{upstream_folder}}` = second argument or `/tmp/vllm-upstream-ref`
- `{{plugin_folder}}` = third argument or current working directory

If `{{upstream_folder}}` doesn't exist, ask user whether to clone it. If `{{plugin_folder}}` doesn't exist, error out.

**→ Tell user**: Confirm parsed model name and paths.

### Step 2: Load references and resolve placeholders

Read these files (relative to this SKILL.md):
- `references/procedure.md` — step-by-step migration procedure
- `references/compatibility-patches.md` — 0.13.0 patch catalog
- `references/operational-rules.md` — communication, TaskList, bash rules, resilience

The procedure references executable scripts in `scripts/`:
- `scripts/validate_migration.py` — automated code review (Step 6)
- `scripts/benchmark.sh` — benchmark verification (Step 9)
- `scripts/serve.sh` — serve model (Step 10.1)
- `scripts/request.sh` — test request (Step 10.2)

Then investigate upstream source + HuggingFace to resolve all placeholders:

| Placeholder | How to derive |
|---|---|
| `{{model_name}}` | Direct from argument |
| `{{model_name_lower}}` | Lowercase of model_name (usually identical, e.g. `qwen3_5`) — used in file paths |
| `{{MODEL_DISPLAY_NAME}}` | From upstream code or HF model card |
| `{{ModelClassName}}` | From upstream model class (PascalCase) |
| `{{model_type}}` | From HF config.json `model_type` field |
| `{{ConfigClassName}}` | From upstream or derive from model_type |
| `{{skill_root}}` | Absolute path to this skill's folder (the directory containing this SKILL.md) |

Naming conventions vary per model — always verify from actual source, never guess.

**→ Tell user**: Present all resolved values. Use AskUserQuestion if anything is ambiguous.

### Step 3: Execute procedure

With placeholders resolved, execute every step in `procedure.md` sequentially. Apply patches from `compatibility-patches.md` during the copy-then-patch step. Follow `operational-rules.md` throughout.

**→ Tell user**: Before starting, output a numbered plan. Report progress at each step boundary.

## Examples

**Example 1: Typical new model**
```
User says: "/model-migrate-fl kimi_k25"
Actions:
  1. Parse → model_name=kimi_k25, defaults for upstream/plugin paths
  2. Clone upstream, find vllm/model_executor/models/kimi_k25.py
  3. Discover it wraps DeepseekV2 → follow kimi_k25 (wrapper) pattern
  4. Copy file, apply P1+P2 patches, create config bridge
  5. Register, validate, test, benchmark, serve+request
Result: kimi_k25 fully working in plugin, all 10 steps passed
```

**Example 2: Re-run after upstream update**
```
User says: "migrate qwen3_5 again, upstream updated"
Actions:
  1. Idempotent re-run — overwrite existing files with fresh upstream copy
  2. Re-apply patches, re-validate, re-test
Result: qwen3_5 updated to match latest upstream, no regressions
```

## Troubleshooting

| Problem | Typical Cause | Fix |
|---|---|---|
| `ImportError` after copy-then-patch | Missing P1 fix (relative→absolute imports) | Verify all `from .xxx` converted to `from vllm.*` or `from vllm_fl.*` |
| `AttributeError: module 'vllm' has no attribute X` | API doesn't exist in 0.13.0 | Check P3 in compatibility-patches.md; stub or remove |
| Config not recognized by vLLM | model_type mismatch or config bridge missing | Verify `_CONFIG_REGISTRY[model_type]` matches HF config.json exactly |
| Registration has no effect | Class name or import path typo | Compare with existing registrations in `__init__.py` |
| Benchmark `KeyError` on config field | Config bridge missing a field | Compare upstream config class vs bridge; add missing fields with defaults |
