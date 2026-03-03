---
name: model-migrate-fl
description: >
  Migrate a model from the latest vLLM upstream repository into the vllm-plugin-FL project
  (pinned at vLLM v0.13.0). Use this skill whenever someone wants to add support for a new
  model to vllm-plugin-FL, port model code from upstream vLLM, or backport a newly released
  model. Trigger when the user says things like "migrate X model", "add X model support",
  "port X from upstream vLLM", "make X work with the FL plugin", or simply
  "/model-migrate-fl <model_name>". The model_name argument uses snake_case
  (e.g. qwen3_5, kimi_k25, deepseek_v4).
argument-hint: <model_name> [upstream_folder] [plugin_folder]
user-invokable: true
metadata:
  version: "1.1"
  author: flagos-ai
---

# FL Plugin — Model Migration Skill

## Purpose

This skill migrates a model from the latest vLLM upstream (`https://github.com/vllm-project/vllm`, main branch) into the **vllm-plugin-FL** project. The plugin is pinned to vLLM v0.13.0 and extends it with multi-hardware support without modifying vLLM core. New models are released against the latest vLLM, so they need to be carefully back-ported into the plugin's v0.13.0-compatible structure.

## Installation

Copy the entire skill directory into your Claude Code project:

```
your-project/
  .claude/
    skills/
      model-migrate-fl/
        SKILL.md                              # This file (skill entry point)
        references/
          procedure.md                        # Step-by-step migration procedure
          compatibility-patches.md            # vLLM 0.13.0 patch catalog
```

## Prerequisites

- **vLLM 0.13.0** installed in the target environment
- **vllm-plugin-FL** cloned and installed with `pip install -e .`
- Access to internet (to clone upstream vLLM for reference) or a local copy of the latest vLLM source
- Python 3.8+

## Bundled Files

| File | Purpose |
|---|---|
| `SKILL.md` | Skill definition, orchestration logic, and behavioral rules |
| `references/procedure.md` | Detailed step-by-step migration procedure with all commands |
| `references/compatibility-patches.md` | Catalog of known vLLM 0.13.0 incompatibilities and their fixes (P1-P5+) |

## Usage

The user provides a snake_case model name as argument, with optional path overrides:

```
/model-migrate-fl <model_name> [upstream_folder] [plugin_folder]
```

| Argument | Required | Default | Description |
|---|---|---|---|
| `model_name` | Yes | — | snake_case model identifier (e.g. `qwen3_5`, `kimi_k25`) |
| `upstream_folder` | No | `/tmp/vllm-upstream-ref` | Path to the upstream vLLM source for reference |
| `plugin_folder` | No | current working directory | Path to the vllm-plugin-FL project to migrate into |

**Examples:**
```
/model-migrate-fl qwen3_5
/model-migrate-fl kimi_k25 /path/to/vllm-upstream
/model-migrate-fl deepseek_v4 /path/to/vllm-upstream /path/to/vllm-plugin-FL
```

## Communication Protocol (MANDATORY)

Throughout the entire migration process, you MUST actively communicate with the user at every step. This is not optional — silent execution is unacceptable. Follow these rules:

### Status Updates

**Before each step**: Output a brief status line telling the user what you're about to do.
**After each step**: Output a summary of what was accomplished, what was found, or what was decided.
**On encountering issues**: Immediately inform the user about the problem, what you've tried, and your proposed solution — before attempting the fix.

### Format

Use the following patterns for status messages (output as regular text, not tool calls):

- **Starting a step**: `🔍 Step N: <what you're doing>...`
- **Discovery/finding**: `📋 Found: <concise description of what was found>`
- **Decision made**: `✅ Decision: <what was decided and why>`
- **Issue encountered**: `⚠️ Issue: <problem description>` followed by `🔧 Fix: <what you're going to do about it>`
- **Step completed**: `✅ Step N complete: <brief summary of outcome>`
- **Asking for user input**: Use the AskUserQuestion tool when you encounter ambiguity or need the user to make a choice (e.g., which HuggingFace model repo to use, which inheritance strategy to pick)

### What to Communicate

Keep it concise. Report these at each step boundary (NOT during the step):

1. **Model identity** — The placeholder values you determined (one-time, after investigation)
2. **File operations** — Which files you created/modified (after each step completes)
3. **Patch summary** — List of patches applied (batch summary, not per-fix)
4. **Verification results** — Whether commands succeeded or failed, and key output
5. **Issues** — Only report issues that block progress or need user input

## TaskList Integration — Checkpoint & Auto-Resume (CRITICAL)

The TaskList is both a progress indicator AND the primary recovery mechanism after API interruptions (e.g. ECONNRESET). Follow these rules strictly:

### On first invocation — create all tasks upfront

After parsing the model name (Step 1), immediately create ALL migration tasks at once using TaskCreate. Each task description MUST include the concrete model context so that a resumed session can continue without re-investigation. Example:

```
Task 1:  "Baseline unit tests"                  — run before any code changes, record pass/fail counts
Task 2:  "Clone/update upstream vLLM"            — description includes model_name
Task 3:  "Investigate model & resolve placeholders" — description includes model_name
Task 4:  "Study existing plugin patterns"
Task 5:  "Create config bridge"                  — description includes model_type, config class name
Task 6:  "Create model file (copy-then-patch)"   — description includes upstream file path, target path
Task 7:  "Register model in __init__.py"         — description includes class name, model_type
Task 8:  "Regression unit tests"                 — compare with baseline, fix any new failures
Task 9:  "Functional tests"                      — run all, warn on skip/fail, continue
Task 10: "Verify — benchmark"                    — description includes full benchmark command
Task 11: "Verify — serve + request"              — description includes full serve & curl commands
```

Once placeholder values are resolved (after Task 2), UPDATE tasks 4-8 descriptions to include the concrete values (not placeholders).

### On every conversation turn — auto-resume protocol

**ALWAYS** start every turn by calling `TaskList`. Then:

1. If there are `in_progress` tasks → **continue from the in_progress task immediately**. Do NOT ask the user "should I continue?" — just do it.
2. If all tasks are `pending` with none `in_progress` → this is a fresh start, begin from the first pending task.
3. If all tasks are `completed` → output the final migration report.
4. If the user says "continue", "继续", or similar → call TaskList and resume from the first non-completed task.

**NEVER ask the user whether to continue.** After an API interruption, the user's intent is always "continue the migration". Just read the task list, determine where you left off, and keep going.

### Task state discipline

- Mark a task `in_progress` BEFORE starting work on it.
- Mark a task `completed` ONLY after the step is fully done and verified.
- If a task fails, keep it as `in_progress` and fix the issue — do NOT mark it completed.
- A task's description is the single source of truth for what needs to be done. Write enough detail that a cold-start can execute it.

## Bash Command Rules (CRITICAL — avoids permission prompts)

To prevent Claude Code from pausing for interactive safety confirmations during migration, follow these rules for ALL Bash calls:

1. **Single-line commands only** — NEVER use backslash `\` continuation or multi-line commands in a single Bash call. Chain with `&&` on one line, or use separate Bash calls.
2. **No process substitution** — NEVER use `<()` or `>()`. Use pipes or temp files instead.
3. **No quoted flag values** — Write `--load-format dummy` NOT `--load-format "dummy"`.
4. **Use Edit tool for file modifications** — NEVER use `sed -i`. Use the Edit tool instead.
5. **Simple command prefixes** — Start each command with a recognizable name (`ls`, `cp`, `grep`, `python3`, `git`, `vllm`). Avoid starting with `then`, `else`, `{`, `(`.
6. **Complex scripts** — Write to a temp `.sh` or `.py` file, then execute it in a separate Bash call.

## Resilience — Auto-Retry & Auto-Resume (MANDATORY)

### Network command retry

For any Bash command that depends on network (git clone, git pull, curl, pip install, wget), if it fails with a network-related error (timeout, connection refused, ECONNRESET, DNS failure, etc.), **automatically retry up to 3 times** with a 5-second sleep between attempts. Use this pattern:

```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-upstream-ref || sleep 5 && git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-upstream-ref
```

Or for more robust retry, write a temp retry script:
```python
# /tmp/_retry_cmd.py — write this, then run: python3 /tmp/_retry_cmd.py
import subprocess, time, sys
for attempt in range(3):
    r = subprocess.run(sys.argv[1:])
    if r.returncode == 0: break
    print(f"Attempt {attempt+1} failed, retrying in 5s...")
    time.sleep(5)
else:
    sys.exit(1)
```

### Auto-resume after API interruption

If you detect (via TaskList) that there are tasks already created with some `completed` and some `pending`/`in_progress`, this means a previous session was interrupted. **Do NOT start over.** Instead:

1. Call `TaskList` to see current state
2. Find the first non-completed task
3. Read its description for the full context (model name, placeholder values, file paths)
4. Continue execution from that exact point
5. **NEVER re-do completed tasks** (they are idempotent but wasteful)
6. **NEVER ask the user "should I continue?"** — the answer is always yes

### Work-until-done principle

You MUST keep working until ALL tasks in the task list are marked `completed` (or the entire migration is verified). Do not stop after a single step and wait for the user. Each turn should make maximum progress — complete as many tasks as possible before yielding.

## Permission Assumptions (IMPORTANT)

To avoid execution interruptions:

* Full **read / write / execute** permissions are granted for the vllm-plugin-FL project directory
* Full **read** permissions are granted for `/usr/local/lib/` (vLLM 0.13.0 source), `/tmp/` (upstream clone), and `/models/` (HuggingFace checkpoints)
* No permission escalation (e.g. `sudo`, `chmod` during runtime) is required
* No sandbox or restricted FS behavior is assumed
* All fixes must remain **inside the plugin directory**

This guarantees Claude Code can:

* Inspect vLLM 0.13.0 source for reference (at `/usr/local/lib/python*/dist-packages/vllm/`)
* Inspect latest vLLM upstream source for reference (cloned to `{{upstream_folder}}`)
* Read model config files from `/models/` for reference
* Modify plugin files freely

## Execution Procedure

Follow these steps **exactly and in order**. The migration procedure at `references/procedure.md` and the patch catalog at `references/compatibility-patches.md` (both relative to this skill file) are the sources of truth for the migration logic.

### Step 0: Parse arguments and validate paths

Extract arguments from the user's input:

- `{{model_name}}` = first argument (required, snake_case identifier)
- `{{upstream_folder}}` = second argument if provided, otherwise `/tmp/vllm-upstream-ref`
- `{{plugin_folder}}` = third argument if provided, otherwise the current working directory

**Path validation:**
- For `{{upstream_folder}}`: if the path does not exist, use AskUserQuestion to ask whether to clone it via `git clone --depth 1 https://github.com/vllm-project/vllm.git {{upstream_folder}}`. Do not clone without confirmation.
- For `{{plugin_folder}}`: if the path does not exist, error out — the plugin project must already exist.

**→ Tell the user**: Confirm the model name and paths you parsed.

### Step 1: Read the migration procedure and patch catalog

Read both reference files relative to this SKILL.md file:

- `references/procedure.md` — the step-by-step migration procedure
- `references/compatibility-patches.md` — the patch catalog for 0.13.0 compatibility fixes

**→ Tell the user**: Briefly confirm you've loaded the migration procedure and are proceeding.

### Step 2: Determine placeholder values

Before you can execute the migration procedure, you need to determine the concrete values for all placeholders. This requires investigation:

1. **Clone the latest vLLM upstream** into `{{upstream_folder}}` (or update if it already exists) for reading
2. **Find the model files** in `vllm/model_executor/models/` matching the model name
3. **Check the HuggingFace config.json** for the model to find `model_type` and `architectures`
4. **Study the upstream class names** and import paths

The placeholders and how to derive them:

| Placeholder | Meaning | How to derive |
|---|---|---|
| `{{model_name}}` | snake_case identifier as given by user | Direct from argument (e.g. `qwen3_5`) |
| `{{model_name_lower}}` | Lowercase for file paths (often same as model_name) | Lowercase of model_name (e.g. `qwen3_5`) |
| `{{MODEL_DISPLAY_NAME}}` | Human-readable name for comments and docs | Read from upstream code or HF model card (e.g. `Qwen3.5`) |
| `{{ModelClassName}}` | PascalCase class name prefix used in code | From upstream model class (e.g. `Qwen3_5Moe`) |
| `{{model_type}}` | The `model_type` value in HF config.json | From model's config.json on HuggingFace (e.g. `qwen3_5_moe`) |
| `{{upstream_folder}}` | Path to upstream vLLM source | From argument or `/tmp/vllm-upstream-ref` |
| `{{plugin_folder}}` | Path to vllm-plugin-FL project | From argument or current working directory |

The naming conventions vary per model, so always verify from the actual upstream source rather than guessing. For example:
- `qwen3_5` -> class `Qwen3_5MoeForConditionalGeneration`, model_type `qwen3_5_moe`
- `kimi_k25` -> class `KimiK25ForConditionalGeneration`, model_type `kimi_k25`

**→ Tell the user**: Present all resolved placeholder values in a clear table/list format. If any value is uncertain or there are multiple candidates, use AskUserQuestion to let the user choose.

### Step 3: Replace placeholders and execute

Mentally replace all `{{...}}` placeholders in procedure.md with the concrete values, then **execute every step** in the procedure sequentially. For the copy-then-patch step, refer to `references/compatibility-patches.md` for the patch catalog.

**→ Tell the user**: Before starting execution, output a numbered plan of all steps you're about to perform. As you complete each sub-step, output a progress line.

### Step 4: Idempotency

The migration must be repeatable. If target files already exist, overwrite them with the latest upstream code (re-adapted for 0.13.0 compatibility). This handles the common case where upstream changes daily and the migration must be re-run to pick up fixes.

**→ Tell the user**: If overwriting existing files, explicitly state which files are being replaced and why.

## Project Structure Reference

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

## Registration Pattern

In `vllm_fl/__init__.py`, the `register_model()` function:

1. Registers custom configs via `_CONFIG_REGISTRY[model_type] = ConfigClass`
2. Registers model classes via `ModelRegistry.register_model(ClassName, import_path)`

Both wrapped in try/except to avoid breaking other models if one fails.

## Final Report (MANDATORY)

After completing all steps (including verification), output a **migration summary** to the user:

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📊 Migration Summary: {{MODEL_DISPLAY_NAME}}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Files created/modified:
  - vllm_fl/configs/xxx.py  (config bridge)
  - vllm_fl/models/xxx.py   (model implementation, ~N lines)
  - vllm_fl/__init__.py      (registration added)

Compatibility fixes applied:
  - <list each fix briefly>

Test results:
  - Unit tests (baseline):    N passed, M failed, K skipped
  - Unit tests (regression):  N passed, M failed, K skipped — regressions: none / list
  - Functional tests:         N passed, M failed, K skipped — notes on skips/failures

Verification results:
  - Benchmark: ✅ / ❌
  - Serve:     ✅ / ❌
  - Request:   ✅ / ❌

Pre-existing issues (not caused by migration):
  - <any pre-existing test failures, or "None">

Known issues / TODOs:
  - <any remaining items>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```
