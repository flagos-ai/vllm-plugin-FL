# Operational Rules

Rules that apply throughout the entire migration process.

## Communication Protocol

Actively communicate at every step boundary. Silent execution is unacceptable.

**Status line patterns:**
- Starting: `🔍 Step N: <what>...`
- Finding: `📋 Found: <what>`
- Decision: `✅ Decision: <what and why>`
- Issue: `⚠️ Issue: <problem>` → `🔧 Fix: <action>`
- Complete: `✅ Step N complete: <summary>`

**What to report** (concise, at step boundaries only):
1. Model identity — resolved placeholder values (once)
2. File operations — files created/modified
3. Patch summary — list of patches applied (batch)
4. Verification results — pass/fail with key output
5. Issues — only blocking problems or decisions needing user input

Use AskUserQuestion for ambiguity or choices (e.g. HuggingFace repo, inheritance strategy).

## TaskList Integration

The TaskList is both a progress indicator AND the recovery mechanism after API interruptions.

### On first invocation — create all tasks upfront

After parsing the model name, create ALL tasks at once. Each description MUST include concrete model context for cold-start recovery:

```
Task 1:  Baseline unit tests
Task 2:  Clone/update upstream vLLM
Task 3:  Investigate model & resolve placeholders
Task 4:  Study existing plugin patterns
Task 5:  Create config bridge
Task 6:  Create model file (copy-then-patch)
Task 7:  Register model in __init__.py
Task 8:  Post-migration code review
Task 9:  Regression unit tests
Task 10: Functional tests
Task 11: Benchmark verification
Task 12: Serve + request verification
```

Once placeholders are resolved, UPDATE task descriptions with concrete values.

### Auto-resume protocol

**ALWAYS** start every turn with `TaskList`. Then:
- `in_progress` tasks exist → continue immediately (do NOT ask user)
- All `pending`, none `in_progress` → fresh start from first pending
- All `completed` → output final report
- User says "continue"/"继续" → resume from first non-completed task

**NEVER ask whether to continue.** After an interruption, just read tasks and keep going.

### Task state discipline

- Mark `in_progress` BEFORE starting work
- Mark `completed` ONLY after fully done and verified
- Keep `in_progress` on failure; fix the issue first
- Task descriptions = single source of truth (enough detail for cold-start)

### Work-until-done principle

Keep working until ALL tasks are completed. Do not stop after one step and wait. Make maximum progress each turn.

## Bash Command Rules

Prevents permission prompts during migration:

1. **Single-line commands only** — no backslash `\` continuation. Chain with `&&` or use separate calls.
2. **No process substitution** — no `<()` or `>()`. Use pipes or temp files.
3. **No quoted flag values** — write `--load-format dummy` not `--load-format "dummy"`.
4. **Use Edit tool for file modifications** — never `sed -i`.
5. **Simple command prefixes** — start with `ls`, `cp`, `grep`, `python3`, `git`, `vllm`. Avoid `then`, `else`, `{`, `(`.
6. **Complex scripts** — write to temp `.sh`/`.py` file, then execute in separate call.

## Resilience

### Network retry

For network-dependent commands (git clone/pull, curl, pip, wget), on failure retry up to 2 more times with 5s sleep between attempts:

```bash
git clone --depth 1 https://github.com/vllm-project/vllm.git /tmp/vllm-upstream-ref
```

If the command fails, run `sleep 5` then retry the same command. Repeat once more if still failing (3 attempts total). If all 3 fail, report the error to the user.

### Auto-resume after interruption

If TaskList shows some tasks `completed` and some not → previous session was interrupted. Do NOT start over. Find first non-completed task, read its description, continue from there. NEVER re-do completed tasks.

## Permission Assumptions

- Full **read/write/execute** for vllm-plugin-FL project directory
- Full **read** for `/usr/local/lib/` (vLLM 0.13.0 source), `/tmp/` (upstream clone), `/models/` (HF checkpoints)
- No `sudo`/`chmod` needed
- All fixes stay **inside the plugin directory**
