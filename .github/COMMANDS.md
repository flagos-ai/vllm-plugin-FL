# PR Comment Commands

Slash commands typed as the first line of a PR comment let contributors and
maintainers manage CI from the PR timeline.

| Command | Who can use | What it does |
|---|---|---|
| `/rerun-failed-ci` | PR author, or collaborators with `write`+ | Re-runs failed and cancelled jobs from the latest failed/cancelled workflow runs for the PR's head commit (e.g. the main CI run and CodeQL, up to 5 runs) |
| `/cancel-ci` | PR author, or collaborators with `write`+ | Cancels all in-progress / queued workflow runs for the PR's head commit — the stop-loss for a hung e2e on a vendor runner |
| `/ci run [platforms\|all]` | Collaborators with `write`+ | Runs selected platforms against the PR's current immutable head SHA |
| `/test [platforms\|all]` | Collaborators with `write`+ | Alias for `/ci run` |

The bot reacts to the command comment when authorized and posts a result
comment afterwards; failures include a link to the command's workflow log.

## Safety model (applies to every command workflow)

- Control workflows use scripts from the trusted default branch. `/ci run`
  dispatches the trusted CI workflow, which then checks out the PR's immutable
  head SHA only inside the normal test jobs.
- Retry/cancel commands allow the PR author or collaborators with write access.
  Commands that start new hardware work require `write`/`maintain`/`admin`.
- Runs awaiting admin approval (`action_required`) are never touched.
- There are intentionally **no** commands that merge, approve, or change
  pull-request state — merging stays with the native GitHub flow
  (checks + review + the merge button / auto-merge).

## Adding a new command

One workflow per command under `.github/workflows/`, following the
structure of `rerun-failed-ci.yml`: exact-match `if:` guard, the shared
authorization step, an ack reaction, the action itself, and a result
comment. Document it in the table above in the same PR.
