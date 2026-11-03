# PR Comment Commands

Slash commands typed as a PR comment (the comment must be **exactly** the
command, with no extra text) so that PR authors — who often have no write
access to this repository — can manage their own PR's CI from the timeline.

| Command | Who can use | What it does |
|---|---|---|
| `/rerun-failed-ci` | PR author, or collaborators with `write`+ | Re-runs the failed jobs of the latest failed workflow runs for the PR's head commit (e.g. the main CI run and CodeQL, up to 5) |
| `/cancel-ci` | PR author, or collaborators with `write`+ | Cancels all in-progress / queued workflow runs for the PR's head commit — the stop-loss for a hung e2e on a vendor runner |

The bot reacts to the command comment when authorized and posts a result
comment afterwards; failures include a link to the command's workflow log.

## Safety model (applies to every command workflow)

- Commands never check out or execute PR code. They only orchestrate
  existing workflow runs through the Actions API, so the classic
  `pull_request_target` code-execution risk does not apply.
- Only the PR author and collaborators with `write`/`maintain`/`admin`
  are honored. Comments from anyone else get an explicit rejection.
- Runs awaiting admin approval (`action_required`) are never touched.
- There are intentionally **no** commands that merge, approve, or change
  pull-request state — merging stays with the native GitHub flow
  (checks + review + the merge button / auto-merge).

## Adding a new command

One workflow per command under `.github/workflows/`, following the
structure of `rerun-failed-ci.yml`: exact-match `if:` guard, the shared
authorization step, an ack reaction, the action itself, and a result
comment. Document it in the table above in the same PR.
