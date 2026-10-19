# Forward sync from 0.4.0-dev to 0.5.0-dev

The source branch retains the vLLM 0.28 adaptation. The target branch starts at
the same commit and hosts the vLLM 0.32 adaptation. Changes flow in one direction:
`0.4.0-dev` → `codex/sync-0.4-to-0.5-<source-sha>` → a PR targeting `0.5.0-dev`.

## Current phase: PR creation only

`.github/workflows/forward-sync.yml` reconciles every push to `0.4.0-dev`,
including merged PRs and direct pushes. Multiple source changes accumulate in
one open PR. It does **not** merge PRs or enable auto-merge. NVIDIA CI for vLLM
0.32 and its required checks will be configured separately.

The controller reads the latest source and target refs on every run. A fixed
concurrency group serializes writes; rerunning an old event still picks up the
latest source commits. If the target already contains the source, no PR is needed.

Each new PR uses a branch suffixed with its initial source SHA. The branch starts
at that source commit, so a PR can be opened even when the
target contains conflicting 0.32 adaptations. Subsequent updates fast-forward or
merge source commits into that branch, preserving human conflict resolutions.
The controller never force-pushes or deletes branches, and never modifies either
development branch directly.

It stops and reports an actionable error for conflicting updates, a manually
closed unmerged PR, an unrelated branch with the same name, rewritten source
history, or a previous sync merged without preserving ancestry. Existing commits
and PRs remain available for review.

## Deploy

1. Create `0.5.0-dev` from the desired `0.4.0-dev` commit. Initial deployment
   uses `5809ea37735de890891d6a8bab91d27c112c745b` (2026-10-19).
2. Merge the workflow, script, tests and this guide into **0.4.0-dev**.
   The resulting push triggers the first run and the first sync PR.
3. In repository Settings → Actions → General → Workflow permissions, allow
   GitHub Actions to create pull requests. The workflow explicitly requests
   `contents: write` and `pull-requests: write`, but these permissions do not
   override the repository or organization policy. No custom secret is needed.
4. Check the workflow result and its PR link. A 403 when creating the PR requires
   a maintainer to check the Actions policy; do not replace the token with an
   unrelated personal credential.
5. Keep automatic merging disabled. Review and resolve PRs manually while the
   target branch's NVIDIA CI is being prepared.

On 2026-10-19, the current credentials could create repository branches but the
Actions workflow-permissions API returned 403. The PR-creation policy therefore
still needs confirmation by an administrator or by a successful workflow run.

## Merging sync PRs

Use **Create a merge commit** when a sync PR is ready to merge. Squash and rebase
lose the source ancestry needed for subsequent comparisons and can repeat old
changes. Ordinary feature PRs targeting `0.4.0-dev` may still use squash.

At initial deployment the repository allows only squash merges. An administrator
must enable merge commits before these cross-branch sync PRs are merged. A
target-branch linear-history requirement would also conflict with this strategy.
Enabling merge commits is separate from enabling auto-merge; automatic merging
remains disabled in this phase.

If a sync PR was accidentally squash-merged, the next cycle stops for manual
history reconciliation. Do not repeatedly squash the same cumulative PR or
reset the sync branch to hide the problem.

## Conflicts and recovery

Resolve conflicts on `codex/sync-0.4-to-0.5-<source-sha>`, not on `0.4.0-dev`. In a clean
checkout with the official repository configured as `upstream`, replace the
placeholder below with the exact head branch shown on the sync PR:

```bash
sync_branch='codex/sync-0.4-to-0.5-<source-sha>'
git fetch upstream
git switch -c forward-sync-review --track "upstream/$sync_branch"
git merge upstream/0.4.0-dev
# Resolve any conflicts, stage the resolution, and complete the merge commit.
git merge upstream/0.5.0-dev
# Resolve target-version conflicts and complete this merge commit as needed.
git push upstream "HEAD:$sync_branch"
```

Inspect the resulting diff against `0.5.0-dev` and run its validation before
merging the PR. Preserve the managed HTML comments in the PR description.
Reopen a manually closed PR to resume it; closing a PR intentionally pauses
automatic updates rather than silently discarding that decision.

After fixing a workflow failure, use **Re-run failed jobs** on its Actions run.
The controller reads current refs, so this also catches up newer source pushes.
If PR creation failed after a branch was created, retry is safe when that branch
still points at its named source commit. A newer source starts a new branch;
orphaned branches are preserved. Unexpected manual branch changes are preserved
and require inspection.

The workflow includes `workflow_dispatch`, but GitHub only enables dispatch
after the workflow also exists on the default branch (`main`). This deployment
targets `0.4.0-dev`; it relies on push events and reruns. It does not yet include
a scheduled reconciliation job or promise an available “Run workflow” button.

## CI and the later auto-merge phase

The initial token is `GITHUB_TOKEN`. Under current GitHub behavior, bot-created
or updated PRs can produce `pull_request` workflow runs that require a user with
write access to approve them. Other token-generated events, such as pushes, do
not recursively run workflows. The current existing platform CI branch filters
also exclude `0.5.0-dev`; this change does not make those checks available there.

Before enabling automatic merging:

- Configure NVIDIA jobs for the actual vLLM 0.32 environment, including images,
  setup version assertions and the target-branch filters.
- Establish stable required checks on the target merge result and satisfy the
  repository's approval rules; account for checks skipped by path filters.
- Use an appropriately scoped GitHub App if unattended PR CI is required.
- Enable merge commits, then explicitly enable and implement auto-merge.
- Exercise conflicts, CI failures, repeated events and another complete sync
  cycle. Do not bypass required checks or use an administrator merge shortcut.

## Local validation

```bash
node --test .github/scripts/tests/forward-sync.test.cjs
```

The API-mock tests exercise reconciliation and failure handling without tokens,
network writes or GPU dependencies. The workflow runs them before each sync and
on PRs that change the controller. They do not replace a real Actions permission
check or target-version GPU tests.

## References

- [GitHub workflow triggering and GITHUB_TOKEN behavior](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow)
- [Repository Actions permissions](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-github-actions-settings-for-a-repository)
- [Long-running branches and merge methods](https://docs.github.com/en/pull-requests/reference/pull-request-merges)
- [Workflow concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency)
