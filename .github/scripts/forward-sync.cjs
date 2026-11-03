/*
 * Copyright 2026 FlagOS Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

"use strict";

const SOURCE = "0.4.0-dev";
const TARGET = "0.5.0-dev";
const SYNC_PREFIX = "codex/sync-0.4-to-0.5-";
const MARKER = "<!-- vllm-fl-forward-sync: 0.4.0-dev -> 0.5.0-dev -->";
const SOURCE_MARKER = /<!-- forward-sync-source: ([0-9a-f]{40}) -->/g;
const PENDING_MARKER = /\n?<!-- forward-sync-pending: [0-9a-f]{40} -->/g;
const CONFLICT_BLOCK = /\n?<!-- forward-sync-conflicts:start -->[\s\S]*?<!-- forward-sync-conflicts:end -->/g;

module.exports = async function forwardSync({ github, context, core, replay }) {
  const payload = context.payload || {};
  if (payload.ref === "refs/heads/" + SOURCE && payload.deleted) {
    core.setOutput("status", "source-deleted");
    return { status: "source-deleted", pullRequest: null, pullRequestUrl: null };
  }
  if (payload.ref === "refs/heads/" + SOURCE && payload.forced) {
    throw new Error("The source push rewrote history; reconcile the sync branch manually.");
  }
  const { owner, repo } = context.repo;
  const api = github.rest;
  const scope = { owner, repo };
  let syncBranch;

  async function ref(branch, optional = false) {
    try {
      return (await api.git.getRef({ ...scope, ref: "heads/" + branch })).data.object.sha;
    } catch (error) {
      if (optional && error.status === 404) return null;
      throw error;
    }
  }

  async function isIncluded(commit, destination) {
    if (commit === destination) return true;
    const { data } = await api.repos.compareCommitsWithBasehead({
      ...scope, basehead: destination + "..." + commit,
    });
    return ["behind", "identical"].includes(data.status);
  }

  function checkpoint(pr) {
    if (!(pr.body || "").includes(MARKER)) {
      throw new Error("The sync branch belongs to an unrecognized PR; refusing to modify it.");
    }
    const markers = [...pr.body.matchAll(SOURCE_MARKER)];
    if (markers.length !== 1) {
      throw new Error("The previous sync PR has no unique source SHA marker; reconcile it manually.");
    }
    return markers[0][1];
  }

  function result(status, pr) {
    core.info("Forward sync: " + status);
    core.setOutput("status", status);
    if (syncBranch) core.setOutput("sync_branch", syncBranch);
    if (status === "conflict-needs-resolution") {
      core.warning("Source conflicts were published for maintainer resolution on " +
        syncBranch + (pr ? ". Draft PR: " + pr.html_url : "."));
    }
    if (pr) {
      core.setOutput("pull_request_number", pr.number);
      core.setOutput("pull_request_url", pr.html_url);
    }
    return { status, pullRequest: pr ? pr.number : null,
      pullRequestUrl: pr ? pr.html_url : null };
  }

  const sourceSha = await ref(SOURCE);
  const targetSha = await ref(TARGET);
  core.setOutput("source_sha", sourceSha);
  core.setOutput("target_sha", targetSha);
  if (await isIncluded(sourceSha, targetSha)) return result("already-included");

  const allPrs = await github.paginate(api.pulls.list, {
    ...scope, state: "all", base: TARGET,
    sort: "created", direction: "desc", per_page: 100,
  });
  const prs = allPrs.filter((pr) => pr.head.repo &&
    pr.head.repo.full_name.toLowerCase() === (owner + "/" + repo).toLowerCase() &&
    pr.head.ref.startsWith(SYNC_PREFIX));
  const open = prs.filter((pr) => pr.state === "open");
  if (open.length > 1) {
    throw new Error("Multiple open forward-sync PRs exist; reconcile them manually.");
  }
  const previous = open[0] || prs[0];
  const completed = prs.filter((pr) => pr.merged_at)
    .sort((a, b) => b.merged_at.localeCompare(a.merged_at))[0];
  if (previous && previous.state === "closed" && !previous.merged_at) {
    throw new Error("The last sync PR #" + previous.number +
      " was closed without merging. Reopen it or reconcile the sync branch manually.");
  }
  for (const pr of new Set([previous, completed].filter(Boolean))) {
    if (!(await isIncluded(checkpoint(pr), sourceSha))) {
      throw new Error(SOURCE +
        " was rewound or rewritten since the previous sync; reconcile manually before continuing.");
    }
  }
  const reviewedSha = completed ? checkpoint(completed) : null;
  if (completed) {
    if (!(await isIncluded(completed.head.sha, targetSha))) {
      throw new Error("The previous merged PR head is not in " + TARGET +
        ". A squash/rebase merge needs manual ancestry reconciliation.");
    }
    if (!open.length) {
      const oldTip = await ref(completed.head.ref, true);
      if (oldTip && !(await isIncluded(oldTip, targetSha))) {
        throw new Error("The previous sync branch has commits not included in " + TARGET +
          "; reconcile those manual commits before starting another sync.");
      }
      if (reviewedSha === sourceSha) return result("already-processed");
    }
    core.setOutput("reviewed_source_sha", reviewedSha);
  }

  syncBranch = open.length ? open[0].head.ref : SYNC_PREFIX + sourceSha;
  const syncSha = await ref(syncBranch, true);
  if (open.length && !syncSha) {
    throw new Error("The open sync PR's branch is missing; restore it manually.");
  }
  // Only target-derived branches are eligible for replay or safe retries.
  // Source checkpoints are metadata, never parents of generated commits.
  const expectedAncestor = completed ? completed.head.sha : targetSha;
  if (syncSha && !(await isIncluded(expectedAncestor, syncSha))) {
    throw new Error("The existing sync branch does not contain the reviewed target; " +
      "rebuild the legacy source snapshot before retrying. No commits were overwritten.");
  }
  const processedSha = open.length ? checkpoint(open[0]) : reviewedSha;
  if (syncSha && processedSha && !(await isIncluded(processedSha, targetSha)) &&
      (await isIncluded(processedSha, syncSha))) {
    throw new Error("The sync branch contains excluded source ancestry from the old ACK mechanism. " +
      "Rebuild it using replay before retrying; no commits were overwritten.");
  }
  let pr = open[0];
  async function ensureDraft() {
    if (!pr || pr.draft) return;
    if (!pr.node_id) throw new Error("The open PR has no node ID for safe draft conversion.");
    await github.graphql(`mutation($pullRequestId: ID!) {
      convertPullRequestToDraft(input: {pullRequestId: $pullRequestId}) {
        pullRequest { id isDraft }
      }
    }`, { pullRequestId: pr.node_id });
    pr = { ...pr, draft: true };
  }
  const replaySource = replay || require("./forward-sync-git.cjs");
  const outcome = await replaySource({
    sourceSha, baseSha: syncSha || targetSha, checkpointSha: processedSha,
    syncBranch, repository: owner + "/" + repo,
    token: process.env.FORWARD_SYNC_TOKEN, core, existingBranch: Boolean(syncSha),
    onConflict: ensureDraft,
  });
  if (outcome.conflicted) await ensureDraft();
  if (!outcome.conflicted && !open.length && !outcome.changed && (!syncSha || syncSha === targetSha)) {
    return result("already-applied");
  }

  const processedCheckpoint = outcome.conflicted ? outcome.checkpointSha : sourceSha;
  if (!/^[0-9a-f]{40}$/.test(processedCheckpoint || "")) {
    throw new Error("Replay did not retain a valid processed source checkpoint.");
  }
  const sourceMarker = "<!-- forward-sync-source: " + processedCheckpoint + " -->";
  const pendingSha = outcome.pendingSha || sourceSha;
  const conflictDetails = outcome.conflicted ? [
    "<!-- forward-sync-conflicts:start -->",
    "### Conflicts awaiting maintainer resolution",
    "Pending source: [`" + pendingSha + "`](https://github.com/" + owner + "/" + repo + "/commit/" + pendingSha + ").",
    "",
    ...(outcome.conflictFiles || []).map((file) => "- `" + file.replace(/`/g, "\\`") + "`"),
    "",
    "1. Resolve the committed conflict markers and keep the target's 0.32 behavior and backend CI switches.",
    "2. Set `status` to `resolved` in `.github/forward-sync-pending.json`, commit, and push your resolution.",
    "3. Start a new manual run on `0.4.0-dev` (`gh workflow run forward-sync.yml --ref 0.4.0-dev`) to validate the resolution and resume pending increments.",
    "4. Mark this PR ready for review after resolution and CI validation. Do not merge source ancestry.",
    "<!-- forward-sync-conflicts:end -->",
  ].join("\n") : "";
  if (pr) {
    let body = pr.body.replace(PENDING_MARKER, "").replace(CONFLICT_BLOCK, "").replace(SOURCE_MARKER, sourceMarker);
    if (outcome.conflicted) body += "\n<!-- forward-sync-pending: " + pendingSha + " -->\n" + conflictDetails;
    if (body !== pr.body) {
      pr = (await api.pulls.update({ ...scope, pull_number: pr.number, body })).data;
    }
    pr = (await api.pulls.get({ ...scope, pull_number: pr.number })).data;
    if (pr.state !== "open") {
      throw new Error("The sync PR closed while its branch was being updated. Review " +
        pr.html_url + " and reconcile " + syncBranch + " before retrying.");
    }
    return result(outcome.conflicted ? "conflict-needs-resolution" : "pr-updated", pr);
  }

  const body = [
    MARKER, sourceMarker,
    "### Description",
    "Forward-sync new " + SOURCE + " commits into " + TARGET + ".",
    "",
    "The source marker records the reviewed source checkpoint, including intentional exclusions.",
    "New cycles replay only first-parent commits after that checkpoint onto the target.",
    "Patch-equivalent changes and commits already recorded by cherry-pick trailers are skipped.",
    "Source commit IDs are recorded as metadata only; excluded ancestors are not merged into this branch.",
    "Automatic merging is disabled.",
    "",
    "### Review and conflict resolution",
    "- Resolve pending conflicts on " + syncBranch + ", commit the resolution, and rerun this workflow.",
    "- Keep the 0.32 adaptation and backend CI enablement on " + TARGET + ".",
    "- Do not merge the source branch or record source checkpoints as merge parents.",
    "- Use a merge commit when merging this PR into the target to retain its reviewed head.",
    "- Verify NVIDIA CI before manually merging.",
    "",
    "GitHub may require a maintainer to approve PR workflow runs created by GITHUB_TOKEN.",
    ...(outcome.conflicted ? ["", "<!-- forward-sync-pending: " + pendingSha + " -->", conflictDetails] : []),
  ].join("\n");
  pr = (await api.pulls.create({
    ...scope, title: "chore: forward-sync " + SOURCE + " into " + TARGET,
    head: syncBranch, base: TARGET, body, draft: Boolean(outcome.conflicted),
  })).data;
  return result(outcome.conflicted ? "conflict-needs-resolution" : "pr-created", pr);
};

module.exports.constants = { SOURCE, TARGET, SYNC_PREFIX, MARKER };
