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

module.exports = async function forwardSync({ github, context, core }) {
  const payload = context.payload || {};
  if (payload.ref === "refs/heads/" + SOURCE && payload.deleted) {
    core.info("Source branch was deleted; no sync was attempted.");
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

  async function compare(base, head) {
    return (await api.repos.compareCommitsWithBasehead({
      ...scope, basehead: base + "..." + head,
    })).data.status;
  }

  async function isIncluded(commit, destination) {
    if (commit === destination) return true;
    return ["behind", "identical"].includes(await compare(destination, commit));
  }

  function result(status, pr) {
    core.info("Forward sync: " + status);
    core.setOutput("status", status);
    if (syncBranch) core.setOutput("sync_branch", syncBranch);
    if (status === "conflict-needs-resolution") {
      core.setFailed("Source changes conflict with the sync branch. Resolve them on " +
        syncBranch + " and retry. PR: " + pr.html_url);
    }
    if (pr) {
      core.setOutput("pull_request_number", pr.number);
      core.setOutput("pull_request_url", pr.html_url);
    }
    return {
      status, pullRequest: pr ? pr.number : null,
      pullRequestUrl: pr ? pr.html_url : null,
    };
  }

  // Reconcile the latest branch tips, including when older queued runs start.
  const sourceSha = await ref(SOURCE);
  const targetSha = await ref(TARGET);
  core.setOutput("source_sha", sourceSha);
  core.setOutput("target_sha", targetSha);
  if (await isIncluded(sourceSha, targetSha)) {
    return result("already-included");
  }

  const allPrs = await github.paginate(api.pulls.list, {
    ...scope, state: "all", base: TARGET,
    sort: "created", direction: "desc", per_page: 100,
  });
  const prs = allPrs.filter((pr) =>
    pr.head.repo && pr.head.repo.full_name.toLowerCase() === (owner + "/" + repo).toLowerCase() &&
    pr.head.ref.startsWith(SYNC_PREFIX));
  const open = prs.filter((pr) => pr.state === "open");
  if (open.length > 1) {
    throw new Error("Multiple open forward-sync PRs exist; reconcile them manually.");
  }
  const previous = open[0] || prs[0];
  if (previous && !(previous.body || "").includes(MARKER)) {
    throw new Error("The sync branch belongs to an unrecognized PR; refusing to modify it.");
  }
  if (previous && previous.state === "closed" && !previous.merged_at) {
    throw new Error("The last sync PR #" + previous.number +
      " was closed without merging. Reopen it or reconcile the sync branch manually.");
  }
  if (previous) {
    const markers = [...(previous.body || "").matchAll(SOURCE_MARKER)];
    if (markers.length !== 1) {
      throw new Error("The previous sync PR has no unique source SHA marker; reconcile it manually.");
    }
    if (!(await isIncluded(markers[0][1], sourceSha))) {
      throw new Error(SOURCE +
        " was rewound or rewritten since the previous sync; reconcile manually before continuing.");
    }
  }

  // Each cycle gets a new source snapshot. Once a PR is open, keep its branch
  // so maintainer fixes survive subsequent source updates.
  syncBranch = open.length ? open[0].head.ref : SYNC_PREFIX + sourceSha;
  if (previous && previous.merged_at) {
    if (!(await isIncluded(previous.head.sha, targetSha))) {
      throw new Error("The previous merged PR head is not in " + TARGET +
        ". A squash/rebase merge needs manual ancestry reconciliation.");
    }
    const oldTip = await ref(previous.head.ref, true);
    if (oldTip && !(await isIncluded(oldTip, targetSha))) {
      throw new Error("The previous sync branch has commits not included in " + TARGET +
        "; reconcile those manual commits before starting another sync.");
    }
  }
  const syncSha = await ref(syncBranch, true);
  if (syncSha && !open.length && syncSha !== sourceSha) {
    throw new Error("The new sync branch already contains unexpected commits; refusing to modify it.");
  }
  if (open.length && !syncSha) {
    throw new Error("The open sync PR's branch is missing; restore it manually.");
  }

  let conflicted = false;
  if (!syncSha) {
    await api.git.createRef({ ...scope, ref: "refs/heads/" + syncBranch, sha: sourceSha });
  } else {
    const status = syncSha === sourceSha ? "identical" : await compare(syncSha, sourceSha);
    if (status === "ahead") {
      // force:false also protects a manual push racing with our earlier read.
      await api.git.updateRef({
        ...scope, ref: "heads/" + syncBranch, sha: sourceSha, force: false,
      });
    } else if (status === "diverged") {
      try {
        await api.repos.merge({
          ...scope, base: syncBranch, head: sourceSha,
          commit_message: "Merge " + SOURCE + " into forward-sync branch",
        });
      } catch (error) {
        if (error.status !== 409) throw error;
        conflicted = true;
        core.warning("Source changes conflict with the sync branch. Merge " + SOURCE +
          " into " + syncBranch + " and resolve conflicts there; all existing commits were preserved.");
      }
    } else if (!["behind", "identical"].includes(status)) {
      throw new Error("Unexpected GitHub comparison status: " + status);
    }
  }

  const sourceMarker = "<!-- forward-sync-source: " + sourceSha + " -->";
  let pr = open[0];
  if (pr) {
    // Preserve the human-authored description; change only our source checkpoint.
    const body = pr.body.replace(SOURCE_MARKER, sourceMarker);
    if (body !== pr.body) {
      pr = (await api.pulls.update({ ...scope, pull_number: pr.number, body })).data;
    }
    // A maintainer may have merged/closed the PR while its branch was updated.
    // Re-read even when the description was unchanged, so new work is not
    // reported as reviewable if there is no longer an open PR.
    pr = (await api.pulls.get({ ...scope, pull_number: pr.number })).data;
    if (pr.state !== "open") {
      throw new Error("The sync PR closed while its branch was being updated. Review " +
        pr.html_url + " and reconcile " + syncBranch + " before retrying.");
    }
    return result(conflicted ? "conflict-needs-resolution" : "pr-updated", pr);
  }

  const body = [
    MARKER,
    sourceMarker,
    "### Description",
    "Forward-sync " + SOURCE + " into " + TARGET + ".",
    "",
    "The machine-readable marker above records the latest observed source SHA.",
    "The workflow creates or updates this PR only. Automatic merging is disabled.",
    "",
    "### Review and conflict resolution",
    "- Resolve conflicts on " + syncBranch + ", then push the resolution to that branch.",
    "- Keep the 0.32 adaptation on " + TARGET + "; review overlapping changes carefully.",
    "- Use a merge commit to preserve branch ancestry. Do not squash or rebase this sync PR.",
    "- If merge commits are unavailable, leave this PR open until a maintainer enables them.",
    "- Verify NVIDIA CI before manually merging. Automatic merge remains a separate future change.",
    "",
    "GitHub may require a maintainer to approve PR workflow runs created by GITHUB_TOKEN.",
    conflicted ? "\nThe latest source could not be merged into the sync branch; resolve it manually." : "",
  ].join("\n");
  pr = (await api.pulls.create({
    ...scope, title: "chore: forward-sync " + SOURCE + " into " + TARGET,
    head: syncBranch, base: TARGET, body,
  })).data;
  return result(conflicted ? "conflict-needs-resolution" : "pr-created", pr);
};

module.exports.constants = { SOURCE, TARGET, SYNC_PREFIX, MARKER };