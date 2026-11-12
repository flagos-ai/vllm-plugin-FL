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
const BASE_PREFIX = "codex/sync-baseline-0.4-to-0.5-";
const MARKER = "<!-- vllm-fl-forward-sync: 0.4.0-dev -> 0.5.0-dev -->";
const BASE_MARKER = "<!-- vllm-fl-forward-sync-baseline -->";
const STATE_PATH = ".github/forward-sync-baseline.json";
const SOURCE_MARKER = /<!-- forward-sync-source: ([0-9a-f]{40}) -->/g;
const PROJECTION_MARKER = /<!-- forward-sync-projection: ([0-9a-f]{40}) -->/g;
const QUEUED_MARKER = /\n?<!-- forward-sync-queued: [0-9a-f]{40} -->/g;
const SHA = /^[0-9a-f]{40}$/;

module.exports = async function forwardSync({ github, context, core, project }) {
  const payload = context.payload || {};
  if (payload.ref === "refs/heads/" + SOURCE && payload.deleted) {
    core.setOutput("status", "source-deleted");
    return { status: "source-deleted", pullRequest: null, pullRequestUrl: null };
  }
  if (payload.ref === "refs/heads/" + SOURCE && payload.forced) {
    throw new Error("The source push rewrote history; reconcile manually.");
  }
  const { owner, repo } = context.repo;
  const scope = { owner, repo };
  const api = github.rest;
  async function ref(branch, optional = false) {
    try { return (await api.git.getRef({ ...scope, ref: "heads/" + branch })).data.object.sha; }
    catch (error) { if (optional && error.status === 404) return null; throw error; }
  }
  async function included(commit, destination) {
    if (commit === destination) return true;
    return ["behind", "identical"].includes((await api.repos.compareCommitsWithBasehead({
      ...scope, basehead: destination + "..." + commit,
    })).data.status);
  }
  function marker(pr, regex, required = true) {
    if (!(pr.body || "").includes(MARKER)) throw new Error("Unrecognized sync PR; refusing to modify it.");
    const values = [...pr.body.matchAll(regex)];
    if (values.length === 0 && !required) return null;
    if (values.length !== 1) throw new Error("The sync PR has no unique checkpoint marker.");
    return values[0][1];
  }
  function result(status, pr) {
    core.info("Forward sync: " + status);
    core.setOutput("status", status);
    if (pr) {
      core.setOutput("pull_request_number", pr.number);
      core.setOutput("pull_request_url", pr.html_url);
    }
    return { status, pullRequest: pr ? pr.number : null, pullRequestUrl: pr ? pr.html_url : null };
  }
  async function jsonAt(path, revision, optional = false) {
    try {
      const { data } = await api.repos.getContent({ ...scope, path, ref: revision });
      if (Array.isArray(data) || data.encoding !== "base64") throw new Error("Invalid sync state file.");
      return JSON.parse(Buffer.from(data.content, "base64").toString("utf8"));
    } catch (error) { if (optional && error.status === 404) return null; throw error; }
  }
  function validateState(state) {
    if (!state || state.version !== 1 || state.source !== SOURCE || state.target !== TARGET ||
        !SHA.test(state.sourceCheckpoint || "") || !SHA.test(state.projectionCheckpoint || "")) {
      throw new Error("Invalid source comparison baseline state.");
    }
    return state;
  }

  const sourceSha = await ref(SOURCE);
  const targetSha = await ref(TARGET);
  core.setOutput("source_sha", sourceSha);
  core.setOutput("target_sha", targetSha);
  const all = await github.paginate(api.pulls.list, {
    ...scope, state: "all", base: TARGET, sort: "created", direction: "desc", per_page: 100,
  });
  const sameRepo = pr => pr.head.repo?.full_name.toLowerCase() === (owner + "/" + repo).toLowerCase();
  const managed = all.filter(pr => sameRepo(pr) && pr.head.ref.startsWith(SYNC_PREFIX));
  const opens = managed.filter(pr => pr.state === "open");
  if (opens.length > 1) throw new Error("Multiple open forward-sync PRs; reconcile manually.");
  const open = opens[0];
  const completed = managed.filter(pr => pr.merged_at)
    .sort((a, b) => b.merged_at.localeCompare(a.merged_at))[0];
  if (!open && managed[0]?.state === "closed" && !managed[0].merged_at) {
    throw new Error("The latest sync PR was closed without merging; reconcile manually.");
  }
  const reviewedSource = completed ? marker(completed, SOURCE_MARKER) : null;
  if (!reviewedSource) throw new Error("A reviewed source checkpoint is required to initialize native sync.");
  if (!(await included(reviewedSource, sourceSha))) throw new Error("Source history was rewritten since the reviewed checkpoint.");
  if (!(await included(completed.head.sha, targetSha))) throw new Error("The merged sync head is missing from target; preserve merge ancestry.");

  let state = await jsonAt(STATE_PATH, targetSha, true);
  if (!state) {
    const baselines = all.filter(pr => sameRepo(pr) && pr.head.ref.startsWith(BASE_PREFIX) &&
      (pr.body || "").includes(BASE_MARKER));
    const waiting = baselines.filter(pr => pr.state === "open");
    if (waiting.length > 1) throw new Error("Multiple baseline PRs exist; reconcile manually.");
    if (waiting[0]) return result("baseline-pr-required", waiting[0]);
    if (baselines[0]?.state === "closed") throw new Error("The baseline PR closed without usable target state; reconcile manually.");
    const branch = BASE_PREFIX + targetSha;
    let tip = await ref(branch, true);
    if (!tip) {
      const targetCommit = (await api.git.getCommit({ ...scope, commit_sha: targetSha })).data;
      const sourceCommit = (await api.git.getCommit({ ...scope, commit_sha: reviewedSource })).data;
      const baseline = (await api.git.createCommit({
        ...scope, tree: sourceCommit.tree.sha, parents: [targetSha],
        message: "Record source comparison snapshot " + reviewedSource +
          "\n\nComparison context only; original source parents are not imported.",
      })).data;
      state = { version: 1, source: SOURCE, target: TARGET,
        sourceCheckpoint: reviewedSource, projectionCheckpoint: baseline.sha };
      const tree = (await api.git.createTree({ ...scope, base_tree: targetCommit.tree.sha, tree: [{
        path: STATE_PATH, mode: "100644", type: "blob", content: JSON.stringify(state, null, 2) + "\n",
      }] })).data;
      tip = (await api.git.createCommit({
        ...scope, tree: tree.sha, parents: [targetSha, baseline.sha],
        message: "Initialize forward-sync comparison ancestry without changing runtime files",
      })).data.sha;
      await api.git.createRef({ ...scope, ref: "refs/heads/" + branch, sha: tip });
    } else {
      state = validateState(await jsonAt(STATE_PATH, tip));
      if (state.sourceCheckpoint !== reviewedSource || !(await included(targetSha, tip))) {
        throw new Error("Unexpected existing baseline branch; no work was overwritten.");
      }
    }
    const pr = (await api.pulls.create({
      ...scope, base: TARGET, head: branch, draft: false,
      title: "chore: initialize native forward-sync comparison baseline",
      body: [BASE_MARKER, "### Purpose", "Initialize a comparison baseline for source checkpoint `" + reviewedSource + "`.",
        "", "The final PR diff adds only `" + STATE_PATH + "`. Runtime files and CI enablement retain the target versions.",
        "A synthetic snapshot records the original source tree for Git's three-way comparison; it imports no original source parents.",
        "Its historical snapshot contains the source-side context, including source CI versions, but those versions are not applied to the target by this PR.",
        "", "Merge this PR with a merge commit. Squash/rebase would lose the comparison ancestor.",
        "After merging, dispatch forward-sync on `0.4.0-dev`; it will replace the legacy WIP draft with original source commit projections and GitHub-native conflicts.",
      ].join("\n"),
    })).data;
    return result("baseline-pr-required", pr);
  }
  state = validateState(state);
  if (!(await included(state.projectionCheckpoint, targetSha))) {
    throw new Error("The comparison snapshot is not an ancestor of target; merge the baseline PR without squash/rebase.");
  }

  const completedProjection = marker(completed, PROJECTION_MARKER, false);
  let sourceCheckpoint = completedProjection ? reviewedSource : state.sourceCheckpoint;
  let projectionCheckpoint = completedProjection || state.projectionCheckpoint;
  let legacy = null;
  let branch;
  if (open) {
    const projection = marker(open, PROJECTION_MARKER, false);
    const observed = marker(open, SOURCE_MARKER);
    if (!(await included(observed, sourceSha))) throw new Error("The source rewrote the open PR checkpoint.");
    const tip = await ref(open.head.ref, true);
    if (tip !== open.head.sha) throw new Error("The sync branch moved while reading it; retry.");
    if (projection) {
      if (!(await included(projection, tip))) throw new Error("The open projection history was rewritten.");
      if (tip !== projection) {
        // A maintainer's merge/resolution takes precedence; queue new source
        // changes for the next cycle rather than resetting their branch.
        const body = open.body.replace(QUEUED_MARKER, "") +
          (observed !== sourceSha ? "\n<!-- forward-sync-queued: " + sourceSha + " -->" : "");
        if (body !== open.body) await api.pulls.update({ ...scope, pull_number: open.number, body });
        return result("pr-awaiting-review", open);
      }
      sourceCheckpoint = observed;
      projectionCheckpoint = projection;
      branch = open.head.ref;
    } else {
      const pending = await jsonAt(".github/forward-sync-pending.json", tip);
      const commit = (await api.git.getCommit({ ...scope, commit_sha: tip })).data;
      if (!open.draft || pending.version !== 1 || pending.status !== "needs-resolution" ||
          pending.checkpointSha !== observed || !commit.message.startsWith("WIP:") || commit.parents.length !== 1 ||
          !(await included(commit.parents[0].sha, targetSha))) {
        throw new Error("The legacy draft contains maintainer work; replace it manually without overwriting commits.");
      }
      legacy = open;
    }
  }
  if (!open && sourceCheckpoint === sourceSha) return result("already-processed");
  const build = project || require("./forward-sync-projection.cjs");
  const projected = await build({ github, scope, sourceSha, sourceCheckpoint, projectionCheckpoint });
  const projectionSha = projected.projectionSha;
  branch = branch || SYNC_PREFIX + (legacy ? "native-" : "") + sourceSha;
  const existing = await ref(branch, true);
  if (!existing) {
    await api.git.createRef({ ...scope, ref: "refs/heads/" + branch, sha: projectionSha });
  } else if (existing !== projectionSha) {
    if (!(await included(existing, projectionSha))) throw new Error("The sync branch contains independent work; no force push was attempted.");
    await api.git.updateRef({ ...scope, ref: "heads/" + branch, sha: projectionSha, force: false });
  }
  const sourceLine = "<!-- forward-sync-source: " + sourceSha + " -->";
  const projectionLine = "<!-- forward-sync-projection: " + projectionSha + " -->";
  if (open && !legacy) {
    const body = open.body.replace(SOURCE_MARKER, sourceLine).replace(PROJECTION_MARKER, projectionLine).replace(QUEUED_MARKER, "");
    if (body !== open.body) await api.pulls.update({ ...scope, pull_number: open.number, body });
    return result("pr-updated", open);
  }
  const pr = (await api.pulls.create({
    ...scope, base: TARGET, head: branch, draft: true,
    title: "chore: forward-sync " + SOURCE + " into " + TARGET,
    body: [MARKER, sourceLine, projectionLine, "### Source commits", "",
      ...projected.commits.map(c => "- [`" + c.sourceSha.slice(0, 12) + "`](https://github.com/" + owner + "/" + repo +
        "/commit/" + c.sourceSha + ") " + c.message.split("\n")[0] + " — " + c.author.name),
      "", "Original source authors, author dates, and full messages are preserved in the projected commits; GitHub Actions is the committer.",
      "GitHub performs the three-way merge against the target and displays native conflicts. No conflict markers are committed as ordinary source text.",
      "Resolve conflicts by merging the target into this branch, preserve the target's 0.32 behavior and CI switches, and mark ready after CI validation.",
      "Do not merge the original source branch, which contains deliberately excluded history. Automatic merging is disabled.",
      ...(legacy ? ["", "Replaces legacy WIP draft #" + legacy.number + "."] : []),
    ].join("\n"),
  })).data;
  if (legacy) {
    const current = (await api.pulls.get({ ...scope, pull_number: legacy.number })).data;
    if (current.state === "open" && current.head.sha === legacy.head.sha && current.draft) {
      await api.pulls.update({ ...scope, pull_number: legacy.number, state: "closed" });
    } else {
      core.warning("The legacy PR changed during replacement; it was left untouched.");
    }
  }
  return result("pr-created", pr);
};

module.exports.constants = { SOURCE, TARGET, SYNC_PREFIX, BASE_PREFIX, MARKER, BASE_MARKER, STATE_PATH };
