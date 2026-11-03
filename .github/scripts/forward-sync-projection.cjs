/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */
"use strict";

const SHA = /^[0-9a-f]{40}$/;
const MAX_COMMITS = 512;
const isSha = (value) => typeof value === "string" && value.length === 40 && SHA.test(value);
const isScopePart = (value) => typeof value === "string" && value.length > 0 && !/[^A-Za-z0-9_.-]/.test(value);

/** Copy source trees onto an acknowledged source baseline without source parents. */
module.exports = async function project({
  github, scope, sourceSha, sourceCheckpoint, projectionCheckpoint,
}) {
  for (const [name, sha] of Object.entries({ sourceSha, sourceCheckpoint, projectionCheckpoint })) {
    if (!isSha(sha)) throw new Error("Invalid " + name + " for forward-sync projection.");
  }
  if (!scope || !isScopePart(scope.owner) || !isScopePart(scope.repo)) {
    throw new Error("Invalid repository scope for forward-sync projection.");
  }
  const api = github && github.rest && github.rest.git;
  if (!api || typeof api.getCommit !== "function" || typeof api.createCommit !== "function") {
    throw new Error("The GitHub Git commit API is required for forward-sync projection.");
  }
  const repository = { owner: scope.owner, repo: scope.repo };
  const cache = new Map();
  async function getCommit(sha) {
    if (!cache.has(sha)) {
      const { data } = await api.getCommit({ ...repository, commit_sha: sha });
      if (!data || data.sha !== sha || !data.tree || !isSha(data.tree.sha)) {
        throw new Error("Invalid Git commit or tree returned for " + sha + ".");
      }
      cache.set(sha, data);
    }
    return cache.get(sha);
  }

  const [sourceBase, projectionBase] = await Promise.all([
    getCommit(sourceCheckpoint), getCommit(projectionCheckpoint),
  ]);
  if (sourceBase.tree.sha !== projectionBase.tree.sha) {
    throw new Error("The projection checkpoint tree does not exactly match the raw source checkpoint tree; reconcile the source baseline manually.");
  }

  // Validate the complete first-parent range before creating any Git objects.
  const pending = [];
  const seen = new Set();
  let cursor = sourceSha;
  while (cursor !== sourceCheckpoint) {
    if (pending.length >= MAX_COMMITS) {
      throw new Error("The source first-parent range exceeds the 512-commit projection limit; reconcile the checkpoint manually.");
    }
    if (seen.has(cursor)) throw new Error("A cycle was returned in the source first-parent history.");
    seen.add(cursor);
    const commit = await getCommit(cursor);
    if (!Array.isArray(commit.parents) || commit.parents.length === 0) {
      throw new Error("The reviewed source checkpoint is not on the current source first-parent history (root reached); reconcile the rewrite manually.");
    }
    const parent = commit.parents[0] && commit.parents[0].sha;
    if (!isSha(parent)) throw new Error("Invalid source first parent for " + cursor + ".");
    const author = commit.author;
    if (!author || typeof author.name !== "string" || !author.name ||
        typeof author.email !== "string" || !author.email ||
        typeof author.date !== "string" || !Number.isFinite(Date.parse(author.date))) {
      throw new Error("The original source author and date are missing or invalid for " + cursor + ".");
    }
    if (typeof commit.message !== "string") {
      throw new Error("The original source commit message is missing for " + cursor + ".");
    }
    const committerDate = commit.committer && commit.committer.date;
    if (committerDate && (typeof committerDate !== "string" || !Number.isFinite(Date.parse(committerDate)))) {
      throw new Error("The original source committer date is invalid for " + cursor + ".");
    }
    pending.push(commit);
    cursor = parent;
  }

  let projectionSha = projectionCheckpoint;
  const commits = [];
  for (const source of pending.reverse()) {
    // Preserve every character of the original message; only append the -x trailer.
    const separator = source.message.endsWith("\n\n") ? "" :
      (source.message.endsWith("\n") ? "\n" : "\n\n");
    const message = source.message + separator + "(cherry picked from commit " + source.sha + ")";
    const author = { name: source.author.name, email: source.author.email, date: source.author.date };
    // Stable metadata makes a retry recreate the same Git object after a partial API failure.
    const committer = {
      name: "github-actions[bot]",
      email: "41898282+github-actions[bot]@users.noreply.github.com",
      date: (source.committer && source.committer.date) || source.author.date,
    };
    const { data } = await api.createCommit({
      ...repository, message, tree: source.tree.sha, parents: [projectionSha], author, committer,
    });
    if (!data || !isSha(data.sha)) {
      throw new Error("Invalid projected commit SHA returned for " + source.sha + ".");
    }
    projectionSha = data.sha;
    commits.push({ sourceSha: source.sha, projectionSha, author, message });
  }
  return { projectionSha, commits };
};
