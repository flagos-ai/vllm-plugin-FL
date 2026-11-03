/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const replay = require("../forward-sync-git.cjs");
const exec = promisify(execFile);
const PREFIX = "codex/sync-0.4-to-0.5-";

async function fixture(t, conflict = false) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "vllm-fl-replay-test-"));
  t.after(async () => {
    const resolved = path.resolve(root);
    assert.equal(path.dirname(resolved), path.resolve(os.tmpdir()));
    assert.ok(path.basename(resolved).startsWith("vllm-fl-replay-test-"));
    await fs.rm(resolved, { recursive: true, force: true });
  });
  const work = path.join(root, "work");
  const remote = path.join(root, "remote.git");
  await fs.mkdir(work);
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith("GIT_")) delete env[key];
  Object.assign(env, {
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_CONFIG_GLOBAL: process.platform === "win32" ? "NUL" : "/dev/null",
    GIT_AUTHOR_NAME: "Replay test", GIT_AUTHOR_EMAIL: "replay@example.invalid",
    GIT_COMMITTER_NAME: "Replay test", GIT_COMMITTER_EMAIL: "replay@example.invalid",
    GIT_TERMINAL_PROMPT: "0", GIT_DEFAULT_HASH: "sha1",
  });
  async function git(args) {
    return (await exec("git", ["-c", "commit.gpgSign=false", "-c", "core.autocrlf=false",
      "-c", "core.hooksPath=" + (process.platform === "win32" ? "NUL" : "/dev/null"),
      ...args], { cwd: work, env, encoding: "utf8", windowsHide: true })).stdout.trim();
  }
  async function commit(name, files, executableFiles = []) {
    for (const [file, text] of Object.entries(files)) {
      await fs.mkdir(path.dirname(path.join(work, file)), { recursive: true });
      await fs.writeFile(path.join(work, file), text);
    }
    await git(["add", "."]);
    for (const file of executableFiles) {
      await fs.chmod(path.join(work, file), 0o755);
      await git(["update-index", "--chmod=+x", "--", file]);
    }
    await git(["commit", "--quiet", "-m", name]);
    return git(["rev-parse", "HEAD"]);
  }
  await git(["init", "--quiet", "--bare", remote]);
  await git(["init", "--quiet", "--initial-branch=source"]);
  const common = await commit("common base", {
    "seed.txt": "common\n", "conflict.txt": "base\n", "run.sh": "#!/bin/sh\necho base\n",
  }, ["run.sh"]);
  const excluded = await commit("excluded CI sync", { "excluded.txt": "must not reach target\n" });
  const checkpoint = await commit("already reviewed source", { "retained.txt": "reviewed\n" });
  const delta = await commit("new source delta", { "new.txt": "new delta\n" });
  let source = delta;
  let pendingSha;
  if (conflict) {
    pendingSha = await commit("conflicting source change", {
      "conflict.txt": "source\n", "run.sh": "#!/bin/sh\necho source\n",
      "clean-in-conflict.txt": "clean change from conflicting source\n",
    });
    source = await commit("source after conflict", { "after-conflict.txt": "later source delta\n" });
  }
  await git(["checkout", "--quiet", "-b", "target", common]);
  await git(["cherry-pick", "-x", checkpoint]);
  const files = { "adaptation.txt": "target adaptation\n" };
  if (conflict) {
    files["conflict.txt"] = "target\n";
    files["run.sh"] = "#!/bin/sh\necho target\n";
  }
  const target = await commit("target adaptation", files);
  await git(["remote", "add", "origin", remote]);
  await git(["push", "--quiet", "origin", "source", "target"]);
  async function ancestor(base, head) {
    try { await git(["merge-base", "--is-ancestor", base, head]); return true; }
    catch (error) { if (error.code === 1) return false; throw error; }
  }
  async function run(sourceSha = source, baseSha = target, existingBranch = false, options = {}) {
    return replay({ sourceSha, baseSha, checkpointSha: checkpoint,
      syncBranch: PREFIX + sourceSha, repository: "example/replay", remoteUrl: remote,
      existingBranch, core: { info() {}, warning() {} }, ...options });
  }
  async function fetchResult(sourceSha) {
    await git(["fetch", "--quiet", "origin", "refs/heads/" + PREFIX + sourceSha]);
  }
  return { git, commit, run, fetchResult, ancestor, common, excluded, checkpoint, delta, pendingSha, source, target };
}

test("replayed increments exclude old source ancestry and retain reviewed target files", async (t) => {
  const f = await fixture(t);
  const result = await f.run();
  assert.equal(result.conflicted, false);
  assert.equal(result.changed, true);
  assert.equal(result.checkpointSha, f.source);
  await f.fetchResult(f.source);
  assert.equal(await f.ancestor(f.target, result.sha), true);
  for (const original of [f.excluded, f.checkpoint, f.source]) {
    assert.equal(await f.ancestor(original, result.sha), false);
  }
  const files = await f.git(["ls-tree", "--name-only", result.sha]);
  assert.doesNotMatch(files, /excluded\.txt/);
  for (const file of ["retained.txt", "new.txt", "adaptation.txt"]) assert.ok(files.includes(file));
  assert.equal(await f.git(["show", result.sha + ":new.txt"]), "new delta");
  const messages = await f.git(["log", "--format=%B", f.target + ".." + result.sha]);
  assert.ok(messages.includes("(cherry picked from commit " + f.source + ")"));
  const retried = await f.run(f.source, result.sha, true);
  assert.equal(retried.sha, result.sha);
  assert.equal(retried.changed, false);
  const remote = await f.git(["ls-remote", "--heads", "origin", "refs/heads/" + PREFIX + f.source]);
  assert.equal(remote.split(/\s+/)[0], result.sha);
});

test("merge commits replay only their first-parent delta without side-parent ancestry", async (t) => {
  const f = await fixture(t);
  await f.git(["checkout", "--quiet", "source"]);
  await f.git(["checkout", "--quiet", "-b", "side"]);
  const side = await f.commit("side branch work", { "side.txt": "side delta\n" });
  await f.git(["checkout", "--quiet", "source"]);
  await f.commit("mainline work", { "main.txt": "main delta\n" });
  await f.git(["merge", "--no-ff", "side", "-m", "merge side changes"]);
  const source = await f.git(["rev-parse", "HEAD"]);
  await f.git(["push", "--quiet", "origin", "source"]);
  const result = await f.run(source);
  assert.equal(result.conflicted, false);
  await f.fetchResult(source);
  for (const original of [f.excluded, side, source]) {
    assert.equal(await f.ancestor(original, result.sha), false);
  }
  assert.equal(await f.ancestor(f.target, result.sha), true);
  assert.equal(await f.git(["rev-list", "--count", f.target + ".." + result.sha]), "3");
  assert.equal(await f.git(["show", result.sha + ":side.txt"]), "side delta");
  assert.equal(await f.git(["show", result.sha + ":main.txt"]), "main delta");
});

test("conflict snapshots stay linear, retry unchanged, and resume only after staff resolution", async (t) => {
  const f = await fixture(t, true);
  const branch = "refs/heads/" + PREFIX + f.source;
  const manifestPath = ".github/forward-sync-pending.json";
  let draftCalls = 0;
  const onConflict = async () => {
    draftCalls++;
    // The callback must complete before a new conflict snapshot is pushed.
    assert.equal(await f.git(["ls-remote", "--heads", "origin", branch]), "");
  };
  await assert.rejects(f.run(f.source, f.target, false, {
    onConflict: async () => { await onConflict(); throw new Error("draft conversion rejected"); },
  }), /draft conversion rejected/);
  assert.equal(await f.git(["ls-remote", "--heads", "origin", branch]), "");

  const result = await f.run(f.source, f.target, false, { onConflict });
  assert.equal(draftCalls, 2);
  assert.deepEqual(result, {
    sha: result.sha, checkpointSha: f.delta, conflicted: true, changed: true,
    pendingSha: f.pendingSha, conflictFiles: ["conflict.txt", "run.sh"],
  });
  await f.fetchResult(f.source);
  const manifest = JSON.parse(await f.git(["show", result.sha + ":" + manifestPath]));
  assert.deepEqual(manifest, {
    version: 1, status: "needs-resolution", sourceSha: f.pendingSha,
    checkpointSha: f.delta, conflictFiles: result.conflictFiles,
  });
  const parentLine = (await f.git(["rev-list", "--parents", "-n", "1", result.sha])).split(" ");
  assert.equal(parentLine.length, 2);
  assert.equal(await f.ancestor(f.target, result.sha), true);
  for (const original of [f.excluded, f.checkpoint, f.delta, f.pendingSha, f.source]) {
    assert.equal(await f.ancestor(original, result.sha), false);
  }
  assert.equal(await f.git(["show", result.sha + ":new.txt"]), "new delta");
  assert.equal(await f.git(["show", result.sha + ":clean-in-conflict.txt"]), "clean change from conflicting source");
  const files = await f.git(["ls-tree", "--name-only", result.sha]);
  assert.doesNotMatch(files, /excluded\.txt|after-conflict\.txt/);
  for (const file of result.conflictFiles) {
    const content = await f.git(["show", result.sha + ":" + file]);
    assert.match(content, /^<<<<<<< /m);
    assert.match(content, /^>>>>>>> /m);
  }
  assert.match(await f.git(["ls-tree", result.sha, "--", "run.sh"]), /^100755 /);
  assert.match(await f.git(["ls-tree", result.sha, "--", "seed.txt"]), /^100644 /);
  const wipMessage = await f.git(["log", "-1", "--format=%B", result.sha]);
  assert.ok(wipMessage.includes("Source-SHA: " + f.pendingSha));
  assert.doesNotMatch(wipMessage, /\(cherry picked from commit /);

  let retryDraftCalls = 0;
  const retryOptions = { checkpointSha: result.checkpointSha, onConflict: async () => { retryDraftCalls++; } };
  const retried = await f.run(f.source, result.sha, true, retryOptions);
  assert.deepEqual(retried, { ...result, changed: false });
  assert.equal(retryDraftCalls, 1);
  assert.equal((await f.git(["ls-remote", "--heads", "origin", branch])).split(/\s+/)[0], result.sha);

  // A resolved flag cannot bypass conflict markers that staff have not removed.
  await f.git(["checkout", "--quiet", "--detach", result.sha]);
  const flagged = await f.commit("mark resolved prematurely", {
    [manifestPath]: JSON.stringify({ ...manifest, status: "resolved" }) + "\n",
  });
  await f.git(["push", "--quiet", "origin", "HEAD:" + branch]);
  const stillPending = await f.run(f.source, flagged, true, retryOptions);
  assert.deepEqual(stillPending, { ...result, sha: flagged, changed: false });
  assert.equal(retryDraftCalls, 2);
  assert.equal((await f.git(["ls-remote", "--heads", "origin", branch])).split(/\s+/)[0], flagged);

  const resolved = await f.commit("staff resolves source conflicts", {
    "conflict.txt": "staff resolution\n", "run.sh": "#!/bin/sh\necho resolved\n",
  });
  await f.git(["push", "--quiet", "origin", "HEAD:" + branch]);
  const resumed = await f.run(f.source, resolved, true, retryOptions);
  assert.equal(resumed.conflicted, false);
  assert.equal(resumed.changed, true);
  assert.equal(resumed.checkpointSha, f.source);
  assert.equal(retryDraftCalls, 2);
  await f.fetchResult(f.source);
  assert.equal(await f.ancestor(resolved, resumed.sha), true);
  assert.equal(await f.ancestor(f.pendingSha, resumed.sha), false);
  assert.equal(await f.ancestor(f.excluded, resumed.sha), false);
  assert.equal(await f.git(["show", resumed.sha + ":conflict.txt"]), "staff resolution");
  assert.equal(await f.git(["show", resumed.sha + ":after-conflict.txt"]), "later source delta");
  assert.doesNotMatch(await f.git(["ls-tree", "--name-only", "-r", resumed.sha]), /forward-sync-pending\.json/);
  assert.equal(await f.git(["rev-list", "--count", resolved + ".." + resumed.sha]), "2");
  const acknowledgement = await f.git(["log", "-1", "--format=%B", resumed.sha + "^"]);
  assert.ok(acknowledgement.includes("(cherry picked from commit " + f.pendingSha + ")"));
  assert.ok(!acknowledgement.includes("(cherry picked from commit " + f.delta + ")"));
  const again = await f.run(f.source, resumed.sha, true, { checkpointSha: resumed.checkpointSha });
  assert.deepEqual(again, { ...resumed, changed: false });
  const target = await f.git(["ls-remote", "--heads", "origin", "refs/heads/target"]);
  assert.equal(target.split(/\s+/)[0], f.target);
});
