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
  async function commit(name, files) {
    for (const [file, text] of Object.entries(files)) await fs.writeFile(path.join(work, file), text);
    await git(["add", "."]);
    await git(["commit", "--quiet", "-m", name]);
    return git(["rev-parse", "HEAD"]);
  }
  await git(["init", "--quiet", "--bare", remote]);
  await git(["init", "--quiet", "--initial-branch=source"]);
  const common = await commit("common base", { "seed.txt": "common\n", "conflict.txt": "base\n" });
  const excluded = await commit("excluded CI sync", { "excluded.txt": "must not reach target\n" });
  const checkpoint = await commit("already reviewed source", { "retained.txt": "reviewed\n" });
  let source = await commit("new source delta", { "new.txt": "new delta\n" });
  if (conflict) source = await commit("conflicting source change", { "conflict.txt": "source\n" });
  await git(["checkout", "--quiet", "-b", "target", common]);
  await git(["cherry-pick", "-x", checkpoint]);
  const files = { "adaptation.txt": "target adaptation\n" };
  if (conflict) files["conflict.txt"] = "target\n";
  const target = await commit("target adaptation", files);
  await git(["remote", "add", "origin", remote]);
  await git(["push", "--quiet", "origin", "source", "target"]);
  async function ancestor(base, head) {
    try { await git(["merge-base", "--is-ancestor", base, head]); return true; }
    catch (error) { if (error.code === 1) return false; throw error; }
  }
  async function run(sourceSha = source, baseSha = target, existingBranch = false) {
    return replay({ sourceSha, baseSha, checkpointSha: checkpoint,
      syncBranch: PREFIX + sourceSha, repository: "example/replay", remoteUrl: remote,
      existingBranch, core: { info() {}, warning() {} } });
  }
  async function fetchResult(sourceSha) {
    await git(["fetch", "--quiet", "origin", "refs/heads/" + PREFIX + sourceSha]);
  }
  return { git, commit, run, fetchResult, ancestor, common, excluded, checkpoint, source, target };
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

test("a later conflict discards the entire attempt without pushing or advancing checkpoint", async (t) => {
  const f = await fixture(t, true);
  const result = await f.run();
  assert.deepEqual(result, { sha: f.target, checkpointSha: f.checkpoint, conflicted: true, changed: false });
  assert.equal(await f.git(["ls-remote", "--heads", "origin", "refs/heads/" + PREFIX + f.source]), "");
  const target = await f.git(["ls-remote", "--heads", "origin", "refs/heads/target"]);
  assert.equal(target.split(/\s+/)[0], f.target);
});
