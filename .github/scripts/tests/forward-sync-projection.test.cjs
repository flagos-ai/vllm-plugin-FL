/* Copyright 2026 FlagOS Contributors. SPDX-License-Identifier: Apache-2.0 */
"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const { execFile } = require("node:child_process");
const { createHash } = require("node:crypto");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");
const project = require("../forward-sync-projection.cjs");
const exec = promisify(execFile);
const sha = (value) => value.toString(16).padStart(40, "0");
const CHECKPOINT = sha(1), PROJECTION = sha(2), TARGET = sha(3), EXCLUDED = sha(4);
const scope = { owner: "flagos-ai", repo: "vllm-plugin-FL" };
const author = { name: "Physics", email: "3150105638@zju.edu.cn", date: "2026-11-03T15:44:02+08:00" };

function commit(id, parents, overrides = {}) {
  return { sha: id, parents: parents.map((parent) => ({ sha: parent })),
    tree: { sha: sha(100) }, author: { ...author }, message: "Original source message", ...overrides };
}

function fixture(entries = [], options = {}) {
  const objects = new Map([
    [CHECKPOINT, commit(CHECKPOINT, [EXCLUDED])],
    [PROJECTION, commit(PROJECTION, [TARGET])],
    ...entries.map((entry) => [entry.sha, entry]),
  ]);
  const reads = [], writes = [];
  const github = { rest: { git: {
    async getCommit(args) {
      assert.equal(args.owner, scope.owner);
      assert.equal(args.repo, scope.repo);
      reads.push(args.commit_sha);
      assert.ok(objects.has(args.commit_sha), "unexpected commit read: " + args.commit_sha);
      return { data: structuredClone(objects.get(args.commit_sha)) };
    },
    async createCommit(args) {
      writes.push(structuredClone(args));
      return { data: { sha: createHash("sha1").update(JSON.stringify(args)).digest("hex") } };
    },
  } } };
  return { objects, reads, writes,
    run: (extra = {}) => project({ github, scope, sourceSha: CHECKPOINT,
      sourceCheckpoint: CHECKPOINT, projectionCheckpoint: PROJECTION, ...options, ...extra }) };
}

test("an unchanged source checkpoint creates no commits", async () => {
  const f = fixture();
  assert.deepEqual(await f.run(), { projectionSha: PROJECTION, commits: [] });
  assert.equal(f.writes.length, 0);
});

test("copies original authors, dates and full messages onto only projected parents", async () => {
  const first = sha(11), second = sha(12), merged = sha(13), side = sha(14);
  const originalMessage = "upgrade(musa): adapt MUSA (#575)\n\nFull original body.\n\nCo-authored-by: Other <other@example.invalid>\n";
  const entries = [
    commit(first, [CHECKPOINT], { tree: { sha: sha(111) }, message: originalMessage, committer: { date: "2026-11-03T15:44:05+08:00" } }),
    commit(second, [first], { tree: { sha: sha(112) },
      author: { name: "Other Author", email: "other@example.invalid", date: "2026-10-01T09:01:02Z" } }),
    commit(merged, [second, side], { tree: { sha: sha(113) }, message: "Merge source work\n\nKeep net first-parent delta." }),
  ];
  const f = fixture(entries);
  const result = await f.run({ sourceSha: merged });
  assert.equal(result.projectionSha, result.commits[2].projectionSha);
  assert.deepEqual(result.commits.map((item) => item.sourceSha), [first, second, merged]);
  assert.deepEqual(f.writes.map((item) => item.parents), [[PROJECTION], [result.commits[0].projectionSha], [result.commits[1].projectionSha]]);
  assert.deepEqual(f.writes.map((item) => item.tree), entries.map((item) => item.tree.sha));
  for (let i = 0; i < entries.length; i++) {
    const original = entries[i], created = f.writes[i];
    assert.deepEqual(created.author, original.author);
    assert.deepEqual(created.committer, {
      name: "github-actions[bot]", email: "41898282+github-actions[bot]@users.noreply.github.com",
      date: (original.committer && original.committer.date) || original.author.date,
    });
    assert.ok(created.message.startsWith(original.message));
    assert.ok(created.message.endsWith("(cherry picked from commit " + original.sha + ")"));
    assert.deepEqual(result.commits[i], { sourceSha: original.sha, projectionSha: result.commits[i].projectionSha,
      author: original.author, message: created.message });
    for (const parent of created.parents) {
      assert.equal([EXCLUDED, CHECKPOINT, first, second, merged, side].includes(parent), false);
    }
  }
  assert.equal(f.writes[0].message, originalMessage + "\n(cherry picked from commit " + first + ")");
  assert.equal(f.reads.includes(EXCLUDED), false);
  assert.equal(f.reads.includes(side), false);
});

test("retries recreate identical projected commit objects with deterministic bot committer metadata", async () => {
  const first = sha(23), tip = sha(24);
  const f = fixture([commit(first, [CHECKPOINT]), commit(tip, [first], { committer: { date: "2026-11-03T16:01:02Z" } })]);
  const result = await f.run({ sourceSha: tip });
  const retried = await f.run({ sourceSha: tip });
  assert.deepEqual(retried, result);
  assert.deepEqual(f.writes.slice(0, 2), f.writes.slice(2));
});

test("does not remove existing original-message whitespace before the trailer", async () => {
  const source = sha(15), message = "Subject  \n\nBody with trailing spaces  \n\n";
  const f = fixture([commit(source, [CHECKPOINT], { message })]);
  await f.run({ sourceSha: source });
  assert.equal(f.writes[0].message, message + "(cherry picked from commit " + source + ")");
});

test("requires identical raw source and projection checkpoint trees", async () => {
  const f = fixture([commit(PROJECTION, [TARGET], { tree: { sha: sha(999) } })]);
  await assert.rejects(f.run(), /does not exactly match the raw source checkpoint tree/);
  assert.equal(f.writes.length, 0);
});

test("rejects rewritten first-parent histories and missing checkpoint roots before writing", async () => {
  const root = sha(16), tip = sha(17);
  const f = fixture([commit(root, []), commit(tip, [root])]);
  await assert.rejects(f.run({ sourceSha: tip }), /not on the current source first-parent history.*root reached.*rewrite/);
  assert.equal(f.writes.length, 0);
});

test("a checkpoint reachable only through a merge side parent is not accepted", async () => {
  const root = sha(18), merged = sha(19);
  const f = fixture([commit(root, []), commit(merged, [root, CHECKPOINT])]);
  await assert.rejects(f.run({ sourceSha: merged }), /source first-parent history/);
  assert.equal(f.writes.length, 0);
});

test("bounds the first-parent walk to 512 new commits without partial writes", async () => {
  const entries = [];
  let parent = CHECKPOINT;
  for (let i = 0; i < 513; i++) {
    const id = sha(2000 + i);
    entries.push(commit(id, [parent]));
    parent = id;
  }
  const f = fixture(entries);
  await assert.rejects(f.run({ sourceSha: parent }), /512-commit projection limit/);
  assert.equal(f.writes.length, 0);
});

test("validates SHAs and scope before any API writes", async () => {
  for (const field of ["sourceSha", "sourceCheckpoint", "projectionCheckpoint"]) {
    const f = fixture();
    for (const invalid of ["refs/heads/source", CHECKPOINT + "\n", CHECKPOINT.toUpperCase().replace(/0/g, "A"), null]) {
      await assert.rejects(f.run({ [field]: invalid }), new RegExp("Invalid " + field));
    }
    assert.equal(f.reads.length, 0);
    assert.equal(f.writes.length, 0);
  }
  const f = fixture();
  await assert.rejects(f.run({ scope: { owner: "bad/owner", repo: "repo" } }), /Invalid repository scope/);
});

test("validates every original author and message before creating any projection", async () => {
  const first = sha(20), tip = sha(21);
  for (const overrides of [{ author: null }, { author: { ...author, date: "invalid" } }, { message: null }, { committer: { date: "invalid" } }]) {
    const f = fixture([commit(first, [CHECKPOINT], overrides), commit(tip, [first])]);
    await assert.rejects(f.run({ sourceSha: tip }), /original source (author and date|commit message).*missing|invalid/);
    assert.equal(f.writes.length, 0);
  }
});

test("rejects malformed source parents and commit API responses", async () => {
  const source = sha(22);
  const f = fixture([commit(source, ["invalid"])]);
  await assert.rejects(f.run({ sourceSha: source }), /Invalid source first parent/);
  const bad = fixture([commit(CHECKPOINT, [EXCLUDED], { tree: { sha: "bad" } })]);
  await assert.rejects(bad.run(), /Invalid Git commit or tree/);
  assert.equal(f.writes.length + bad.writes.length, 0);
});

async function gitFixture(t) {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "vllm-fl-projection-test-"));
  t.after(async () => {
    const resolved = path.resolve(root);
    assert.equal(path.dirname(resolved), path.resolve(os.tmpdir()));
    assert.ok(path.basename(resolved).startsWith("vllm-fl-projection-test-"));
    await fs.rm(resolved, { recursive: true, force: true });
  });
  const env = { ...process.env };
  for (const key of Object.keys(env)) if (key.startsWith("GIT_")) delete env[key];
  Object.assign(env, {
    GIT_CONFIG_NOSYSTEM: "1", GIT_CONFIG_GLOBAL: process.platform === "win32" ? "NUL" : "/dev/null",
    GIT_AUTHOR_NAME: author.name, GIT_AUTHOR_EMAIL: author.email, GIT_AUTHOR_DATE: author.date,
    GIT_COMMITTER_NAME: "github-actions[bot]", GIT_COMMITTER_EMAIL: "41898282+github-actions[bot]@users.noreply.github.com",
    GIT_COMMITTER_DATE: "2026-11-03T16:00:00+08:00", GIT_DEFAULT_HASH: "sha1", GIT_TERMINAL_PROMPT: "0",
  });
  async function git(args, extraEnv = {}) {
    return (await exec("git", ["-c", "commit.gpgSign=false", "-c", "core.autocrlf=false",
      "-c", "core.hooksPath=" + (process.platform === "win32" ? "NUL" : "/dev/null"), ...args],
    { cwd: root, env: { ...env, ...extraEnv }, encoding: "utf8", windowsHide: true })).stdout.trim();
  }
  async function writeCommit(message, files) {
    for (const [file, contents] of Object.entries(files)) await fs.writeFile(path.join(root, file), contents);
    await git(["add", "."]);
    await git(["commit", "--quiet", "-m", message]);
    return git(["rev-parse", "HEAD"]);
  }
  async function ancestor(base, head) {
    try { await git(["merge-base", "--is-ancestor", base, head]); return true; }
    catch (error) { if (error.code === 1) return false; throw error; }
  }
  const github = { rest: { git: {
    async getCommit({ commit_sha }) {
      const values = (await git(["show", "--no-patch", "--format=%an%x00%ae%x00%aI%x00%B%x00%T%x00%P%x00%cI", commit_sha])).split("\0");
      return { data: { sha: commit_sha, author: { name: values[0], email: values[1], date: values[2] },
        message: values[3], tree: { sha: values[4] }, committer: { date: values[6] },
        parents: values[5] ? values[5].split(" ").map((parent) => ({ sha: parent })) : [] } };
    },
    async createCommit({ tree, parents, author: originalAuthor, committer, message }) {
      assert.equal(committer.name, "github-actions[bot]");
      assert.equal(committer.email, "41898282+github-actions[bot]@users.noreply.github.com");
      return { data: { sha: await git(["commit-tree", tree, ...parents.flatMap((parent) => ["-p", parent]), "-m", message],
        { GIT_AUTHOR_NAME: originalAuthor.name, GIT_AUTHOR_EMAIL: originalAuthor.email, GIT_AUTHOR_DATE: originalAuthor.date,
          GIT_COMMITTER_NAME: committer.name, GIT_COMMITTER_EMAIL: committer.email, GIT_COMMITTER_DATE: committer.date }) } };
    },
  } } };
  await git(["init", "--quiet", "--initial-branch=source"]);
  return { git, writeCommit, ancestor, github };
}

test("raw-tree bootstrap preserves excluded decisions and exposes a real native merge conflict", async (t) => {
  const f = await gitFixture(t);
  const common = await f.writeCommit("common", { "runtime.txt": "base runtime\n", "ci.txt": "old CI\n", "reviewed.txt": "base\n" });
  const excluded = await f.writeCommit("f6c1f92 excluded CI sync", { "ci.txt": "excluded CI stages\n" });
  const checkpoint = await f.writeCommit("reviewed source checkpoint", { "reviewed.txt": "reviewed source\n" });
  const source = await f.writeCommit("upgrade(musa): original full message\n\nOriginal explanation.",
    { "runtime.txt": "source vLLM 0.28\n", "new.txt": "new source delta\n" });
  await f.git(["checkout", "--quiet", "-b", "target", common]);
  const target = await f.writeCommit("target vLLM 0.32 and reviewed decisions",
    { "runtime.txt": "target vLLM 0.32\n", "ci.txt": "target CI retained\n", "reviewed.txt": "reviewed source\n" });
  const sourceTree = await f.git(["rev-parse", checkpoint + "^{tree}"]);
  const targetTree = await f.git(["rev-parse", target + "^{tree}"]);
  const projection = await f.git(["commit-tree", sourceTree, "-p", target, "-m", "Record raw source checkpoint without source parents"]);
  const bootstrap = await f.git(["commit-tree", targetTree, "-p", target, "-p", projection, "-m", "Acknowledge reviewed source and preserve target tree"]);
  const result = await project({ github: f.github, scope, sourceSha: source,
    sourceCheckpoint: checkpoint, projectionCheckpoint: projection });
  assert.deepEqual(await project({ github: f.github, scope, sourceSha: source,
    sourceCheckpoint: checkpoint, projectionCheckpoint: projection }), result);
  const incoming = result.projectionSha;
  assert.equal(await f.git(["rev-parse", bootstrap + "^{tree}"]), targetTree);
  assert.equal(await f.git(["merge-base", bootstrap, incoming]), projection);
  for (const head of [bootstrap, incoming]) {
    for (const original of [excluded, checkpoint, source]) assert.equal(await f.ancestor(original, head), false);
  }
  assert.equal(await f.git(["diff", projection, incoming]), await f.git(["diff", checkpoint, source]));
  assert.equal(await f.git(["show", incoming + ":runtime.txt"]), "source vLLM 0.28");
  assert.deepEqual((await f.github.rest.git.getCommit({ commit_sha: incoming })).data.author, author);
  let conflictOutput;
  try {
    await f.git(["merge-tree", "--write-tree", bootstrap, incoming]);
    assert.fail("Expected a genuine Git merge conflict between target 0.32 and source 0.28");
  } catch (error) {
    assert.equal(error.code, 1);
    conflictOutput = error.stdout;
  }
  assert.match(conflictOutput, /CONFLICT.*runtime\.txt/);
  const mergedTree = conflictOutput.split(/\r?\n/)[0];
  assert.match(mergedTree, /^[0-9a-f]{40}$/);
  assert.equal(await f.git(["show", mergedTree + ":ci.txt"]), "target CI retained");
  assert.equal(await f.git(["show", mergedTree + ":new.txt"]), "new source delta");
});
