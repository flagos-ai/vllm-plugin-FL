/*
 * Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const test = require("node:test");
const assert = require("node:assert/strict");
const forwardSync = require("../forward-sync.cjs");
const { SOURCE, TARGET, SYNC_PREFIX, BASE_PREFIX, MARKER, BASE_MARKER, STATE_PATH } = forwardSync.constants;
const SHA = letter => letter.repeat(40);
const S = SHA("a"), T = SHA("b"), P = SHA("c"), OLD = SHA("d"), R = SHA("e");
const BASE = SHA("f"), H = SHA("1"), WIP = SHA("2"), STAFF = SHA("3");
const TARGET_TREE = SHA("4"), SOURCE_TREE = SHA("5"), BASE_TREE = SHA("6"), INITIAL = SHA("7");
const repository = "flagos-ai/vllm-plugin-FL";
const syncBranch = SYNC_PREFIX + S;
const oldBranch = SYNC_PREFIX + OLD;
const baselineBranch = BASE_PREFIX + T;
const sourceMarker = sha => "<!-- forward-sync-source: " + sha + " -->";
const projectionMarker = sha => "<!-- forward-sync-projection: " + sha + " -->";
const queuedMarker = sha => "<!-- forward-sync-queued: " + sha + " -->";
const baseline = overrides => ({ version: 1, source: SOURCE, target: TARGET,
  sourceCheckpoint: OLD, projectionCheckpoint: P, ...overrides });

function pr(overrides = {}) {
  return {
    number: 12, draft: true, state: "open", merged_at: null,
    html_url: "https://github.com/" + repository + "/pull/12",
    body: [MARKER, sourceMarker(OLD), projectionMarker(P), "Maintainer notes: keep this."].join("\n"),
    head: { sha: P, ref: oldBranch, repo: { full_name: repository } },
    ...overrides,
  };
}

function mergedPr(overrides = {}) {
  return pr({ number: 11, draft: false, state: "closed", merged_at: "2026-11-03T04:00:00Z",
    body: MARKER + "\n" + sourceMarker(OLD),
    head: { sha: H, ref: oldBranch, repo: { full_name: repository } }, ...overrides });
}

function legacyPr(overrides = {}) {
  return pr({ body: MARKER + "\n" + sourceMarker(OLD) + "\n<!-- forward-sync-pending: " + S + " -->",
    head: { sha: WIP, ref: syncBranch, repo: { full_name: repository } }, ...overrides });
}

function baselinePr(overrides = {}) {
  return pr({ number: 13, draft: false, body: BASE_MARKER,
    head: { sha: BASE, ref: baselineBranch, repo: { full_name: repository } }, ...overrides });
}

function fixture(options = {}) {
  const refs = { ["heads/" + SOURCE]: options.source || S, ["heads/" + TARGET]: T,
    ...(options.refs || {}) };
  const prs = structuredClone(options.prs || [mergedPr()]);
  for (const item of prs) {
    if (item.state === "open" && item.head.repo?.full_name === repository) {
      refs["heads/" + item.head.ref] ??= item.head.sha;
    }
  }
  const comparisons = {
    [S + "..." + OLD]: "behind", [T + "..." + H]: "behind",
    [T + "..." + P]: "behind", [R + "..." + P]: "behind",
    [STAFF + "..." + P]: "behind", [BASE + "..." + T]: "behind",
    [T + "..." + INITIAL]: "behind",
    ...(options.comparisons || {}),
  };
  const contents = {};
  const initialState = Object.hasOwn(options, "state") ? options.state : baseline();
  if (initialState !== null) contents[T + ":" + STATE_PATH] = initialState;
  const pending = { version: 1, status: "needs-resolution", checkpointSha: OLD,
    pendingSha: S, ...(options.pending || {}) };
  contents[WIP + ":.github/forward-sync-pending.json"] = pending;
  if (options.baselineRef) {
    refs["heads/" + baselineBranch] = BASE;
    contents[BASE + ":" + STATE_PATH] = baseline();
  }
  const commits = {
    [T]: { sha: T, tree: { sha: TARGET_TREE }, message: "Target adaptation", parents: [] },
    [OLD]: { sha: OLD, tree: { sha: SOURCE_TREE }, message: "Reviewed source checkpoint", parents: [] },
    [WIP]: { sha: WIP, tree: { sha: SOURCE_TREE },
      message: "WIP: resolve MUSA forward-sync conflicts", parents: [{ sha: T }] },
    ...(options.commits || {}),
  };
  const calls = [], warnings = [], outputs = {}, errors = { ...(options.errors || {}) };
  const fail = status => Object.assign(new Error("GitHub API failure"), { status });
  const method = (name, fn) => async args => {
    calls.push({ name, args: structuredClone(args) });
    if (errors[name]) throw fail(errors[name]);
    return { data: structuredClone(fn(args)) };
  };
  let stateInTree;
  const github = {
    rest: {
      git: {
        getRef: method("getRef", ({ ref }) => {
          if (!refs[ref]) throw fail(404);
          return { object: { sha: refs[ref] } };
        }),
        getCommit: method("getCommit", ({ commit_sha }) => {
          assert.ok(commits[commit_sha], "Missing commit fixture: " + commit_sha);
          return commits[commit_sha];
        }),
        createCommit: method("createCommit", args => {
          const sha = args.tree === SOURCE_TREE ? P : BASE;
          commits[sha] = { sha, tree: { sha: args.tree }, message: args.message,
            parents: args.parents.map(parent => ({ sha: parent })) };
          if (args.tree === BASE_TREE) contents[sha + ":" + STATE_PATH] = stateInTree;
          return commits[sha];
        }),
        createTree: method("createTree", args => {
          assert.equal(args.tree.length, 1, "Bootstrap must change only its state file");
          assert.equal(args.tree[0].path, STATE_PATH);
          stateInTree = JSON.parse(args.tree[0].content);
          return { sha: BASE_TREE };
        }),
        createRef: method("createRef", ({ ref, sha }) => {
          assert.notEqual(ref, "refs/heads/" + TARGET, "The target is updated only through review");
          assert.equal(ref.startsWith("refs/heads/"), true);
          const shortRef = ref.slice("refs/".length);
          assert.equal(refs[shortRef], undefined, "Existing branches must never be recreated");
          refs[shortRef] = sha;
          return { ref, object: { sha } };
        }),
        updateRef: method("updateRef", ({ ref, sha, force }) => {
          assert.notEqual(ref, "heads/" + TARGET);
          assert.equal(force, false, "Source updates must never force-push over reviewer work");
          refs[ref] = sha;
          return { ref, object: { sha } };
        }),
      },
      repos: {
        compareCommitsWithBasehead: method("compare", ({ basehead }) => {
          assert.ok(basehead in comparisons, "Missing ancestry fixture: " + basehead);
          return { status: comparisons[basehead] };
        }),
        getContent: method("getContent", ({ path, ref }) => {
          const key = ref + ":" + path;
          if (!(key in contents)) throw fail(404);
          return { encoding: "base64", content: Buffer.from(JSON.stringify(contents[key])).toString("base64") };
        }),
        merge: method("merge", () => assert.fail("GitHub must calculate conflicts without an automatic merge")),
      },
      pulls: {
        list: method("listPRs", args => {
          assert.equal(args.base, TARGET);
          assert.equal(args.state, "all");
          return prs;
        }),
        create: method("createPR", args => {
          const created = pr({ number: 20, draft: args.draft, body: args.body,
            html_url: "https://github.com/" + repository + "/pull/20",
            head: { sha: refs["heads/" + args.head], ref: args.head, repo: { full_name: repository } } });
          prs.unshift(created);
          return created;
        }),
        get: method("getPR", ({ pull_number }) => {
          const found = prs.find(item => item.number === pull_number);
          assert.ok(found);
          if (options.legacyMovedOnRead && found.number === 12) {
            found.head.sha = STAFF;
            refs["heads/" + found.head.ref] = STAFF;
          }
          return found;
        }),
        update: method("updatePR", ({ pull_number, ...changes }) => {
          const found = prs.find(item => item.number === pull_number);
          assert.ok(found);
          for (const key of ["body", "state", "draft"]) {
            if (Object.hasOwn(changes, key)) found[key] = changes[key];
          }
          return found;
        }),
      },
    },
    paginate: async (fn, args) => (await fn(args)).data,
  };
  const core = {
    info() {}, warning(message) { warnings.push(message); },
    setOutput(name, value) { outputs[name] = value; },
  };
  const project = async args => {
    calls.push({ name: "project", args });
    if (errors.project) throw fail(errors.project);
    return { projectionSha: options.projectionSha || R, commits: args.sourceSha === args.sourceCheckpoint ? [] : [{
      sourceSha: args.sourceSha, projectionSha: options.projectionSha || R,
      author: { name: "Physics", email: "physics@example.com", date: "2026-11-03T07:44:02Z" },
      message: "upgrade(musa): adapt MUSA to vLLM 0.28.0 (#575)\n\nOriginal source message.",
    }] };
  };
  const context = { repo: { owner: "flagos-ai", repo: "vllm-plugin-FL" }, payload: options.payload || {} };
  return {
    run: () => forwardSync({ github, context, core, project }),
    calls, refs, prs, errors, outputs, warnings, github, comparisons, contents,
    projects: () => calls.filter(call => call.name === "project"),
    writes: () => calls.filter(call => ["createCommit", "createTree", "createRef", "updateRef", "createPR", "updatePR"].includes(call.name)),
  };
}

const firstProject = f => f.projects()[0].args;
const writeNames = f => f.writes().map(call => call.name);

test("initialization reviews only comparison ancestry before publishing incoming source changes", async () => {
  const f = fixture({ state: null });
  assert.equal((await f.run()).status, "baseline-pr-required");
  assert.deepEqual(f.projects(), []);
  const created = f.calls.filter(call => call.name === "createCommit");
  assert.deepEqual(created[0].args.parents, [T]);
  assert.equal(created[0].args.tree, SOURCE_TREE);
  assert.deepEqual(created[1].args.parents, [T, P]);
  assert.equal(created[1].args.tree, BASE_TREE);
  const tree = f.calls.find(call => call.name === "createTree").args;
  assert.equal(tree.base_tree, TARGET_TREE);
  assert.deepEqual(JSON.parse(tree.tree[0].content), baseline());
  assert.equal(f.calls.some(call => call.name === "getCommit" && call.args.commit_sha === S), false);
  const createdPr = f.calls.find(call => call.name === "createPR").args;
  assert.equal(createdPr.base, TARGET);
  assert.equal(createdPr.head, baselineBranch);
  assert.ok(createdPr.body.includes(BASE_MARKER));
  assert.match(createdPr.body, /historical snapshot contains the source-side context/);
  assert.match(createdPr.body, /Squash\/rebase would lose/);
  assert.equal(f.refs["heads/" + TARGET], T);
  assert.equal(f.refs["heads/" + syncBranch], undefined);
});

test("a waiting bootstrap PR is reused without creating a normal sync PR", async () => {
  const f = fixture({ state: null, prs: [baselinePr(), mergedPr()] });
  const result = await f.run();
  assert.equal(result.status, "baseline-pr-required");
  assert.equal(result.pullRequest, 13);
  assert.deepEqual(f.writes(), []);
  assert.deepEqual(f.projects(), []);
});

test("bootstrap PR creation retries reuse the published ancestry and state", async () => {
  const f = fixture({ state: null, errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + baselineBranch], BASE);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "baseline-pr-required");
  assert.equal(f.calls.filter(call => call.name === "createCommit").length, 2);
  assert.equal(f.calls.filter(call => call.name === "createRef").length, 1);
  assert.deepEqual(f.projects(), []);
});

test("invalid baseline state and squash-lost comparison ancestry stop before writes", async () => {
  const invalid = fixture({ state: baseline({ target: "main" }) });
  await assert.rejects(invalid.run(), /Invalid source comparison baseline/);
  assert.deepEqual(invalid.writes(), []);
  const lost = fixture({ comparisons: { [T + "..." + P]: "diverged" } });
  await assert.rejects(lost.run(), /not an ancestor of target/);
  assert.deepEqual(lost.writes(), []);
  assert.deepEqual(lost.projects(), []);
});

test("an unchanged reviewed source checkpoint needs no new projection or PR", async () => {
  const f = fixture({ source: OLD });
  assert.equal((await f.run()).status, "already-processed");
  assert.deepEqual(f.projects(), []);
  assert.deepEqual(f.writes(), []);
});

test("a new PR projects only the increment and names the original source author", async () => {
  const f = fixture();
  const result = await f.run();
  assert.equal(result.status, "pr-created");
  const args = firstProject(f);
  assert.equal(args.github, f.github);
  assert.deepEqual(args.scope, { owner: "flagos-ai", repo: "vllm-plugin-FL" });
  assert.equal(args.sourceSha, S);
  assert.equal(args.sourceCheckpoint, OLD);
  assert.equal(args.projectionCheckpoint, P);
  assert.deepEqual(writeNames(f), ["createRef", "createPR"]);
  const created = f.calls.find(call => call.name === "createPR").args;
  assert.equal(created.base, TARGET);
  assert.equal(created.draft, true);
  assert.ok(created.body.includes(sourceMarker(S)));
  assert.ok(created.body.includes(projectionMarker(R)));
  assert.ok(created.body.includes("/commit/" + S));
  assert.match(created.body, /upgrade\(musa\).* — Physics/);
  assert.match(created.body, /native conflicts/);
  assert.doesNotMatch(created.body, /resolve the committed conflict markers/i);
  assert.equal(f.refs["heads/" + syncBranch], R);
  assert.equal(f.refs["heads/" + TARGET], T);
});

test("an untouched open projection extends with a non-force update and preserves notes", async () => {
  const f = fixture({ prs: [pr(), mergedPr()] });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(firstProject(f).sourceCheckpoint, OLD);
  assert.equal(firstProject(f).projectionCheckpoint, P);
  assert.deepEqual(writeNames(f), ["updateRef", "updatePR"]);
  assert.deepEqual(f.calls.find(call => call.name === "updateRef").args,
    { owner: "flagos-ai", repo: "vllm-plugin-FL", ref: "heads/" + oldBranch, sha: R, force: false });
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
  assert.ok(f.prs[0].body.includes(projectionMarker(R)));
  assert.match(f.prs[0].body, /Maintainer notes: keep this/);
  assert.equal(f.prs[0].draft, true);
});

test("maintainer resolutions freeze the branch and queue source changes without advancing checkpoints", async () => {
  const active = pr({ head: { sha: STAFF, ref: oldBranch, repo: { full_name: repository } } });
  const f = fixture({ prs: [active, mergedPr()] });
  assert.equal((await f.run()).status, "pr-awaiting-review");
  assert.deepEqual(f.projects(), []);
  assert.deepEqual(writeNames(f), ["updatePR"]);
  assert.equal(f.refs["heads/" + oldBranch], STAFF);
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  assert.ok(f.prs[0].body.includes(projectionMarker(P)));
  assert.ok(f.prs[0].body.includes(queuedMarker(S)));
  assert.match(f.prs[0].body, /Maintainer notes: keep this/);
});

test("the next cycle extends the source projection rather than a merged resolution head", async () => {
  const completed = mergedPr({ body: [MARKER, sourceMarker(OLD), projectionMarker(P)].join("\n") });
  const f = fixture({ state: baseline({ projectionCheckpoint: INITIAL }), prs: [completed] });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstProject(f).sourceCheckpoint, OLD);
  assert.equal(firstProject(f).projectionCheckpoint, P);
  assert.notEqual(firstProject(f).projectionCheckpoint, H);
  assert.ok(f.calls.some(call => call.name === "compare" && call.args.basehead === T + "..." + H));
});

test("an unchanged native PR retry leaves its ref, body, and draft state intact", async () => {
  const active = pr({ body: [MARKER, sourceMarker(S), projectionMarker(R), "Maintainer notes: keep this."].join("\n"),
    head: { sha: R, ref: syncBranch, repo: { full_name: repository } } });
  const f = fixture({ prs: [active, mergedPr()] });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(firstProject(f).sourceCheckpoint, S);
  assert.equal(firstProject(f).projectionCheckpoint, R);
  assert.deepEqual(f.writes(), []);
  assert.equal(f.prs[0].draft, true);
});

test("legacy WIP replacement creates a separate native branch before closing the old draft", async () => {
  const f = fixture({ prs: [legacyPr(), mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  const create = f.calls.find(call => call.name === "createPR");
  assert.notEqual(create.args.head, syncBranch);
  assert.equal(f.refs["heads/" + syncBranch], WIP);
  assert.equal(f.refs["heads/" + create.args.head], R);
  assert.deepEqual(writeNames(f), ["createRef", "createPR", "updatePR"]);
  const close = f.calls.find(call => call.name === "updatePR");
  assert.equal(close.args.pull_number, 12);
  assert.equal(close.args.state, "closed");
  assert.ok(f.calls.indexOf(create) < f.calls.indexOf(close));
  assert.equal(f.prs.find(item => item.number === 12).state, "closed");
  assert.match(create.args.body, /Replaces legacy WIP draft #12/);
});

test("a failed native PR creation leaves the legacy draft and its branch untouched", async () => {
  const f = fixture({ prs: [legacyPr(), mergedPr()], errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + syncBranch], WIP);
  assert.equal(f.prs.find(item => item.number === 12).state, "open");
  assert.equal(f.calls.some(call => call.name === "updatePR"), false);
  assert.equal(f.calls.some(call => call.name === "updateRef"), false);
});

test("a legacy draft changed during replacement is preserved instead of closed", async () => {
  const f = fixture({ prs: [legacyPr(), mergedPr()], legacyMovedOnRead: true });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.prs.find(item => item.number === 12).state, "open");
  assert.equal(f.refs["heads/" + syncBranch], STAFF);
  assert.equal(f.calls.some(call => call.name === "updatePR"), false);
  assert.equal(f.warnings.length, 1);
});

test("legacy replacement requires the original draft, pending manifest, and WIP commit", async () => {
  for (const options of [
    { prs: [legacyPr({ draft: false }), mergedPr()] },
    { prs: [legacyPr(), mergedPr()], pending: { checkpointSha: S } },
    { prs: [legacyPr(), mergedPr()], commits: { [WIP]: {
      sha: WIP, tree: { sha: SOURCE_TREE }, message: "Maintainer resolution", parents: [{ sha: T }],
    } } },
  ]) {
    const f = fixture(options);
    await assert.rejects(f.run(), /legacy draft contains maintainer work/);
    assert.deepEqual(f.projects(), []);
    assert.deepEqual(f.writes(), []);
  }
});

test("force-push and source deletion events stop before repository access", async () => {
  const forced = fixture({ payload: { ref: "refs/heads/" + SOURCE, forced: true } });
  await assert.rejects(forced.run(), /rewrote history/);
  assert.deepEqual(forced.calls, []);
  const deleted = fixture({ payload: { ref: "refs/heads/" + SOURCE, deleted: true } });
  assert.equal((await deleted.run()).status, "source-deleted");
  assert.deepEqual(deleted.calls, []);
});

test("a source rewrite after the reviewed checkpoint cannot publish a new projection", async () => {
  const f = fixture({ comparisons: { [S + "..." + OLD]: "diverged" } });
  await assert.rejects(f.run(), /Source history was rewritten/);
  assert.deepEqual(f.projects(), []);
  assert.deepEqual(f.writes(), []);
});

test("an open PR with a rewritten checkpoint or moved ref is protected", async () => {
  const rewritten = fixture({ prs: [pr({
    body: [MARKER, sourceMarker(INITIAL), projectionMarker(P)].join("\n"),
  }), mergedPr()], comparisons: { [S + "..." + INITIAL]: "diverged" } });
  await assert.rejects(rewritten.run(), /rewrote the open PR checkpoint/);
  assert.deepEqual(rewritten.writes(), []);
  const moved = fixture({ prs: [pr(), mergedPr()], refs: { ["heads/" + oldBranch]: STAFF } });
  await assert.rejects(moved.run(), /branch moved while reading/);
  assert.deepEqual(moved.writes(), []);
});

test("a missing merged PR head in target cannot be hidden by comparison state", async () => {
  const f = fixture({ comparisons: { [T + "..." + H]: "diverged" } });
  await assert.rejects(f.run(), /merged sync head is missing from target/);
  assert.deepEqual(f.projects(), []);
  assert.deepEqual(f.writes(), []);
});

test("independent work on an existing destination branch is never force-pushed", async () => {
  const f = fixture({ refs: { ["heads/" + syncBranch]: STAFF },
    comparisons: { [R + "..." + STAFF]: "diverged" } });
  await assert.rejects(f.run(), /independent work; no force push/);
  assert.equal(f.refs["heads/" + syncBranch], STAFF);
  assert.deepEqual(f.writes(), []);
});

test("multiple managed open PRs or missing reviewed checkpoints fail without writes", async () => {
  const duplicate = fixture({ prs: [pr(), pr({ number: 13 }), mergedPr()] });
  await assert.rejects(duplicate.run(), /Multiple open forward-sync/);
  assert.deepEqual(duplicate.writes(), []);
  const missing = fixture({ prs: [mergedPr({ body: MARKER })] });
  await assert.rejects(missing.run(), /no unique checkpoint marker/);
  assert.deepEqual(missing.writes(), []);
});
