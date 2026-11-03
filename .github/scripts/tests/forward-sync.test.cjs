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

const test = require("node:test");
const assert = require("node:assert/strict");
const forwardSync = require("../forward-sync.cjs");
const { SOURCE, TARGET, SYNC_PREFIX, MARKER } = forwardSync.constants;
const SHA = (letter) => letter.repeat(40);
const S = SHA("a"), T = SHA("b"), H = SHA("c"), OLD = SHA("d"), MERGED = SHA("e");
const RETRY = SHA("f");
const SYNC = SYNC_PREFIX + S, OLD_BRANCH = SYNC_PREFIX + OLD;
const repository = "flagos-ai/vllm-plugin-FL";
const sourceMarker = (sha) => "<!-- forward-sync-source: " + sha + " -->";
const pendingMarker = (sha) => "<!-- forward-sync-pending: " + sha + " -->";

function pr(overrides = {}) {
  return {
    number: 12, state: "open", merged_at: null,
    html_url: "https://github.com/" + repository + "/pull/12",
    body: MARKER + "\n" + sourceMarker(OLD) + "\nMaintainer notes: preserve this.",
    head: { sha: H, ref: SYNC, repo: { full_name: repository } }, ...overrides,
  };
}

function mergedPr(overrides = {}) {
  return pr({
    state: "closed", merged_at: "2026-10-19T00:00:00Z",
    head: { sha: H, ref: OLD_BRANCH, repo: { full_name: repository } },
    ...overrides,
  });
}

function fixture(options = {}) {
  const refs = {
    ["heads/" + SOURCE]: options.source || S, ["heads/" + TARGET]: T,
    ...(options.sync ? { ["heads/" + SYNC]: options.sync } : {}),
    ...(options.oldTip ? { ["heads/" + OLD_BRANCH]: options.oldTip } : {}),
    ...(options.refs || {}),
  };
  const prs = structuredClone(options.prs || []);
  // These compare statuses validate source history and reviewed target ancestry.
  // A replayed source SHA is intentionally not an ancestor of its cherry-pick.
  const comparisons = {
    [T + "..." + S]: "diverged", [T + "..." + OLD]: "diverged",
    [S + "..." + OLD]: "behind", [T + "..." + H]: "behind",
    [MERGED + "..." + H]: "behind", [MERGED + "..." + T]: "behind",
    [H + "..." + T]: "behind", [RETRY + "..." + H]: "behind",
    [H + "..." + OLD]: "diverged", [RETRY + "..." + OLD]: "diverged",
    [MERGED + "..." + OLD]: "diverged", [MERGED + "..." + S]: "diverged",
    [RETRY + "..." + T]: "behind",
    ...(options.comparisons || {}),
  };
  const errors = { ...(options.errors || {}) };
  const state = { conflict: Boolean(options.conflict), replayChanged: options.replayChanged };
  const calls = [], warnings = [], failures = [], outputs = {};
  const fail = (status, message = "GitHub API failure") => Object.assign(new Error(message), { status });
  const method = (name, fn) => async (args) => {
    calls.push({ name, args });
    if (errors[name]) throw fail(errors[name]);
    return { data: fn(args) };
  };
  const forbidden = (name) => method(name, () => {
    assert.fail(name + " must not publish original source ancestry or create snapshot refs");
  });
  const github = {
    rest: {
      git: {
        getCommit: forbidden("getCommit"),
        createCommit: forbidden("createCommit"),
        createRef: forbidden("createRef"),
        updateRef: forbidden("updateRef"),
        getRef: method("getRef", ({ ref }) => {
          if (!refs[ref]) throw fail(404);
          return { object: { sha: refs[ref] } };
        }),
      },
      repos: {
        compareCommitsWithBasehead: method("compare", ({ basehead }) => {
          assert.ok(basehead in comparisons, "Missing comparison fixture: " + basehead);
          return { status: comparisons[basehead] };
        }),
        merge: forbidden("merge"),
      },
      pulls: {
        get: method("getPR", ({ pull_number }) => {
          const found = prs.find((item) => item.number === pull_number);
          if (options.closeOnRead) found.state = "closed";
          return found;
        }),
        list: method("listPRs", ({ base, head, state: requestedState }) => {
          assert.equal(base, TARGET);
          assert.equal(head, undefined);
          assert.equal(requestedState, "all");
          return prs;
        }),
        create: method("createPR", (args) => {
          const created = pr({
            number: 20, html_url: "https://github.com/" + repository + "/pull/20",
            body: args.body, head: { sha: refs["heads/" + args.head], ref: args.head,
              repo: { full_name: repository } },
          });
          prs.unshift(created);
          return created;
        }),
        update: method("updatePR", ({ pull_number, body }) => {
          const found = prs.find((item) => item.number === pull_number);
          found.body = body;
          return found;
        }),
      },
    },
    paginate: async (fn, args) => (await fn(args)).data,
  };
  const core = {
    info() {},
    warning(message) { warnings.push(message); },
    setFailed(message) { failures.push(message); },
    setOutput(name, value) { outputs[name] = value; },
  };
  const context = { repo: { owner: "flagos-ai", repo: "vllm-plugin-FL" }, payload: options.payload || {} };
  const replay = async (args) => {
    calls.push({ name: "replay", args });
    assert.equal(args.repository, repository);
    assert.equal(args.core, core);
    if (errors.replay) throw fail(errors.replay, "Replay failed");
    if (state.conflict) {
      return {
        sha: args.baseSha, checkpointSha: args.checkpointSha,
        pending: args.sourceSha, conflicted: true, changed: false,
      };
    }
    const changed = state.replayChanged ?? args.baseSha !== MERGED;
    if (changed) {
      // The helper owns Git writes. The GitHub orchestration layer never
      // creates a commit with the original source/checkpoint SHA as a parent.
      refs["heads/" + args.syncBranch] = MERGED;
      calls.push({ name: "replayPush", args: { branch: args.syncBranch, sha: MERGED } });
    }
    return {
      sha: changed ? MERGED : args.baseSha, checkpointSha: args.sourceSha,
      conflicted: false, changed,
    };
  };
  return {
    run: () => forwardSync({ github, context, core, replay }),
    calls, refs, prs, warnings, failures, outputs, errors, comparisons, state,
    replays: () => calls.filter(({ name }) => name === "replay"),
    mutations: () => calls.filter(({ name }) => ["replayPush", "createPR", "updatePR"].includes(name)),
  };
}

const firstReplay = (f) => f.replays()[0].args;
const mutationNames = (f) => f.mutations().map(({ name }) => name);

function assertNoSourceAncestryWrites(f) {
  assert.deepEqual(f.calls.filter(({ name }) => [
    "getCommit", "createCommit", "createRef", "updateRef", "merge",
  ].includes(name)), []);
}

test("the first sync replays onto target instead of creating a source snapshot", async () => {
  const f = fixture();
  const result = await f.run();
  assert.equal(result.status, "pr-created");
  assert.equal(result.pullRequestUrl, f.prs[0].html_url);
  assert.deepEqual(mutationNames(f), ["replayPush", "createPR"]);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(firstReplay(f).sourceSha, S);
  assert.equal(firstReplay(f).checkpointSha, null);
  assert.equal(firstReplay(f).syncBranch, SYNC);
  assert.equal(firstReplay(f).existingBranch, false);
  const created = f.calls.find(({ name }) => name === "createPR").args;
  assert.equal(created.base, TARGET);
  assert.equal(created.head, SYNC);
  assert.match(created.body, /Automatic merging is disabled/);
  assert.ok(created.body.includes(sourceMarker(S)));
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.notEqual(f.refs["heads/" + SYNC], S);
  assert.equal(f.refs["heads/" + TARGET], T);
  assertNoSourceAncestryWrites(f);
});

test("the replay helper receives its token without GitHub commit-parent writes", async () => {
  const oldToken = process.env.FORWARD_SYNC_TOKEN;
  process.env.FORWARD_SYNC_TOKEN = "fixture-token";
  try {
    const f = fixture();
    await f.run();
    assert.equal(firstReplay(f).token, "fixture-token");
    assert.equal(firstReplay(f).sourceSha, S);
    assertNoSourceAncestryWrites(f);
  } finally {
    if (oldToken === undefined) delete process.env.FORWARD_SYNC_TOKEN;
    else process.env.FORWARD_SYNC_TOKEN = oldToken;
  }
});

test("equivalent patches require no new branch or empty PR", async () => {
  const f = fixture({ replayChanged: false });
  assert.equal((await f.run()).status, "already-applied");
  assert.equal(firstReplay(f).baseSha, T);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.refs["heads/" + SYNC], undefined);
  assert.equal(f.prs.length, 0);
});

test("an unchanged open retry reuses the PR without writes", async () => {
  const f = fixture({
    sync: MERGED, replayChanged: false,
    prs: [pr({ body: MARKER + "\n" + sourceMarker(S) })],
  });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(firstReplay(f).baseSha, MERGED);
  assert.equal(firstReplay(f).checkpointSha, S);
  assert.equal(firstReplay(f).existingBranch, true);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.calls.filter(({ name }) => name === "getPR").length, 1);
});

test("source already included in target requires no replay, branch or PR writes", async () => {
  const f = fixture({ comparisons: { [T + "..." + S]: "behind" } });
  assert.equal((await f.run()).status, "already-included");
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
  assert.equal(f.calls.some(({ name }) => name === "listPRs"), false);
});

test("human conflict-resolution commits are preserved as the replay base", async () => {
  const f = fixture({ sync: H, prs: [pr()] });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(firstReplay(f).baseSha, H);
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assert.equal(firstReplay(f).existingBranch, true);
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.match(f.prs[0].body, /Maintainer notes: preserve this/);
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
  assertNoSourceAncestryWrites(f);
});

test("an open replay conflict preserves its branch, checkpoint and maintainer notes", async () => {
  const f = fixture({ sync: H, prs: [pr()], conflict: true });
  const result = await f.run();
  assert.equal(result.status, "conflict-needs-resolution");
  assert.equal(result.pullRequest, 12);
  assert.equal(f.refs["heads/" + SYNC], H);
  assert.equal(f.prs.length, 1);
  assert.equal(f.failures.length, 1);
  assert.ok(f.failures[0].includes(f.prs[0].html_url));
  assert.deepEqual(mutationNames(f), ["updatePR"]);
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  assert.ok(!f.prs[0].body.includes(sourceMarker(S)));
  assert.ok(f.prs[0].body.includes(pendingMarker(S)));
  assert.match(f.prs[0].body, /Maintainer notes: preserve this/);
});

test("an open PR advances its marker even when every new patch is equivalent", async () => {
  const f = fixture({
    sync: H, replayChanged: false,
    prs: [pr({ body: pr().body + "\n" + pendingMarker(S) })],
  });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(f.refs["heads/" + SYNC], H);
  assert.deepEqual(mutationNames(f), ["updatePR"]);
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
  assert.ok(!f.prs[0].body.includes(pendingMarker(S)));
  assert.match(f.prs[0].body, /Maintainer notes: preserve this/);
});

test("a closed, unmerged PR is not silently reopened or overwritten", async () => {
  const f = fixture({ sync: H, prs: [pr({ state: "closed" })] });
  await assert.rejects(f.run(), /closed without merging/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("an unrelated existing branch is protected", async () => {
  const f = fixture({ sync: H, comparisons: { [H + "..." + T]: "diverged" } });
  await assert.rejects(f.run(), /reviewed target/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("an unmarked PR is protected", async () => {
  const f = fixture({ sync: H, prs: [pr({ body: "Someone else's PR" })] });
  await assert.rejects(f.run(), /unrecognized PR/);
  assert.deepEqual(f.mutations(), []);
});

test("a merged PR with lost ancestry stops before replaying squash-merged history", async () => {
  const f = fixture({
    oldTip: H, prs: [mergedPr()], comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("a new cycle replays only the increment from the reviewed source checkpoint", async () => {
  const f = fixture({ oldTip: H, prs: [mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(mutationNames(f), ["replayPush", "createPR"]);
  assert.equal(firstReplay(f).sourceSha, S);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assert.equal(firstReplay(f).existingBranch, false);
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
  assert.equal(f.refs["heads/" + TARGET], T);
  assert.equal(f.outputs.reviewed_source_sha, OLD);
  assertNoSourceAncestryWrites(f);
});

test("a deleted merged branch starts a new increment after historical-head validation", async () => {
  const f = fixture({ prs: [mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(f.refs["heads/" + OLD_BRANCH], undefined);
  assertNoSourceAncestryWrites(f);
});

test("source history rewrite detected from the prior checkpoint stops all mutations", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [S + "..." + OLD]: "diverged" },
  });
  await assert.rejects(f.run(), /rewound or rewritten/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("a force-push event fails before accessing any repository refs", async () => {
  const f = fixture({ payload: { ref: "refs/heads/" + SOURCE, forced: true } });
  await assert.rejects(f.run(), /rewrote history/);
  assert.deepEqual(f.calls, []);
});

test("source deletion is a no-op", async () => {
  const f = fixture({ payload: { ref: "refs/heads/" + SOURCE, deleted: true } });
  assert.equal((await f.run()).status, "source-deleted");
  assert.deepEqual(f.calls, []);
});

test("permission errors are propagated instead of mistaken for a missing ref", async () => {
  const f = fixture({ errors: { getRef: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("PR creation permission failure retries the pushed branch without duplicating patches", async () => {
  const f = fixture({ errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.replays().length, 2);
  assert.equal(f.replays()[1].args.baseSha, MERGED);
  assert.equal(f.replays()[1].args.existingBranch, true);
  assert.equal(f.calls.filter(({ name }) => name === "replayPush").length, 1);
  assert.equal(f.prs.length, 1);
  assertNoSourceAncestryWrites(f);
});

test("missing checkpoint and duplicate open PRs fail without writes", async () => {
  const missing = fixture({ sync: H, prs: [pr({ body: MARKER })] });
  await assert.rejects(missing.run(), /no unique source SHA marker/);
  assert.deepEqual(missing.mutations(), []);
  const duplicate = fixture({ sync: H, prs: [pr(), pr({ number: 13 })] });
  await assert.rejects(duplicate.run(), /Multiple open/);
  assert.deepEqual(duplicate.mutations(), []);
});

test("unexpected replay failures propagate without a replacement PR", async () => {
  const f = fixture({ sync: H, prs: [pr()], errors: { replay: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], H);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.calls.some(({ name }) => name === "createPR"), false);
});

test("a new-cycle PR creation failure reuses the reviewed branch without duplicate replay", async () => {
  const f = fixture({ oldTip: H, prs: [mergedPr()], errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
  assert.equal(f.replays()[1].args.baseSha, MERGED);
  assert.equal(f.replays()[1].args.checkpointSha, OLD);
  assert.equal(f.calls.filter(({ name }) => name === "replayPush").length, 1);
  assertNoSourceAncestryWrites(f);
});

test("manual commits added after a merged PR are not silently abandoned", async () => {
  const f = fixture({
    oldTip: MERGED, prs: [mergedPr()], comparisons: { [T + "..." + MERGED]: "diverged" },
  });
  await assert.rejects(f.run(), /reconcile those manual commits/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("resetting an old branch cannot hide a squash merge's missing PR ancestry", async () => {
  const f = fixture({
    oldTip: T, prs: [mergedPr()], comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
});

test("a fork PR with a matching branch prefix cannot impersonate the managed PR", async () => {
  const f = fixture({ prs: [pr({
    body: "Untrusted fork PR",
    head: { sha: H, ref: SYNC, repo: { full_name: "someone/vllm-plugin-FL" } },
  })] });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).baseSha, T);
});

test("an open PR with a missing branch stops for manual recovery", async () => {
  const f = fixture({ prs: [pr()] });
  await assert.rejects(f.run(), /branch is missing/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("a PR closed during replay fails visibly instead of reporting success", async () => {
  const f = fixture({ sync: H, prs: [pr()], closeOnRead: true });
  await assert.rejects(f.run(), (error) =>
    /closed while its branch was being updated/.test(error.message) &&
    error.message.includes(f.prs[0].html_url));
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.equal(f.outputs.status, undefined);
  assert.equal(f.calls.some(({ name }) => name === "createPR"), false);
});

test("an unchanged reviewed checkpoint needs no PR despite excluded source ancestry", async () => {
  const f = fixture({ source: OLD, oldTip: H, prs: [mergedPr()] });
  assert.equal((await f.run()).status, "already-processed");
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
  assertNoSourceAncestryWrites(f);
});

test("a reviewed checkpoint never becomes a commit parent even if absent from target ancestry", async () => {
  const f = fixture({ prs: [mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(f.calls.some(({ name, args }) => name === "compare" && args.basehead === T + "..." + OLD), false);
  assertNoSourceAncestryWrites(f);
});

test("a conflicting new increment creates neither a snapshot branch nor an empty PR", async () => {
  const f = fixture({ prs: [mergedPr()], conflict: true });
  const result = await f.run();
  assert.equal(result.status, "conflict-needs-resolution");
  assert.equal(result.pullRequest, null);
  assert.equal(result.pullRequestUrl, null);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assert.equal(f.refs["heads/" + SYNC], undefined);
  assert.equal(f.prs.length, 1);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.outputs.pull_request_number, undefined);
  assert.match(f.failures[0], /No PR was created before resolution/);
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  assertNoSourceAncestryWrites(f);
});

test("a first-cycle conflict also avoids publishing original source history", async () => {
  const f = fixture({ conflict: true });
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  assert.equal(firstReplay(f).checkpointSha, null);
  assert.equal(f.refs["heads/" + SYNC], undefined);
  assert.deepEqual(f.mutations(), []);
  assertNoSourceAncestryWrites(f);
});

test("a new increment retries a conflict from target without duplicate branch writes", async () => {
  const f = fixture({ prs: [mergedPr()], conflict: true });
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  f.state.conflict = false;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.replays().length, 2);
  assert.ok(f.replays().every(({ args }) => args.baseSha === T && !args.existingBranch));
  assert.equal(f.calls.filter(({ name }) => name === "replayPush").length, 1);
  assert.equal(f.refs["heads/" + SYNC], MERGED);
});

test("an open sync still validates the previous merged head in target", async () => {
  const f = fixture({ sync: H, prs: [pr(), mergedPr({ number: 11 })] });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(f.outputs.reviewed_source_sha, OLD);
  assert.ok(f.calls.some(({ name, args }) => name === "compare" && args.basehead === T + "..." + H));
  assertNoSourceAncestryWrites(f);
});

test("an open PR cannot hide lost ancestry from the last merged sync", async () => {
  const f = fixture({
    sync: H, prs: [pr(), mergedPr({ number: 11 })],
    comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("an open PR cannot replace a missing merged checkpoint with an observed SHA", async () => {
  const f = fixture({ sync: H, prs: [pr(), mergedPr({ number: 11, body: MARKER })] });
  await assert.rejects(f.run(), /no unique source SHA marker/);
  assert.deepEqual(f.mutations(), []);
});

test("the most recently merged checkpoint wins over PR creation order", async () => {
  const reviewed = SHA("2"), reviewedHead = SHA("3");
  const f = fixture({
    prs: [mergedPr(), mergedPr({
      number: 11, merged_at: "2026-10-20T00:00:00Z",
      body: MARKER + "\n" + sourceMarker(reviewed),
      head: { sha: reviewedHead, ref: SYNC_PREFIX + reviewed, repo: { full_name: repository } },
    })],
    comparisons: { [S + "..." + reviewed]: "behind", [T + "..." + reviewedHead]: "behind" },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).checkpointSha, reviewed);
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(f.outputs.reviewed_source_sha, reviewed);
  assertNoSourceAncestryWrites(f);
});

test("a pending source marker is never the reviewed checkpoint", async () => {
  const f = fixture({ prs: [mergedPr({
    body: MARKER + "\n" + sourceMarker(OLD) + "\n" + pendingMarker(S),
  })] });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).checkpointSha, OLD);
});

test("a replay retry does not require original source SHA ancestry", async () => {
  const f = fixture({
    sync: RETRY, prs: [mergedPr()], comparisons: { [RETRY + "..." + OLD]: "diverged" },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(firstReplay(f).baseSha, RETRY);
  assert.equal(firstReplay(f).checkpointSha, OLD);
  assertNoSourceAncestryWrites(f);
});

test("a retry branch missing the merged PR head is rejected", async () => {
  const f = fixture({
    sync: RETRY, prs: [mergedPr()], comparisons: { [RETRY + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /reviewed target|legacy source snapshot/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});

test("a source-only branch cannot bypass the reviewed target", async () => {
  const f = fixture({
    sync: S, prs: [mergedPr()], comparisons: { [S + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /reviewed target|legacy source snapshot/);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.refs["heads/" + SYNC], S);
});

test("an open source snapshot lacking the last merged head requires migration", async () => {
  const f = fixture({
    sync: S, prs: [pr({ body: MARKER + "\n" + sourceMarker(S) }), mergedPr({ number: 11 })],
    comparisons: { [S + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /reviewed target|legacy source snapshot/);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.refs["heads/" + SYNC], S);
});

test("an open replay branch need not contain its original source checkpoint SHA", async () => {
  for (const hasCompleted of [false, true]) {
    const f = fixture({
      sync: H, prs: [pr(), ...(hasCompleted ? [mergedPr({ number: 11 })] : [])],
      comparisons: { [H + "..." + OLD]: "diverged" },
    });
    assert.equal((await f.run()).status, "pr-updated");
    assert.equal(firstReplay(f).checkpointSha, OLD);
    assert.equal(firstReplay(f).baseSha, H);
    assertNoSourceAncestryWrites(f);
  }
});

test("a retry branch based on target with no remaining patches creates no empty PR", async () => {
  const f = fixture({ sync: T, replayChanged: false });
  assert.equal((await f.run()).status, "already-applied");
  assert.equal(firstReplay(f).baseSha, T);
  assert.equal(firstReplay(f).existingBranch, true);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.prs.length, 0);
});

test("legacy ACK ancestry is refused when the source checkpoint was excluded from target", async () => {
  const f = fixture({
    sync: H, prs: [pr(), mergedPr({ number: 11 })],
    comparisons: { [H + "..." + OLD]: "behind" },
  });
  await assert.rejects(f.run(), /excluded source ancestry/);
  assert.deepEqual(f.mutations(), []);
  assert.deepEqual(f.replays(), []);
});
