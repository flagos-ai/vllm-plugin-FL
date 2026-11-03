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
const ACK = SHA("f"), TARGET_TREE = SHA("1");
const SYNC = SYNC_PREFIX + S, OLD_BRANCH = SYNC_PREFIX + OLD;
const sourceMarker = (sha) => "<!-- forward-sync-source: " + sha + " -->";
const pendingMarker = (sha) => "<!-- forward-sync-pending: " + sha + " -->";

function pr(overrides = {}) {
  return {
    number: 12, state: "open", merged_at: null,
    html_url: "https://github.com/flagos-ai/vllm-plugin-FL/pull/12",
    body: MARKER + "\n" + sourceMarker(OLD) + "\nMaintainer notes: preserve this.",
    head: { sha: H, ref: SYNC, repo: { full_name: "flagos-ai/vllm-plugin-FL" } }, ...overrides,
  };
}

function mergedPr(overrides = {}) {
  return pr({
    state: "closed", merged_at: "2026-10-19T00:00:00Z",
    head: { sha: H, ref: OLD_BRANCH, repo: { full_name: "flagos-ai/vllm-plugin-FL" } },
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
  const comparisons = {
    [T + "..." + S]: "diverged", [S + "..." + OLD]: "behind",
    [T + "..." + H]: "behind", [T + "..." + OLD]: "diverged",
    [H + "..." + OLD]: "behind",
    [ACK + "..." + S]: "diverged", [ACK + "..." + H]: "behind",
    [ACK + "..." + OLD]: "behind", [MERGED + "..." + S]: "behind",
    [MERGED + "..." + H]: "behind", [MERGED + "..." + OLD]: "behind",
    ...(options.comparisons || {}),
  };
  const errors = { ...(options.errors || {}) };
  const calls = [], warnings = [], failures = [], outputs = {};
  const fail = (status, message = "GitHub API failure") => Object.assign(new Error(message), { status });
  const method = (name, fn) => async (args) => {
    calls.push({ name, args });
    if (errors[name]) throw fail(errors[name]);
    return { data: fn(args) };
  };
  const github = {
    rest: {
      git: {
        getCommit: method("getCommit", ({ commit_sha }) => {
          assert.equal(commit_sha, T);
          return { sha: T, tree: { sha: TARGET_TREE } };
        }),
        createCommit: method("createCommit", ({ tree, parents }) => {
          assert.equal(tree, TARGET_TREE);
          assert.equal(parents[0], T);
          return { sha: ACK };
        }),
        getRef: method("getRef", ({ ref }) => {
          if (!refs[ref]) throw fail(404);
          return { object: { sha: refs[ref] } };
        }),
        createRef: method("createRef", ({ ref, sha }) => {
          refs[ref.replace(/^refs\//, "")] = sha;
          return {};
        }),
        updateRef: method("updateRef", ({ ref, sha, force }) => {
          assert.equal(force, false);
          refs[ref] = sha;
          return {};
        }),
      },
      repos: {
        compareCommitsWithBasehead: method("compare", ({ basehead }) => {
          assert.ok(basehead in comparisons, "Missing comparison fixture: " + basehead);
          return { status: comparisons[basehead] };
        }),
        merge: method("merge", ({ base }) => {
          refs["heads/" + base] = MERGED;
          return { sha: MERGED };
        }),
      },
      pulls: {
        get: method("getPR", ({ pull_number }) => {
          const found = prs.find((item) => item.number === pull_number);
          if (options.closeOnRead) found.state = "closed";
          return found;
        }),
        list: method("listPRs", ({ base, head, state }) => {
          assert.equal(base, TARGET);
          assert.equal(head, undefined);
          assert.equal(state, "all");
          return prs;
        }),
        create: method("createPR", (args) => {
          const created = pr({
            number: 20, html_url: "https://github.com/flagos-ai/vllm-plugin-FL/pull/20",
            body: args.body, head: { sha: refs["heads/" + args.head], ref: args.head,
              repo: { full_name: "flagos-ai/vllm-plugin-FL" } },
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
  const context = {
    repo: { owner: "flagos-ai", repo: "vllm-plugin-FL" }, payload: options.payload || {},
  };
  return {
    run: () => forwardSync({ github, context, core }), calls, refs, prs, warnings, failures, outputs, errors, comparisons,
    mutations: () => calls.filter(({ name }) => [
      "createCommit", "createRef", "updateRef", "merge", "createPR", "updatePR",
    ].includes(name)),
  };
}

test("first sync creates a source snapshot and a manual PR even when target diverged", async () => {
  const f = fixture();
  const result = await f.run();
  assert.equal(result.status, "pr-created");
  assert.equal(result.pullRequestUrl, f.prs[0].html_url);
  assert.deepEqual(f.mutations().map((c) => c.name), ["createRef", "createPR"]);
  const created = f.calls.find((c) => c.name === "createPR").args;
  assert.equal(created.base, TARGET);
  assert.equal(created.head, SYNC);
  assert.match(created.body, /Automatic merging is disabled/);
  assert.match(created.body, /Do not squash or rebase/);
  assert.ok(created.body.includes(sourceMarker(S)));
  assert.equal(f.refs["heads/" + SYNC], S);
});

test("unchanged retry reuses the PR without writes", async () => {
  const f = fixture({ sync: S, prs: [pr({ body: MARKER + "\n" + sourceMarker(S) })] });
  assert.equal((await f.run()).status, "pr-updated");
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.calls.filter((c) => c.name === "getPR").length, 1);
});

test("source already included in target requires no branch or PR writes", async () => {
  const f = fixture({ comparisons: { [T + "..." + S]: "behind" } });
  assert.equal((await f.run()).status, "already-included");
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.calls.some((c) => c.name === "listPRs"), false);
});

test("fast-forward uses a non-forced ref update", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "ahead" },
  });
  await f.run();
  assert.equal(f.refs["heads/" + SYNC], S);
  assert.equal(f.calls.find((c) => c.name === "updateRef").args.force, false);
});

test("human conflict-resolution commits are preserved by merging source into the sync branch", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "diverged" },
  });
  await f.run();
  const merge = f.calls.find((c) => c.name === "merge").args;
  assert.equal(merge.base, SYNC);
  assert.equal(merge.head, S);
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.match(f.prs[0].body, /Maintainer notes: preserve this/);
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
  assert.equal(f.calls.some((c) => c.name === "updateRef"), false);
});

test("a source conflict preserves the branch and keeps the existing PR", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "diverged" },
    errors: { merge: 409 },
  });
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  assert.equal(f.refs["heads/" + SYNC], H);
  assert.equal(f.prs.length, 1);
  assert.equal(f.warnings.length, 1);
  assert.equal(f.failures.length, 1);
  assert.ok(f.failures[0].includes(f.prs[0].html_url));
  assert.equal(f.calls.some((c) => c.name === "createPR"), false);
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  assert.ok(!f.prs[0].body.includes(sourceMarker(S)));
  assert.ok(f.prs[0].body.includes(pendingMarker(S)));
  assert.match(f.prs[0].body, /Maintainer notes: preserve this/);
});

test("a branch already containing source does not receive a redundant merge", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "behind" },
  });
  await f.run();
  assert.deepEqual(f.mutations().map((c) => c.name), ["updatePR"]);
});

test("a closed, unmerged PR is not silently reopened or overwritten", async () => {
  const f = fixture({ sync: H, prs: [pr({ state: "closed" })] });
  await assert.rejects(f.run(), /closed without merging/);
  assert.deepEqual(f.mutations(), []);
});

test("an unrelated existing branch is protected", async () => {
  const f = fixture({ sync: H });
  await assert.rejects(f.run(), /unexpected commits/);
  assert.deepEqual(f.mutations(), []);
});

test("an unmarked PR is protected", async () => {
  const f = fixture({ sync: H, prs: [pr({ body: "Someone else's PR" })] });
  await assert.rejects(f.run(), /unrecognized PR/);
  assert.deepEqual(f.mutations(), []);
});

test("a merged PR with lost ancestry stops instead of replaying squash-merged history", async () => {
  const f = fixture({
    oldTip: H, prs: [mergedPr()],
    comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
});

test("a new cycle preserves reviewed exclusions instead of restarting from the source tree", async () => {
  const f = fixture({ oldTip: H, prs: [mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.mutations().map((c) => c.name), [
    "createCommit", "createRef", "merge", "createPR",
  ]);
  const acknowledgment = f.calls.find((c) => c.name === "createCommit").args;
  // The source checkpoint can include deliberately excluded commits. Recording
  // its ancestry must preserve the reviewed target tree, not copy source files.
  assert.equal(acknowledgment.tree, TARGET_TREE);
  assert.deepEqual(acknowledgment.parents, [T, OLD]);
  assert.match(acknowledgment.message, /PR #12/);
  assert.equal(f.calls.find((c) => c.name === "createRef").args.sha, ACK);
  assert.equal(f.calls.find((c) => c.name === "merge").args.head, S);
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
  assert.equal(f.refs["heads/" + TARGET], T);
  assert.equal(f.outputs.reviewed_source_sha, OLD);
});

test("a deleted merged branch starts a new incremental sync after historical-head validation", async () => {
  const f = fixture({ prs: [mergedPr()] });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.mutations().map((c) => c.name), [
    "createCommit", "createRef", "merge", "createPR",
  ]);
  assert.equal(f.refs["heads/" + OLD_BRANCH], undefined);
});

test("source history rewrite detected from the prior checkpoint stops all mutations", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [S + "..." + OLD]: "diverged" },
  });
  await assert.rejects(f.run(), /rewound or rewritten/);
  assert.deepEqual(f.mutations(), []);
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
});

test("PR creation permission failure can be retried safely after branch creation", async () => {
  const f = fixture({ errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], S);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.calls.filter((c) => c.name === "createRef").length, 1);
  assert.equal(f.prs.length, 1);
});

test("missing checkpoint and duplicate open PRs fail without writes", async () => {
  const missing = fixture({ sync: H, prs: [pr({ body: MARKER })] });
  await assert.rejects(missing.run(), /no unique source SHA marker/);
  assert.deepEqual(missing.mutations(), []);
  const duplicate = fixture({ sync: H, prs: [pr(), pr({ number: 13 })] });
  await assert.rejects(duplicate.run(), /Multiple open/);
  assert.deepEqual(duplicate.mutations(), []);
});

test("unexpected merge failures propagate and never create a replacement PR", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "diverged" },
    errors: { merge: 403 },
  });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.calls.some((c) => c.name === "createPR"), false);
});

test("a new-cycle PR creation failure reuses the reviewed branch without duplicate merges", async () => {
  const f = fixture({ oldTip: H, prs: [mergedPr()], errors: { createPR: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
  for (const method of ["createCommit", "createRef", "merge"]) {
    assert.equal(f.calls.filter((c) => c.name === method).length, 1);
  }
});

test("manual commits added after a merged PR are not silently abandoned", async () => {
  const f = fixture({
    oldTip: MERGED, prs: [mergedPr()],
    comparisons: { [T + "..." + H]: "behind", [T + "..." + MERGED]: "diverged" },
  });
  await assert.rejects(f.run(), /reconcile those manual commits/);
  assert.deepEqual(f.mutations(), []);
});

test("resetting an old branch cannot hide a squash merge's missing PR ancestry", async () => {
  const f = fixture({
    oldTip: T, prs: [mergedPr()], comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
});

test("a fork PR with a matching branch prefix does not impersonate the managed PR", async () => {
  const f = fixture({
    prs: [pr({
      body: "Untrusted fork PR",
      head: { sha: H, ref: SYNC, repo: { full_name: "someone/vllm-plugin-FL" } },
    })],
  });
  assert.equal((await f.run()).status, "pr-created");
});

test("an open PR with a missing branch stops for manual recovery", async () => {
  const f = fixture({ prs: [pr()] });
  await assert.rejects(f.run(), /branch is missing/);
  assert.deepEqual(f.mutations(), []);
});

test("a PR closed during its branch update fails visibly instead of reporting success", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "ahead" },
    closeOnRead: true,
  });
  await assert.rejects(f.run(), (error) =>
    /closed while its branch was being updated/.test(error.message) &&
    error.message.includes(f.prs[0].html_url));
  assert.equal(f.refs["heads/" + SYNC], S);
  assert.equal(f.outputs.status, undefined);
  assert.equal(f.calls.some((c) => c.name === "createPR"), false);
});


test("an unchanged reviewed checkpoint creates no PR even when source ancestry was intentionally excluded", async () => {
  const f = fixture({ source: OLD, oldTip: H, prs: [mergedPr()] });
  assert.equal((await f.run()).status, "already-processed");
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.calls.some((c) => c.name === "getCommit"), false);
});

test("a target already containing the reviewed checkpoint needs no acknowledgment commit", async () => {
  const f = fixture({
    prs: [mergedPr()], comparisons: { [T + "..." + OLD]: "behind" },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.mutations().map((c) => c.name), ["createRef", "merge", "createPR"]);
  assert.equal(f.calls.find((c) => c.name === "createRef").args.sha, T);
  assert.equal(f.calls.some((c) => c.name === "getCommit"), false);
});

test("a conflicting new increment keeps its reviewed branch without creating an empty PR", async () => {
  const f = fixture({ prs: [mergedPr()], errors: { merge: 409 } });
  const result = await f.run();
  assert.equal(result.status, "conflict-needs-resolution");
  assert.equal(result.pullRequest, null);
  assert.equal(result.pullRequestUrl, null);
  assert.equal(f.refs["heads/" + SYNC], ACK);
  assert.equal(f.prs.length, 1);
  assert.deepEqual(f.mutations().map((c) => c.name), ["createCommit", "createRef", "merge"]);
  assert.equal(f.outputs.pull_request_number, undefined);
  assert.match(f.failures[0], /No PR was created before resolution/);
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
});

test("a new increment can retry from its preserved acknowledgment after a merge conflict", async () => {
  const f = fixture({ prs: [mergedPr()], errors: { merge: 409 } });
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  delete f.errors.merge;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.refs["heads/" + SYNC], MERGED);
  assert.equal(f.calls.filter((c) => c.name === "createCommit").length, 1);
  assert.equal(f.calls.filter((c) => c.name === "createRef").length, 1);
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
});

test("retrying an open conflict never advances its checkpoint or duplicates the pending marker", async () => {
  const f = fixture({
    sync: H, prs: [pr()], comparisons: { [H + "..." + S]: "diverged" },
    errors: { merge: 409 },
  });
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  assert.equal((await f.run()).status, "conflict-needs-resolution");
  assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  assert.equal(f.prs[0].body.split(pendingMarker(S)).length - 1, 1);
  assert.equal(f.calls.filter((c) => c.name === "updatePR").length, 1);
});

test("successful conflict resolution advances the checkpoint and removes its pending marker", async () => {
  const f = fixture({
    sync: H, prs: [pr({ body: MARKER + "\n" + sourceMarker(OLD) + "\n" + pendingMarker(S) })],
    comparisons: { [H + "..." + S]: "behind" },
  });
  assert.equal((await f.run()).status, "pr-updated");
  assert.ok(f.prs[0].body.includes(sourceMarker(S)));
  assert.ok(!f.prs[0].body.includes("forward-sync-pending"));
  assert.equal(f.calls.some((c) => c.name === "merge"), false);
});

test("an open PR still validates the last merged checkpoint and exposes it separately", async () => {
  const f = fixture({
    sync: H, prs: [pr({ body: MARKER + "\n" + sourceMarker(S) }), mergedPr({ number: 11 })],
    comparisons: { [H + "..." + S]: "behind" },
  });
  assert.equal((await f.run()).status, "pr-updated");
  assert.equal(f.outputs.reviewed_source_sha, OLD);
  assert.ok(f.calls.some((c) => c.name === "compare" && c.args.basehead === T + "..." + H));
  assert.equal(f.calls.some((c) => c.name === "createCommit"), false);
});

test("an open PR cannot hide lost ancestry from the last merged sync", async () => {
  const f = fixture({
    sync: H, prs: [pr(), mergedPr({ number: 11 })],
    comparisons: { [T + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /squash\/rebase merge/);
  assert.deepEqual(f.mutations(), []);
});

test("an open PR cannot replace a missing merged checkpoint with its observed source SHA", async () => {
  const f = fixture({
    sync: H, prs: [pr(), mergedPr({ number: 11, body: MARKER })],
  });
  await assert.rejects(f.run(), /no unique source SHA marker/);
  assert.deepEqual(f.mutations(), []);
});

test("the most recently merged checkpoint wins over PR creation order", async () => {
  const reviewed = SHA("2"), reviewedHead = SHA("3");
  const f = fixture({
    prs: [mergedPr(), mergedPr({
      number: 11, merged_at: "2026-10-20T00:00:00Z",
      body: MARKER + "\n" + sourceMarker(reviewed),
      head: { sha: reviewedHead, ref: SYNC_PREFIX + reviewed,
        repo: { full_name: "flagos-ai/vllm-plugin-FL" } },
    })],
    comparisons: {
      [S + "..." + reviewed]: "behind", [T + "..." + reviewedHead]: "behind",
      [T + "..." + reviewed]: "diverged",
    },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.calls.find((c) => c.name === "createCommit").args.parents, [T, reviewed]);
  assert.equal(f.outputs.reviewed_source_sha, reviewed);
});

test("a pending source marker is never used as the reviewed checkpoint", async () => {
  const f = fixture({
    prs: [mergedPr({ body: MARKER + "\n" + sourceMarker(OLD) + "\n" + pendingMarker(S) })],
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.calls.find((c) => c.name === "createCommit").args.parents, [T, OLD]);
});

test("a checkpoint-aware retry rejects a branch missing the reviewed source ancestry", async () => {
  const f = fixture({
    sync: ACK, prs: [mergedPr()], comparisons: { [ACK + "..." + OLD]: "diverged" },
  });
  await assert.rejects(f.run(), /unexpected commits/);
  assert.deepEqual(f.mutations(), []);
});

test("a checkpoint-aware retry rejects a branch missing the merged PR head", async () => {
  const f = fixture({
    sync: ACK, prs: [mergedPr()], comparisons: { [ACK + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /unexpected commits/);
  assert.deepEqual(f.mutations(), []);
});

test("an acknowledgment API failure never publishes a source snapshot as a fallback", async () => {
  const f = fixture({ prs: [mergedPr()], errors: { createCommit: 403 } });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], undefined);
  assert.equal(f.calls.some((c) => c.name === "createRef" || c.name === "createPR"), false);
});


test("a legacy source-only branch without a PR cannot bypass the reviewed target checkpoint", async () => {
  const f = fixture({
    sync: S, prs: [mergedPr()], comparisons: { [S + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /legacy source snapshot/);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.refs["heads/" + SYNC], S);
});

test("a legacy open source snapshot is preserved for migration instead of replaying exclusions", async () => {
  const f = fixture({
    sync: S, prs: [pr({ body: MARKER + "\n" + sourceMarker(S) }), mergedPr({ number: 11 })],
    comparisons: { [S + "..." + H]: "diverged" },
  });
  await assert.rejects(f.run(), /legacy source snapshot/);
  assert.deepEqual(f.mutations(), []);
  assert.equal(f.refs["heads/" + SYNC], S);
});

test("rewriting an open sync cannot discard its source checkpoint before the next update", async () => {
  for (const hasCompleted of [false, true]) {
    const f = fixture({
      sync: H, prs: [pr(), ...(hasCompleted ? [mergedPr({ number: 11 })] : [])],
      comparisons: { [H + "..." + OLD]: "diverged" },
    });
    await assert.rejects(f.run(), /no longer contains its recorded source checkpoint/);
    assert.deepEqual(f.mutations(), []);
    assert.equal(f.refs["heads/" + SYNC], H);
    assert.ok(f.prs[0].body.includes(sourceMarker(OLD)));
  }
});
