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
const SYNC = SYNC_PREFIX + S, OLD_BRANCH = SYNC_PREFIX + OLD;
const sourceMarker = (sha) => "<!-- forward-sync-source: " + sha + " -->";

function pr(overrides = {}) {
  return {
    number: 12, state: "open", merged_at: null,
    html_url: "https://github.com/flagos-ai/vllm-plugin-FL/pull/12",
    body: MARKER + "\n" + sourceMarker(OLD) + "\nMaintainer notes: preserve this.",
    head: { sha: H, ref: SYNC, repo: { full_name: "flagos-ai/vllm-plugin-FL" } }, ...overrides,
  };
}

function mergedPr() {
  return pr({
    state: "closed", merged_at: "2026-10-19T00:00:00Z",
    head: { sha: H, ref: OLD_BRANCH, repo: { full_name: "flagos-ai/vllm-plugin-FL" } },
  });
}

function fixture(options = {}) {
  const refs = {
    ["heads/" + SOURCE]: S, ["heads/" + TARGET]: T,
    ...(options.sync ? { ["heads/" + SYNC]: options.sync } : {}),
    ...(options.oldTip ? { ["heads/" + OLD_BRANCH]: options.oldTip } : {}),
  };
  const prs = structuredClone(options.prs || []);
  const comparisons = {
    [T + "..." + S]: "diverged", [S + "..." + OLD]: "behind",
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
    run: () => forwardSync({ github, context, core }), calls, refs, prs, warnings, failures, outputs, errors,
    mutations: () => calls.filter(({ name }) => [
      "createRef", "updateRef", "merge", "createPR", "updatePR",
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

test("a new cycle starts from source even if it conflicts with previous resolution history", async () => {
  const f = fixture({
    oldTip: H, prs: [mergedPr()],
    comparisons: { [T + "..." + H]: "behind", [H + "..." + S]: "diverged" },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.mutations().map((c) => c.name), ["createRef", "createPR"]);
  assert.equal(f.refs["heads/" + SYNC], S);
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
});

test("a deleted merged branch is recreated only after validating its historical head", async () => {
  const f = fixture({
    prs: [mergedPr()],
    comparisons: { [T + "..." + H]: "behind" },
  });
  assert.equal((await f.run()).status, "pr-created");
  assert.deepEqual(f.mutations().map((c) => c.name), ["createRef", "createPR"]);
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

test("a new-cycle PR creation failure can be retried without touching the merged branch", async () => {
  const f = fixture({
    oldTip: H, prs: [mergedPr()], comparisons: { [T + "..." + H]: "behind" },
    errors: { createPR: 403 },
  });
  await assert.rejects(f.run(), { status: 403 });
  assert.equal(f.refs["heads/" + SYNC], S);
  delete f.errors.createPR;
  assert.equal((await f.run()).status, "pr-created");
  assert.equal(f.refs["heads/" + OLD_BRANCH], H);
  assert.equal(f.calls.filter((c) => c.name === "createRef").length, 1);
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
