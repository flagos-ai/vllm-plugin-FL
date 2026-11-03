/*
 * Copyright 2026 FlagOS Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

"use strict";

const { execFile } = require("node:child_process");
const fs = require("node:fs/promises");
const os = require("node:os");
const path = require("node:path");
const { promisify } = require("node:util");

const execFileAsync = promisify(execFile);
const SHA = /^[0-9a-f]{40}$/;
const SYNC_BRANCH = /^codex\/sync-0\.4-to-0\.5-[0-9a-f]{40}$/;

/** Replay reviewed source increments without importing source ancestry. */
module.exports = async function replay({
  sourceSha, baseSha, checkpointSha = null, syncBranch,
  repository, token, core, existingBranch = false, remoteUrl,
}) {
  for (const [name, sha] of Object.entries({ sourceSha, baseSha, checkpointSha })) {
    if ((name !== "checkpointSha" || sha !== null) && !SHA.test(sha || "")) {
      throw new Error("Invalid " + name + " for forward-sync replay.");
    }
  }
  if (!SYNC_BRANCH.test(syncBranch || "")) {
    throw new Error("Refusing to push outside the managed forward-sync branch namespace.");
  }
  if (!/^[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(repository || "")) {
    throw new Error("Invalid GitHub repository for forward-sync replay.");
  }
  if (remoteUrl !== undefined && (typeof remoteUrl !== "string" || !path.isAbsolute(remoteUrl))) {
    throw new Error("The test transport must be an absolute local repository path.");
  }
  if ((!remoteUrl && (typeof token !== "string" || !token)) ||
      (token !== undefined && (typeof token !== "string" || /[\r\n\0]/.test(token)))) {
    throw new Error("A GitHub token is required for forward-sync replay.");
  }

  const authorization = token ? "AUTHORIZATION: basic " + Buffer.from("x-access-token:" + token).toString("base64") : "";
  const env = { ...process.env };
  // Ignore caller Git state, trace settings, global helpers and signing hooks.
  // Authentication is child-process environment only, never a URL/config file.
  for (const name of Object.keys(env)) {
    if (name.startsWith("GIT_")) delete env[name];
  }
  Object.assign(env, {
    GIT_CONFIG_NOSYSTEM: "1",
    GIT_CONFIG_GLOBAL: process.platform === "win32" ? "NUL" : "/dev/null",
    GIT_TERMINAL_PROMPT: "0",
    GIT_NO_REPLACE_OBJECTS: "1",
    GIT_DEFAULT_HASH: "sha1",
    GIT_COMMITTER_NAME: "github-actions[bot]",
    GIT_COMMITTER_EMAIL: "41898282+github-actions[bot]@users.noreply.github.com",
    GCM_INTERACTIVE: "Never",
  });
  const config = [
    ["core.hooksPath", process.platform === "win32" ? "NUL" : "/dev/null"],
    ["core.autocrlf", "false"],
    ["commit.gpgSign", "false"],
    ["tag.gpgSign", "false"],
    ["user.name", "github-actions[bot]"],
    ["user.email", "41898282+github-actions[bot]@users.noreply.github.com"],
    ["credential.helper", ""],
    ["gc.auto", "0"],
  ];
  if (authorization) config.push(["http.https://github.com/.extraheader", authorization]);
  env.GIT_CONFIG_COUNT = String(config.length);
  config.forEach(([key, value], index) => {
    env["GIT_CONFIG_KEY_" + index] = key;
    env["GIT_CONFIG_VALUE_" + index] = value;
  });

  const directory = await fs.mkdtemp(path.join(os.tmpdir(), "vllm-fl-forward-sync-"));
  const originalCheckpoint = checkpointSha;
  function info(message) {
    if (core && typeof core.info === "function") core.info(message);
  }
  function errorFor(operation, result) {
    // Even Git's diagnostic output must not expose credentials if an upstream
    // HTTP client happens to include a header in an error message.
    const encoded = authorization.slice(authorization.lastIndexOf(" ") + 1);
    let detail = String(result.stderr || result.stdout || "");
    if (token) detail = detail.split(token).join("[REDACTED]");
    if (encoded) detail = detail.split(encoded).join("[REDACTED]");
    detail = detail.trim();
    return new Error("git " + operation + " failed (exit " + result.code + ")." +
      (detail ? " " + detail : ""));
  }
  async function git(args, allowFailure = false) {
    let result;
    try {
      const output = await execFileAsync("git", args, {
        cwd: directory, env, encoding: "utf8", windowsHide: true,
        maxBuffer: 64 * 1024 * 1024,
      });
      result = { code: 0, stdout: output.stdout, stderr: output.stderr };
    } catch (error) {
      if (typeof error.code !== "number") {
        throw new Error("Could not execute git " + args[0] + " for forward sync.");
      }
      result = { code: error.code, stdout: error.stdout, stderr: error.stderr };
    }
    if (result.code !== 0 && !allowFailure) throw errorFor(args[0], result);
    return result;
  }

  try {
    await git(["init", "--quiet"]);
    await git(["remote", "add", "origin", (remoteUrl || "https://github.com/" + repository + ".git")]);
    // Fetch complete ancestry for the SHA snapshots, not a shallow source tip.
    const snapshots = [...new Set([sourceSha, baseSha, checkpointSha].filter(Boolean))];
    await git(["fetch", "--quiet", "--no-tags", "--no-recurse-submodules", "origin", ...snapshots]);
    if (checkpointSha === null) {
      checkpointSha = (await git(["merge-base", baseSha, sourceSha])).stdout.trim();
    }
    const firstParents = new Set((await git(["rev-list", "--first-parent", sourceSha]))
      .stdout.trim().split(/\r?\n/));
    if (!firstParents.has(checkpointSha)) {
      throw new Error("The reviewed source checkpoint is not on the current source first-parent history; reconcile the rewrite manually.");
    }

    const commits = (await git([
      "rev-list", "--first-parent", "--parents", "--reverse", checkpointSha + ".." + sourceSha,
    ])).stdout.trim().split(/\r?\n/).filter(Boolean).map((line) => line.split(" "));
    await git(["checkout", "--quiet", "--detach", baseSha]);

    const included = new Set((await git(["rev-list", baseSha])).stdout.trim().split(/\r?\n/));
    const messages = (await git([
      "log", "--format=%B%x00", "--fixed-strings", "--grep=(cherry picked from commit ", baseSha,
    ])).stdout;
    const replayed = new Set([...messages.matchAll(/^\(cherry picked from commit ([0-9a-f]{40})\)$/gm)]
      .map((match) => match[1]));
    // Patch equivalence handles independently cherry-picked ordinary commits;
    // exact -x trailers also cover resolutions whose patch IDs have changed.
    const equivalent = new Set((await git(["cherry", baseSha, sourceSha, checkpointSha]))
      .stdout.trim().split(/\r?\n/).filter((line) => line.startsWith("- "))
      .map((line) => line.slice(2).trim()));

    for (const [sha, ...parents] of commits) {
      if (included.has(sha) || replayed.has(sha)) {
        info("Already processed source commit " + sha + "; skipping replay.");
        continue;
      }
      if (parents.length === 1 && equivalent.has(sha)) {
        // A historical patch match may have been reverted later. Let the
        // actual pick prove it is empty instead of silently dropping a re-add.
        info("Patch-equivalent source commit " + sha + "; checking its current-tree effect.");
      }
      const args = ["cherry-pick", "-x"];
      if (parents.length > 1) args.push("-m", "1");
      args.push(sha);
      const picked = await git(args, true);
      if (picked.code === 0) continue;

      const unresolved = await git(["diff", "--name-only", "--diff-filter=U", "-z"]);
      if (unresolved.stdout) {
        await git(["cherry-pick", "--abort"], true);
        await git(["reset", "--hard", baseSha]);
        info("Source commit " + sha + " conflicts; the complete replay was discarded without a push.");
        return { sha: baseSha, checkpointSha: originalCheckpoint, conflicted: true, changed: false };
      }
      const inProgress = await git(["rev-parse", "--quiet", "--verify", "CHERRY_PICK_HEAD"], true);
      const staged = await git(["diff", "--cached", "--quiet"], true);
      const unstaged = await git(["diff", "--quiet"], true);
      if (inProgress.code === 0 && staged.code === 0 && unstaged.code === 0) {
        // Git stops on initially empty and redundant picks on older runners.
        // Skip these without creating an empty commit or requiring a new flag.
        await git(["cherry-pick", "--skip"]);
        info("Source commit " + sha + " is empty after replay; skipping it.");
        continue;
      }
      throw errorFor("cherry-pick", picked);
    }

    const sha = (await git(["rev-parse", "HEAD"])).stdout.trim();
    if (sha === baseSha) {
      return { sha, checkpointSha: sourceSha, conflicted: false, changed: false };
    }
    if ((await git(["merge-base", "--is-ancestor", baseSha, sha], true)).code !== 0) {
      throw new Error("Replay no longer descends from the reviewed base; refusing to push.");
    }
    const advertised = (await git(["ls-remote", "--heads", "origin", "refs/heads/" + syncBranch]))
      .stdout.trim().split(/\s+/)[0];
    if ((existingBranch && advertised !== baseSha) || (!existingBranch && advertised)) {
      throw new Error("The sync branch changed during replay; retry without overwriting its commits.");
    }
    // The server also rejects concurrent non-fast-forward changes. No source
    // parents, force push, or target-branch write is used to record progress.
    await git(["push", "--porcelain", "origin", "HEAD:refs/heads/" + syncBranch]);
    return { sha, checkpointSha: sourceSha, conflicted: false, changed: true };
  } finally {
    const resolved = path.resolve(directory);
    if (path.dirname(resolved) !== path.resolve(os.tmpdir()) ||
        !path.basename(resolved).startsWith("vllm-fl-forward-sync-")) {
      throw new Error("Refusing to remove an unexpected replay temporary directory.");
    }
    await fs.rm(resolved, { recursive: true, force: true });
  }
};
