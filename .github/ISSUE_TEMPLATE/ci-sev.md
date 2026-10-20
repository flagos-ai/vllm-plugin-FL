---
name: CI SEV
about: Report a CI infrastructure or pipeline failure
title: "[CI SEV]: "
labels: "ci-sev"
assignees: ''
---

<!--
 Copyright 2026 FlagOS Contributors

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 -->


## Severity

<!-- Select one:
  SEV 0 - Main branch CI completely broken, all jobs failing, blocks all development
  SEV 1 - Critical job failing (e.g. functional GPU tests, build check), blocks merging
  SEV 2 - Non-critical job failing or flaky (e.g. lint, single Python version), does not fully block merging
-->

**Severity Level:**

## Failing Job

<!-- Which CI job is affected? Check all that apply. -->

- [ ] Lint (ruff check / ruff format / typos)
- [ ] Unit Tests
- [ ] Build Check
- [ ] Functional Tests - ops / compilation (GPU)
- [ ] Functional Tests - inference (GPU)
- [ ] Functional Tests - serving (GPU)
- [ ] Other: <!-- specify -->

## Description

<!-- Describe the failure. What is broken and how does it impact development? -->

## Failed Run Link

<!-- Paste the link to the failing GitHub Actions run. -->

## Error Logs

<!-- Paste the relevant error output (use <details> for long logs). -->

<details>
<summary>Error logs</summary>

```
paste logs here
```

</details>

## Likely Cause

<!-- If known, describe the suspected root cause. Examples:
  - Upstream dependency update
  - Self-hosted GPU runner offline / driver issue
  - Flaky test
  - Recent commit (link the commit)
-->

## Affected Branch / PR

<!-- Which branch or PR is affected? e.g. main, or PR #123 -->
