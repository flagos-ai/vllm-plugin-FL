#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fail when newly added lines in .py diffs contain Chinese characters.

Only *added* lines are checked: Chinese already present in the tree is
legacy debt this check deliberately ignores, so it never fails on
unchanged code. The diff is taken against the checked-out commit's first
parent, which is the PR base for pull_request merge-ref checkouts and
the previous commit for pushes.

Usage (from the workflow)::

    python.github / scripts / check_no_chinese_additions.py
"""

from __future__ import annotations

import re
import subprocess
import sys

# CJK Unified Ideographs (the Chinese character block).
CJK = re.compile(r"[\u4e00-\u9fff]")


def main() -> int:
    has_parent = (
        subprocess.run(
            ["git", "rev-parse", "--verify", "-q", "HEAD~1"],
            capture_output=True,
        ).returncode
        == 0
    )
    if not has_parent:
        print("No parent commit; nothing to diff.")
        return 0

    diff = subprocess.run(
        ["git", "diff", "HEAD~1..HEAD", "--unified=0", "--", "*.py"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout

    current_file: str | None = None
    line_no = 0
    bad = 0
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            current_file = line[len("+++ b/") :]
        elif line.startswith("@@"):
            match = re.search(r"\+(\d+)", line)
            line_no = int(match.group(1)) if match else 0
        elif line.startswith("+"):
            if current_file and CJK.search(line):
                text = line[1:].strip()
                print(
                    f"::error file={current_file},line={line_no}::"
                    "newly added line contains Chinese characters"
                )
                print(f"  {current_file}:{line_no}: {text}")
                bad += 1
            line_no += 1

    if bad:
        print(
            f"::error::{bad} added .py line(s) contain Chinese characters "
            "(CJK U+4E00-U+9FFF). Translate or remove them; existing "
            "Chinese elsewhere in the tree is not checked."
        )
        return 1
    print("No newly added Chinese characters in .py diffs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
