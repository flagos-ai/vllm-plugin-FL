# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Remove authenticated URLs from Docker image metadata, preserving layers/config.

Uses a standard Docker save/load archive; no host daemon storage is edited.
The output archive is retained for review. Does not push an image.
"""

import argparse
import copy
import hashlib
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--source", required=True)
parser.add_argument("--target", required=True)
parser.add_argument("--archive", required=True)
parser.add_argument("--report", required=True)
args = parser.parse_args()
auth_url = re.compile(r"(https?://)[^/\s:@]+:[^@\s/]+@")


def scrub(value):
    if isinstance(value, str):
        return auth_url.sub(r"\1", value)
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def inspect(name):
    return json.loads(subprocess.check_output(["docker", "image", "inspect", name]))[0]


before = inspect(args.source)
metadata = {}
print("Exporting image layers and sanitizing metadata", flush=True)
with subprocess.Popen(["docker", "save", args.source], stdout=subprocess.PIPE) as save:
    with (
        tarfile.open(fileobj=save.stdout, mode="r|") as source,
        tarfile.open(args.archive, mode="w") as output,
    ):
        for entry in source:
            basename = Path(entry.name).name
            if entry.isfile() and (
                entry.name.endswith(".json") or basename in ("json", "repositories")
            ):
                metadata[entry.name] = (
                    copy.copy(entry),
                    json.load(source.extractfile(entry)),
                )
            else:
                output.addfile(
                    entry, source.extractfile(entry) if entry.isfile() else None
                )
        manifest_entry, manifest = metadata["manifest.json"]
        assert len(manifest) == 1, "Save exactly one source image"
        old_name = manifest[0]["Config"]
        config_entry, config = metadata[old_name]
        cleaned = scrub(config)
        assert config["rootfs"] == cleaned["rootfs"]
        assert config["config"] == cleaned["config"], "Runtime config would change"
        config_bytes = json.dumps(cleaned, separators=(",", ":")).encode()
        new_name = hashlib.sha256(config_bytes).hexdigest() + ".json"
        manifest[0]["Config"] = new_name
        manifest[0]["RepoTags"] = [args.target]
        for name, (entry, value) in metadata.items():
            if Path(name).name == "repositories":
                continue  # Modern manifest supplies the destination tag.
            if name == old_name:
                entry.name = new_name
                payload = config_bytes
            else:
                payload = json.dumps(scrub(value), separators=(",", ":")).encode()
            entry.size = len(payload)
            output.addfile(entry, io.BytesIO(payload))
    if save.wait() != 0:
        raise RuntimeError("docker save failed")
print("Loading sanitized image archive", flush=True)
subprocess.run(["docker", "load", "--input", args.archive], check=True)
after = inspect(args.target)
assert before["RootFS"] == after["RootFS"], "Filesystem layers differ"
assert before["Config"] == after["Config"], "Runtime config differs"
history = subprocess.check_output(
    ["docker", "history", "--no-trunc", args.target], text=True
)
assert not auth_url.search(history), "Authenticated URL remains in image history"
report = {
    "source_image": args.source,
    "source_id": before["Id"],
    "target_image": args.target,
    "target_id": after["Id"],
    "filesystem_layers_equal": True,
    "runtime_config_equal": True,
    "authenticated_urls_in_published_history": False,
    "archive": args.archive,
}
Path(args.report).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
