"""Exercise ordinary-wheel evidence boundaries with small real ZIP inputs."""

import base64
import csv
import hashlib
import importlib.util
import io
import tempfile
import unittest
import zipfile
from pathlib import Path

source = Path(__file__).with_name("verify_stack.py")
spec = importlib.util.spec_from_file_location("image_stack_verifier", source)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def make_wheel(root, *, forge_record=False, extra=None):
    payload = {
        "example/__init__.py": b"VALUE = 1\n",
        "example-1.0.dist-info/METADATA": b"Metadata-Version: 2.1\nName: example\nVersion: 1.0\n",
    }
    if extra:
        payload.update(extra)
    record = []
    for name, body in payload.items():
        digest = hashlib.sha256(body).digest()
        encoded = base64.urlsafe_b64encode(digest).decode().rstrip("=")
        record.append([name, "sha256=" + encoded, str(len(body))])
    if forge_record:
        record[0][1] = "sha256=" + "A" * 43
    record.append(["example-1.0.dist-info/RECORD", "", ""])
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(record)
    payload["example-1.0.dist-info/RECORD"] = buf.getvalue().encode()
    path = Path(root) / "example-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        for name, body in payload.items():
            wheel.writestr(name, body)
    raw = path.read_bytes()
    return path, {
        "name": "example",
        "version": "1.0",
        "filename": path.name,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


class EvidenceBoundaryTests(unittest.TestCase):
    def test_whole_sha_does_not_replace_record_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = make_wheel(directory)
            self.assertEqual(len(verifier.wheel_review(path, row)["payload"]), 2)
            path, row = make_wheel(directory, forge_record=True)
            with self.assertRaisesRegex(ValueError, "RECORD hash mismatch"):
                verifier.wheel_review(path, row)

    def test_source_path_hook_rejected_even_with_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = make_wheel(
                directory, extra={"__editable__.example.pth": b"/private/source\n"}
            )
            with self.assertRaisesRegex(ValueError, "editable/source path hook"):
                verifier.wheel_review(path, row)

    def test_escape_rejected_even_with_valid_record(self):
        with tempfile.TemporaryDirectory() as directory:
            path, row = make_wheel(directory, extra={"../escape.py": b"VALUE = 2\n"})
            with self.assertRaisesRegex(ValueError, "Unsafe wheel member"):
                verifier.wheel_review(path, row)


if __name__ == "__main__":
    unittest.main()
