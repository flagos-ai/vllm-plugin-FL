"""Exercise the real curl retry path against a truncated HTTP transfer."""

import hashlib
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2] / ".github/scripts/iluvatar/install_flagtree.sh"
)


@pytest.mark.parametrize("mode", ["resume", "corrupt", "no_range", "always_fail"])
def test_interrupted_download_resumes_and_checks_hash(tmp_path, mode):
    payload = b"download transport fixture" * 4096
    ranges = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            value = self.headers.get("Range")
            ranges.append(value)
            if mode == "always_fail":
                self.send_error(503)
                return
            if mode == "no_range":
                value = None
            start = int(value.removeprefix("bytes=").rstrip("-")) if value else 0
            self.send_response(206 if value else 200)
            self.send_header("Content-Length", str(len(payload) - start))
            if value:
                self.send_header(
                    "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
                )
            self.end_headers()
            if len(ranges) == 1:
                self.wfile.write(payload[:1024])
                self.close_connection = True
            else:
                self.wfile.write(payload[start:])

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    source = SCRIPT.read_text()
    source = source.replace(
        'url="https://resource.flagos.net/repository/flagos-pypi-hosted/packages/flagtree/0.7.0+iluvatar3.6/${wheel}"',
        f'url="http://127.0.0.1:{server.server_port}/wheel"',
    )
    digest = "0" * 64 if mode == "corrupt" else hashlib.sha256(payload).hexdigest()
    source = source.replace(
        "af6a7c9eddf167441fd872c3e6d673e9da572608dee79dba655b0775db740176", digest
    )
    script = tmp_path / "install.sh"
    script.write_text(source)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    marker = tmp_path / "pip-called"
    (bindir / "python").write_text(f'#!/bin/sh\nprintf "%s\\n" "$@" > "{marker}"\n')
    (bindir / "python").chmod(0o755)
    (bindir / "sleep").write_text("#!/bin/sh\nexit 0\n")
    (bindir / "sleep").chmod(0o755)
    try:
        result = subprocess.run(
            ["bash", str(script)],
            env=dict(
                os.environ, PATH=f"{bindir}:{os.environ['PATH']}", NO_PROXY="127.0.0.1"
            ),
            text=True,
            capture_output=True,
            timeout=30,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
    assert "retrying" in result.stderr
    if mode == "always_fail":
        assert len(ranges) == 4
        assert result.returncode != 0
        assert not marker.exists()
        return
    expected = [None, "bytes=1024-"]
    if mode == "no_range":
        expected.append(None)
    assert ranges == expected
    if mode == "corrupt":
        assert result.returncode != 0
        assert "FAILED" in result.stdout
        assert not marker.exists()
    else:
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
        assert "--no-deps" in marker.read_text()
        archive = Path(marker.read_text().splitlines()[-1])
        assert not archive.exists(), "temporary wheel must be removed on exit"
