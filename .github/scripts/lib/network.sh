#!/bin/bash
# Copyright (c) 2026 BAAI. All rights reserved.
# Shared transport hardening for the CI "Install project" stage.
#
# Observed on this org's self-hosted runners (2026-10-13/14): transient
# git/HTTP failures such as "RPC failed; curl 92 HTTP/2 stream ... was not
# closed cleanly" and "gnutls_handshake() failed: The TLS connection was
# non-properly terminated". Checkout steps already retry three times; the
# install steps had no retry at all.
#
# These helpers never change WHAT gets installed. They only retry the exact
# same command with the same arguments and tune transport timeouts. When
# every attempt fails, the original non-zero exit code propagates.

FL_NET_RETRIES_DEFAULT=3
FL_NET_BACKOFF_SECONDS="${FL_NET_BACKOFF_SECONDS:-10}"

# Transport knobs applied before installs:
#   - pin git to HTTP/1.1: avoids HTTP/2 stream resets on lossy links
#   - raise pip/uv network timeouts from their aggressive defaults
fl_net_hardening() {
    git config --global http.version HTTP/1.1
    export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-60}"
    export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-60}"
}

# Retry a command with linear backoff: fl_net_retry [attempts] -- cmd args...
# Defaults to FL_NET_RETRIES (or 3). errexit-safe: the command runs in the
# until-condition position, so intermediate failures do not abort scripts
# running under `set -e`.
fl_net_retry() {
    local attempts="${FL_NET_RETRIES:-${FL_NET_RETRIES_DEFAULT}}"
    if [[ "${1:-}" =~ ^[0-9]+$ ]]; then
        attempts="$1"
        shift
    fi
    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    if (( attempts < 1 )); then
        echo "::error::fl_net_retry: attempts must be >= 1, got '${attempts}'" >&2
        return 2
    fi
    local n=1 status delay
    until "$@"; do
        status=$?
        if (( n >= attempts )); then
            echo "::error::fl_net_retry: '$*' failed (exit ${status}) after ${n} attempt(s)" >&2
            return "${status}"
        fi
        delay=$(( n * FL_NET_BACKOFF_SECONDS ))
        echo "::warning::fl_net_retry: attempt ${n}/${attempts} of '$*' failed (exit ${status}); retrying in ${delay}s" >&2
        sleep "${delay}"
        n=$(( n + 1 ))
    done
}
