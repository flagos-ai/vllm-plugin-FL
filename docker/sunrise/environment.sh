#!/bin/bash
# Source the published image runtime/compiler environment in every build layer
# and interactive invocation; retain caller-selected visible devices.
for sunrise_profile in /etc/profile.d/runtime_env.sh /etc/profile.d/zz-flagos.sh; do
    if [ -f "${sunrise_profile}" ]; then
        # Vendor profile scripts are supplied by the base image.
        # shellcheck source=/dev/null
        source "${sunrise_profile}"
    fi
done
unset sunrise_profile
# Reuse the image's /flagos environment: a nested --system-site-packages venv
# cannot inherit the parent venv's vendor torch_ptpu packages.
sunrise_python=$(command -v "${SUNRISE_PYTHON:-/flagos/bin/python}")
PATH="$(dirname "${sunrise_python}"):${PATH}"
export PATH
VIRTUAL_ENV=$("${sunrise_python}" -c 'import sys; print(sys.prefix)')
export VIRTUAL_ENV
unset sunrise_python
