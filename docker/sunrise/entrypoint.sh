#!/bin/bash
set -e
# shellcheck source=docker/sunrise/environment.sh
source /opt/sunrise-build/environment.sh
exec "$@"
