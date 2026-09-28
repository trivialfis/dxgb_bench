#!/usr/bin/env bash

set -eo pipefail
source /etc/profile.d/pixi.sh
unset NVCC_PREPEND_FLAGS
exec "$@"
