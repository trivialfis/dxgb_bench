#!/usr/bin/env bash

set -euox pipefail

DXGB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# CMake supplies the CUDA host compiler itself.
unset NVCC_PREPEND_FLAGS
pip install -e "${DXGB_ROOT}" --no-deps --no-build-isolation \
    --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=RelWithDebInfo" \
    --config-settings=cmake.args="-DCMAKE_CUDA_ARCHITECTURES=${1:-all}"
