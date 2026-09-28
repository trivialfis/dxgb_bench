#!/usr/bin/env bash

set -euox pipefail

DXGB_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
# CMake supplies the CUDA host compiler itself.
unset NVCC_PREPEND_FLAGS
cmake -S "${DXGB_ROOT}/dxgb_bench" -B "${DXGB_ROOT}/build" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDXGB_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="${1:-all}" -GNinja
cmake --build "${DXGB_ROOT}/build"
cd "${DXGB_ROOT}"
pip install -e . --no-deps --no-build-isolation
