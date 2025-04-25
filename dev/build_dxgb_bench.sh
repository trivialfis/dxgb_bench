#!/usr/bin/env bash

mkdir build
cd build
cmake ../dxgb_bench/dxgb_bench -DCMAKE_BUILD_TYPE=RelWithDebInfo -DDXGB_USE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=all -GNinja
ninja
cd ../dxgb_bench
pip install -e . --no-deps --no-build-isolation
