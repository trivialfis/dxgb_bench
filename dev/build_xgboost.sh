#!/usr/bin/env bash

SM=$1

cd ws
mkdir build
cd build
cmake -GNinja ../xgboost/ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DUSE_CUDA=ON \
      -DUSE_NCCL=ON \
      -DUSE_DLOPEN_NCCL=ON \
      -DUSE_OPENMP=ON \
      -DUSE_NVTX=ON \
      -DUSE_NVCOMP=ON \
      -DPLUGIN_RMM=ON \
      -DCMAKE_CUDA_ARCHITECTURES=SM -DENABLE_ALL_WARNINGS=ON

time ninja && \
    cd ../xgboost/python-package && \
    pip install . --no-deps --no-build-isolation && \
    cd ../../ && \
    rm -rf build
