#!/usr/bin/env bash

SM=$1

git clone --recursive  https://github.com/dmlc/xgboost.git
cd xgboost
git checkout f41be2cef10eaaa450271016355c41e3a9125502
cd -


mkdir build
cd build
cmake -GNinja ../xgboost/ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DUSE_CUDA=ON \
      -DUSE_NCCL=ON \
      -DUSE_DLOPEN_NCCL=ON \
      -DUSE_OPENMP=ON \
      -DUSE_NVTX=ON \
      -DPLUGIN_RMM=ON \
      -DCMAKE_CUDA_ARCHITECTURES=SM -DENABLE_ALL_WARNINGS=ON

time ninja && \
    cd ../xgboost/python-package && \
    pip install . --no-deps --no-build-isolation && \
    cd ../../ && \
    rm -rf build
