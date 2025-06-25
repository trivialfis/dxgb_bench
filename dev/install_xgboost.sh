#!/usr/bin/env bash

INSTALL_XGBOOST=$1
SM=$2

echo "CONDA_ENV ${CONDA_PREFIX}"

if [[ -n ${INSTALL_XGBOOST} && -n ${SM} ]]; then
    cd /ws
    rm -rf ./build
    mkdir build
    cd build
    cmake -GNinja ../xgboost/ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	  -DUSE_CUDA=ON \
	  -DUSE_NCCL=ON \
	  -DUSE_DLOPEN_NCCL=ON \
	  -DUSE_OPENMP=ON \
	  -DUSE_NVTX=ON \
	  -DPLUGIN_RMM=ON \
	  -DUSE_NVCOMP=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$SM -DENABLE_ALL_WARNINGS=ON

    cd /ws/build && time ninja

    cd /ws/xgboost/python-package
    pip install . --no-deps --no-build-isolation
fi
