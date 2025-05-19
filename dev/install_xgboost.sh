INSTALL_XGBOOST=$1

if [[ -n ${INSTALL_XGBOOST} ]]; then
    cd /ws && \
    rm -rf ./build && \
    mkdir build && \
    cd build && \
    cmake -GNinja ../xgboost/ -DCMAKE_BUILD_TYPE=RelWithDebInfo \
	  -DUSE_CUDA=ON \
	  -DUSE_NCCL=ON \
	  -DUSE_DLOPEN_NCCL=ON \
	  -DUSE_OPENMP=ON \
	  -DUSE_NVTX=ON \
	  -DPLUGIN_RMM=ON \
	  -DCMAKE_CUDA_ARCHITECTURES=$SM -DENABLE_ALL_WARNINGS=ON
fi
