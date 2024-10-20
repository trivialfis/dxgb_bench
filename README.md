Some simple scripts for running benchmarks for [XGBoost](https://github.com/dmlc/xgboost).  Heavily inspired by [gbm-bench](https://github.com/NVIDIA/gbm-bench).

To build a CPU image:
``` sh
./dxgb_bench/dev/build_image.sh cpu
```

Synthetic data
--------------
There are two implementations for the dense data generator, one in C++/CUDA, another one
in Python. To use the one in C++, we need to build it using CMake first. One can use the
`dxgb-bench` to generate the data, or generate them on the fly with the external memory
version of XGBoost.

For both the batched `datagen` and the data iterator, the output should be consistent for
different number of batches and for different devices. For example, generating 2 batches
with 1024 samples for each batch should produce the exact same result as generating a
single batch with 2048 samples. When compiled with CUDA, CPU and GPU output should match
each other.

Commands
--------
- dxgb-bench
- dxgb-dask-bench
- dxgb-ext-bench