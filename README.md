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

Commands
--------
- dxgb-bench
- dxgb-dask-bench
- dxgb-ext-bench