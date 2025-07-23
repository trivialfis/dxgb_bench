Some scripts for running benchmarks with XGBoost.

Container image
---------------
One level above the `dxgb_bench` directory, run

``` sh
python ./dxgb_bench/dev/build_image.py --target=gpu --arch=x86 --sm=89 --install-xgboost
```

Run
``` sh
python ./dxgb_bench/dev/build_image.py --help
```
for more options.

Building from source
--------------------
To use the data gen written C++, we need to build it using CMake first. One can use the
`dxgb-bench` to generate the data, or generate them on the fly with the external memory
version of XGBoost.

``` sh
git clone https://github.com/trivialfis/dxgb_bench.git
cd dxgb_bench
mkdir build && cd build
cmake ../dxgb_bench -DCMAKE_CUDA_ARCHITECTURS=89 -GNinja
ninja
cd ../
pip install -e . --no-build-isolation --no-deps
```

Synthetic data
--------------
For both the batched `datagen` and the data iterator, the output should be consistent for
different number of batches and for different devices. For example, generating 2 batches
with 1024 samples for each batch should produce the exact same result as generating a
single batch with 2048 samples. When compiled with CUDA, CPU and GPU output should match
each other.

Examples
--------

Run datagen:
``` sh
dxgb-bench datagen --n_samples_per_batch=4194304 --n_batches=4 --n_features=512 --device=cpu --fmt=npy
```

Run external memory test with synthesized on the fly:
``` sh
dxgb-ext-bench --fly --n_samples_per_batch=2097152 --n_features=256 --n_batches=8 --device=cuda --task=ext-qdm --n_rounds=8 --verbosity=1 --mr=arena
```

Run external memory test on a distributed system (SNMG):
``` sh
dxgb-dist-bench --n_workers=4 --cluster_type=local --fly --mr=arena --n_samples_per_batch=4194304 --n_features=512 --n_batches=196 --device=cuda --n_rounds=128 --verbosity=2
```

Commands
--------
- dxgb-bench
- dxgb-dist-bench
- dxgb-ext-bench

Run `${COMMAND} --help` for more info.

Python dependencies
-------------------

The yml files contain dependencies that are not strictly necessary for running the
commands. I use them for docker build, which requires the entire tool chain to compile
XGBoost and dxgb-bench C++ code.

## Build time
- setuptools
- setuptools-scm

## Run time (CUDA)
- nvml (python)
- cupy
- numpy
- xgboost
- pandas
- tqdm
- packaging
- typing_extensions
- scipy