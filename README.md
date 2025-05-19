Synthetic data
--------------
To use the data gen written C++, we need to build it using CMake first. One can use the
`dxgb-bench` to generate the data, or generate them on the fly with the external memory
version of XGBoost.

``` sh
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURS=89 -GNinja
ninja
cd ../
pip install -e . --no-build-isolation --no-deps
```

For both the batched `datagen` and the data iterator, the output should be consistent for
different number of batches and for different devices. For example, generating 2 batches
with 1024 samples for each batch should produce the exact same result as generating a
single batch with 2048 samples. When compiled with CUDA, CPU and GPU output should match
each other.

Container image
---------------
One level above the `dxgb_bench` directory, run

``` sh
python ./dxgb_bench/dev/build_image.py --target=gpu --arch=x86 --sm=89 --install-xgboost
```

Examples
--------

Run datagen:
``` sh
dxgb-bench datagen --n_samples_per_batch=4194304 --n_batches=4 --n_features=512 --device=cpu --fmt=npy
```

Run external memory test with synthesized on the fly.
``` sh
dxgb-ext-bench --fly --n_samples_per_batch=2097152 --n_features=256 --n_batches=8 --device=cuda --task=ext-qdm --n_rounds=8 --verbosity=1 --mr=arena
```

Commands
--------
- dxgb-bench
- dxgb-dist-bench
- dxgb-ext-bench