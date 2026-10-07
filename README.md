Some scripts for running benchmarks with XGBoost.

Container image
---------------
From the repository root, run

``` sh
python dev/build_image.py --arch=x86 --sm=89 --install-xgboost
```

Run `python dev/build_image.py --help` for more options.

Building from source
--------------------
Install the Python package and compile its native data generator in one step:

``` sh
pip install .
```

Pass CMake options using `cmake.args` (repeatable), for example:

``` sh
pip install . --config-settings=cmake.args="-DCMAKE_CUDA_ARCHITECTURES=89"
pip install -e . --config-settings=cmake.args="-DCMAKE_BUILD_TYPE=Debug"
```

`CMAKE_ARGS` also accepts space-separated CMake options; pip's `cmake.args`
options take precedence. Builds use `build/hatch`, and honor `CMAKE_GENERATOR`
and `CMAKE_BUILD_PARALLEL_LEVEL`. Editable installs also build the native library;
rerun the install after changing C++ or CUDA sources.

To produce an sdist and a wheel rebuilt from that sdist:

``` sh
pip install build
python -m build
```

Bare-metal and container builds share [pixi.toml](pixi.toml). Build the C++ data
generator and install the benchmark with:

``` sh
git clone https://github.com/trivialfis/dxgb_bench.git
cd dxgb_bench
pixi install
pixi run build 89
pixi shell
```

`pixi run build` defaults to all CUDA architectures. To use a separately built
XGBoost checkout, select the dependency-only `dev` environment instead:

``` sh
pixi install --environment dev
pixi run --environment dev pip install /path/to/xgboost/python-package --no-deps --no-build-isolation
pixi run --environment dev build 89
pixi shell --environment dev
```

Synthetic data
--------------
For both the batched `datagen` and the data iterator, the output should be consistent for
different number of batches and for different devices. For example, generating 2 batches
with 1024 samples for each batch should produce the exact same result as generating a
single batch with 2048 samples. When compiled with CUDA, CPU and GPU output should match
each other. The benchmark script can synthesize data on-the-fly when running out-of-core
training, which helps us to avoid storage issues during development.

Examples
--------

- Run datagen:
``` sh
dxgb-bench datagen --n_samples_per_batch=4194304 --n_batches=4 --n_features=512 --device=cpu --fmt=npy
```

- Generate imbalanced feature groups (3072 binary columns followed by 1008 standard
  normal columns) with four regression targets:
``` sh
dxgb-bench datagen --n_samples_per_batch=131072 --n_batches=4 --n_features=4080 --n_binary=3072 --n_targets=4 --device=cpu --fmt=npy --saveto=imb-data
```

`--n_binary` enables the mixed-feature generator. It accepts zero through
`--n_features`, supports CPU and CUDA, and requires dense regression data with zero
sparsity. Targets follow a fixed linear model with independent unit normal noise;
changing the batch size or target count does not change the features. `--data_seed`
defaults to 2026 in this mode. Rebuild the native library when updating the generator.

- Reuse stored features and generate new regression targets:
``` sh
dxgb-bench datagen --loadfrom=imb-data --saveto=imb-targets --n_targets=4 --data_seed=2027 --device=cpu
```

This reads the existing batch shapes and storage format, links the source `X`
directories, and writes new `y` strips. Source and destination must have the same
number of shard directories; comma-separated paths use the usual sorted shard order.
Keep the source features available while using the new dataset. The default seed for
replacement targets is 2027. Both modes require destinations without existing `X` or
`y` entries to prevent mixing old and new data.

`imb/gen.py` and `imb/gen_y.py` also call these backend functions. Their random samples
differ from the original scripts, and batches now share one fixed coefficient matrix.

- Run training with the generated data:
``` sh
dxgb-bench bench --task=qdm --n_rounds=10
```

- Run external memory test with data synthesized on the fly:
``` sh
dxgb-bench bench --fly --n_samples_per_batch=2097152 --n_features=256 --n_batches=8 --device=cuda --task=ext-qdm-iter --n_rounds=8 --verbosity=1 --mr=arena
```

| Task           | Matrix                    | Input                                    |
|----------------|---------------------------|------------------------------------------|
| `qdm`          | In-core `QuantileDMatrix` | Load and concatenate all stored batches  |
| `qdm-iter`     | In-core `QuantileDMatrix` | Iterate over stored or generated batches |
| `ext-qdm-iter` | `ExtMemQuantileDMatrix`   | Iterate over stored or generated batches |
| `ext-dm-iter`  | External-memory `DMatrix` | Iterate over stored or generated batches |

All tasks share training options, `--valid`, and `--model_path`. Quantile matrices require
`--tree_method=hist` (or `auto`); `ext-dm-iter` also supports `approx`.  `--fly` requires
an iterator task and a positive `--n_samples_per_batch`; `--n_features` defaults
to 512. Without `--fly`, shapes and batch counts come from the stored data. Binary
classification accepts stored binary labels as well as generated data. `--assparse` and
`--fmt` belong to `datagen`; benchmark loading detects the stored format. The
`ext-dm-iter` task accepts dense inputs too.

- Run external memory test on a distributed system (SNMG) with data synthesized on the fly:
``` sh
dxgb-dist-bench --n_workers=4 --cluster_type=local --fly --mr=arena --n_samples_per_batch=4194304 --n_features=512 --n_batches=196 --device=cuda --n_rounds=128 --verbosity=2
```

Kvikio
------
We use kvikio for data IO, some environment variables might be useful:

``` sh
export KVIKIO_NTHREADS=8
export KVIKIO_COMPAT_MODE=1
```

Commands
--------
- dxgb-bench
- dxgb-datasets
- dxgb-dist-bench

Run `${COMMAND} --help` for more info, including file formats, where to save the synthetic
data, hyper-parameters, etc.

There are some additional utilities like the RMM log parser:

``` sh
dxgb-bench rmmpeak --path=/bench/rmm_log.dev0
```

Public datasets
---------------

`dxgb_bench.datasets.public` provides a reusable pipeline for public numerical and
categorical benchmark datasets. It downloads original sources atomically and preserves
categorical columns as pandas categories. Prepared DataFrames are stored as Parquet;
numerical arrays, labels, and split arrays use memory-mapped NumPy files. Metadata records
source hashes, citations, licenses, feature information, and split semantics.

Use either the dedicated command or the main command's `datasets` subcommand:

``` sh
dxgb-datasets --list
dxgb-datasets covertype poker_hand
dxgb-bench datasets anneal congressional_voting
dxgb-bench datasets --download-only aloi
dxgb-bench datasets --validate-only --offline covertype
```

The default cache is `${XDG_CACHE_HOME:-~/.cache}/dxgb_bench/datasets`. Set
`DXGB_BENCH_DATASET_CACHE` or pass `--cache-dir` to use a shared location.

The stages are also available through Python:

``` python
from dxgb_bench.datasets.public import PublicDatasetPipeline

pipeline = PublicDatasetPipeline()
dataset = pipeline.ensure("covertype")
print(dataset.X.shape, dataset.y.shape)
```

The result of a test is saved into a JSON file under the working directory. An example output from in-core training:

<details>

<summary>Example output</summary>

``` json
{
  "opts": {
    "n_samples_per_batch": 32768,
    "n_features": 512,
    "n_batches": 1,
    "sparsity": 0.0,
    "on_the_fly": false,
    "validation": false,
    "device": "cuda",
    "mr": null,
    "target_type": "reg",
    "cache_host_ratio": null,
    "tree_method": "hist",
    "max_depth": 6,
    "grow_policy": "depthwise",
    "subsample": null,
    "colsample_bynode": null,
    "colsample_bytree": null,
    "max_bin": 256,
    "lambda": null,
    "gamma": null,
    "eta": null,
    "min_child_weight": null,
    "verbosity": 1,
    "objective": null,
    "n_rounds": 2,
    "n_workers": 1
  },
  "timer": {
    "load-batches": {
      "load": 0.8841826915740967
    },
    "load-all": {
      "concat": 0.017390012741088867
    },
    "Train": {
      "DMatrix-Train": 0.0640714168548584,
      "Train": 0.982398271560669
    }
  },
  "evals": {
    "Train": {
      "rmse": [
        33.3493520474823,
        32.88331998446392
      ]
    }
  },
  "machine": {
    "system": "Linux",
    "arch": "x86_64",
    "cpus": 24,
    "gpus": [
      "NVIDIA GeForce RTX 4070 Ti SUPER",
      "NVIDIA GeForce RTX 4070 Ti SUPER"
    ],
    "drivers": [
      "570.124.06",
      "570.124.06"
    ],
    "c2c": null
  },
  "version": {
    "dxgb_bench": "0.1.dev345+g77eabb5",
    "xgboost": "3.1.0-dev-ab24a469d"
  }
}
```

</details>
