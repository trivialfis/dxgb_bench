#!/usr/bin/env bash

set -e

run_bench() {
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    dxgb-ext-bench --task=ext-qdm --n_rounds=128 --device=cuda --n_bins=256 --verbosity=2 --loadfrom="./data"
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    dxgb-ext-bench --task=ext-qdm --n_rounds=128 --device=cuda --n_bins=256 --verbosity=2 --loadfrom="./data" --valid
}

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "DG f512,sparsity=0.6"
dxgb-bench datagen --n_samples_per_batch=4194304 --n_features=512 --sparsity=0.6 --n_batches=64 --device=cuda --saveto="./data"
echo "Run f512,sparsity=0.6"
run_bench

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "DG f256,sparsity=0.6"
dxgb-bench datagen --n_samples_per_batch=8388608 --n_features=256 --sparsity=0.6 --n_batches=64 --device=cuda --saveto="./data"
echo "Run f256,sparsity=0.6"
run_bench

sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
echo "DG f1024,sparsity=0.6"
dxgb-bench datagen --n_samples_per_batch=2097152 --n_features=1024 --sparsity=0.6 --n_batches=64 --device=cuda --saveto="./data"
echo "Run f1024,sparsity=0.6"
run_bench
