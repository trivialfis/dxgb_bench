#!/usr/bin/env bash

dxgb-bench datagen --n_samples_per_batch=512 --n_batches=8 --n_features=256 --saveto=./_testing_data

dxgb-bench bench --device=cpu --task=qdm --valid --loadfrom=./_testing_data
dxgb-bench bench --device=cuda --task=qdm --valid --loadfrom=./_testing_data

dxgb-ext-bench --task=ext-qdm --device=cpu --valid --loadfrom=./_testing_data
dxgb-ext-bench --task=ext-qdm --device=cpu --valid --loadfrom=./_testing_data

rm -rf ./_testing_data
