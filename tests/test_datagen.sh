#!/usr/bin/env bash

DATADIRS="./_tdata0,./_tdata1"

dxgb-bench datagen --n_samples_per_batch=512 --n_batches=8 --n_features=256 --saveto=${DATADIRS}

dxgb-bench bench --device=cpu --task=qdm --valid --loadfrom=${DATADIRS}
dxgb-bench bench --device=cuda --task=qdm --valid --loadfrom=${DATADIRS}

dxgb-ext-bench --task=ext-qdm --device=cpu --valid --loadfrom=${DATADIRS}
dxgb-ext-bench --task=ext-qdm --device=cuda --valid --loadfrom=${DATADIRS}

rm -rf ./_tdata0
rm -rf ./_tdata1
