# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.

from __future__ import annotations

from itertools import product

import pytest

from dxgb_bench.dxgb_bench import datagen
from dxgb_bench.dxgb_ext_bench import qdm_train
from dxgb_bench.external_mem import Opts
from dxgb_bench.testing import Device, TmpDir, devices, formats
from dxgb_bench.utils import Timer


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_qdm_train(device: Device, fmt: str) -> None:
    Timer.reset()
    timer = Timer.global_timer()

    params = {"device": device, "max_bin": 8, "max_depth": 2, "eta": 0.1}

    n_samples_per_batch = 256
    n_features = 128
    n_batches = 8

    opts = Opts(
        n_samples_per_batch=n_samples_per_batch,
        n_features=n_features,
        n_targets=1,
        n_batches=n_batches,
        sparsity=0.0,
        on_the_fly=False,
        validation=True,
        device=device,
        mr=None,
        target_type="reg",
        cache_host_ratio=None,
        min_cache_page_bytes=None,
    )

    with TmpDir(2, True) as outdirs:
        datagen(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_targets=1,
            n_batches=n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt=fmt,
        )

        booster, results = qdm_train(opts, params, 8, outdirs)

    assert booster.num_features() == n_features
    assert booster.num_boosted_rounds() == 8

    assert timer["Train"]["Train"] > 0
    assert timer["Train"]["DMatrix-Train"] > 0
    assert "opts" in results
