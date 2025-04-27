# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import pytest
from distributed import Client

from dxgb_bench.dxgb_dist_bench import bench, local_cluster
from dxgb_bench.testing import Device, devices
from dxgb_bench.utils import Opts


@pytest.mark.parametrize("device", devices())
def test_dist(device: Device) -> None:
    params = {"device": device, "max_bin": 8, "max_depth": 2, "eta": 0.1}
    opts = Opts(
        n_samples_per_batch=256,
        n_features=128,
        n_batches=8,
        sparsity=0.0,
        on_the_fly=True,
        validation=False,
        device=device,
        target_type="reg",
        mr=None,
    )

    with local_cluster(device=device, n_workers=2) as cluster:
        with Client(cluster) as client:
            booster = bench(client, 8, opts, params, verbosity=1)
            assert booster.num_features() == opts.n_features
            assert booster.num_boosted_rounds() == 8
