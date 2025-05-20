# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import numpy as np
import pytest
from distributed import Client
from xgboost.compat import concat

from dxgb_bench.dataiter import IterImpl, LoadIterStrip, StridedIter, SynIterImpl
from dxgb_bench.dxgb_bench import datagen
from dxgb_bench.dxgb_dist_bench import bench, local_cluster
from dxgb_bench.testing import Device, TmpDir, devices
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
        cache_host_ratio=None,
    )

    with local_cluster(device=device, n_workers=2) as cluster:
        with Client(cluster) as client:
            booster = bench(client, 8, opts, params, loadfrom=[], verbosity=1)
            assert booster.num_features() == opts.n_features
            assert booster.num_boosted_rounds() == 8


def strided_iter(
    device: Device, n_batches: int, it_impl: IterImpl, stride: int
) -> None:
    batches = []

    def callback(data: np.ndarray, label: np.ndarray) -> None:
        batches.append((data, label))

    for start in range(0, stride):
        it = StridedIter(
            it_impl,
            start=start,
            stride=stride,
            is_ext=True,
            is_valid=False,
            device=device,
        )
        it.reset()
        while it.next(callback):
            continue

    assert len(batches) == n_batches
    Xs, ys = zip(*batches)
    assert len(ys) == n_batches

    X = concat(Xs)
    if device == "cpu":
        values = np.unique(X)
    else:
        import cupy as cp

        values = cp.unique(X)

    r = values.size / X.size
    assert r > 0.9


@pytest.mark.parametrize("device", devices())
def test_strided_syn_iter(device: Device) -> None:
    n_batches = 5
    stride = 3
    n_samples_per_batch = 2048
    n_features = 256
    rs = 0

    it_impl = SynIterImpl(
        n_samples_per_batch=n_samples_per_batch,
        n_features=n_features,
        n_batches=n_batches,
        sparsity=0.0,
        assparse=False,
        target_type="reg",
        device=device,
        rs=rs,
    )

    strided_iter(device, n_batches, it_impl, stride)


@pytest.mark.parametrize("device", ["cpu"])
def test_strided_load_iter(device: Device) -> None:
    n_samples_per_batch = 256
    n_features = 16
    stride = 3
    n_batches = 5
    assparse = False
    target_type = "reg"

    # Call the datagen function with test parameters
    with TmpDir(2, True) as outdirs:
        datagen(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_batches=n_batches,
            assparse=assparse,
            target_type=target_type,
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )

        it_impl = LoadIterStrip(outdirs, False, 0.0, device)

        strided_iter(device, n_batches, it_impl, stride)
