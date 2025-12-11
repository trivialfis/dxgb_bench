# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

from itertools import product

import pytest

from dxgb_bench.dxgb_bench import bench, datagen
from dxgb_bench.testing import Device, TmpDir, devices, formats
from dxgb_bench.utils import Opts, Timer


def make_data(saveto: list[str], device: Device, fmt: str) -> Opts:
    n_samples_per_batch = 256
    n_features = 16
    n_batches = 8
    assparse = False
    target_type = "reg"

    # Call the datagen function with test parameters
    datagen(
        n_samples_per_batch=n_samples_per_batch,
        n_features=n_features,
        n_targets=1,
        n_batches=n_batches,
        assparse=assparse,
        target_type=target_type,
        sparsity=0.0,
        device=device,
        outdirs=saveto,
        fmt=fmt,
    )

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
    return opts


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_bench_qdm(device: Device, fmt: str) -> None:
    Timer.reset()
    with TmpDir(2, True) as outdirs:
        opts = make_data(outdirs, device, fmt)
        # Call the bench function with test parameters
        params = {"device": device, "max_bin": 256}
        bench(
            task="qdm",
            loadfrom=outdirs,
            model_path=None,
            params=params,
            opts=opts,
            n_rounds=8,
        )
        timer = Timer.global_timer()
        assert timer["Train"]["DMatrix-Train"] > 0
        assert timer["Train"]["DMatrix-Valid"] > 0
        assert timer["Train"]["Train"] > 0


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_bench_iter(device: Device, fmt: str) -> None:
    Timer.reset()
    with TmpDir(2, True) as outdirs:
        opts = make_data(outdirs, device, fmt)
        # Call the bench function with test parameters
        params = {"device": device, "max_bin": 256}
        bench(
            task="qdm-iter",
            loadfrom=outdirs,
            model_path=None,
            params=params,
            opts=opts,
            n_rounds=8,
        )
        timer = Timer.global_timer()
        assert timer["Train"]["DMatrix-Train"] > 0
        assert timer["Train"]["DMatrix-Valid"] > 0
        assert timer["Train"]["Train"] > 0


@pytest.mark.parametrize("device", devices())
def test_bench_iter_fly(device: Device) -> None:
    Timer.reset()
    timer = Timer.global_timer()

    n_samples_per_batch = 256
    n_features = 128
    n_batches = 8

    opts = Opts(
        n_samples_per_batch=n_samples_per_batch,
        n_features=n_features,
        n_targets=1,
        n_batches=n_batches,
        sparsity=0.0,
        on_the_fly=True,
        validation=True,
        device=device,
        mr=None,
        target_type="reg",
        cache_host_ratio=None,
        min_cache_page_bytes=None,
    )

    params = {"device": device, "max_bin": 256}
    bench(
        task="qdm-iter",
        loadfrom=[],
        model_path=None,
        params=params,
        opts=opts,
        n_rounds=8,
    )

    assert timer["Train"]["DMatrix-Train"] > 0
    assert timer["Train"]["DMatrix-Valid"] > 0
    assert timer["Train"]["Train"] > 0
