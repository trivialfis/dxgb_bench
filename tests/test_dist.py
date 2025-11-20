# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest
from xgboost.compat import concat

from dxgb_bench.dataiter import IterImpl, LoadIterStrip, StridedIter, SynIterImpl
from dxgb_bench.dxgb_bench import datagen
from dxgb_bench.dxgb_dist_bench import bench
from dxgb_bench.dxgb_ext_bench import qdm_train
from dxgb_bench.testing import Chdir, Device, TmpDir, devices
from dxgb_bench.utils import Opts, Timer


@pytest.mark.parametrize("device,extmem", product(devices(), [True, False]))
def test_dist(device: Device, extmem: bool) -> None:
    params = {"device": device, "max_bin": 8, "max_depth": 2, "eta": 0.1}
    opts = Opts(
        n_samples_per_batch=256,
        n_features=128,
        n_targets=1,
        n_batches=8,
        sparsity=0.0,
        on_the_fly=True,
        validation=False,
        device=device,
        target_type="reg",
        mr=None,
        cache_host_ratio=None,
    )
    with TmpDir(1, True) as tmpdir:
        booster, _ = bench(
            tmpdir[0],
            n_rounds=8,
            opts=opts,
            params=params,
            n_workers=2,
            loadfrom=[],
            verbosity=1,
            is_extmem=extmem,
        )
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
        n_targets=1,
        n_batches=n_batches,
        sparsity=0.0,
        assparse=False,
        target_type="reg",
        device=device,
        rs=rs,
    )

    strided_iter(device, n_batches, it_impl, stride)


@pytest.mark.parametrize("device", devices())
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
            n_targets=1,
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


def get_keys(results: dict, opts: Opts) -> list[str]:
    stack = [results]
    keys = []
    while len(stack) != 0:
        obj = stack.pop()
        for k, v in obj.items():
            if isinstance(v, dict):
                stack.append(v)
            else:
                keys.append(k)
            if k == "n_samples_per_batch":
                assert v == opts.n_samples_per_batch
            elif k == "n_features":
                assert v == opts.n_features
            elif k == "n_batches":
                assert v == opts.n_batches
            elif k == "sparsity":
                assert np.allclose(v, opts.sparsity, rtol=1e-2)
    return keys


@pytest.mark.parametrize("device", devices())
def test_syn_json(device: Device) -> None:
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
        on_the_fly=True,
        validation=False,
        device=device,
        target_type="reg",
        mr=None,
        cache_host_ratio=None,
    )

    with TmpDir(n_dirs=1, delete=True) as tmpdirs, Chdir(tmpdirs[0]):
        Timer.reset()
        booster_0, results_0 = qdm_train(opts, params, 8, [])

        Timer.reset()
        booster_1, results_1 = bench(
            tmpdirs[0],
            8,
            opts,
            params,
            n_workers=2,
            loadfrom=[],
            verbosity=1,
            is_extmem=True,
        )

        keys_0 = get_keys(results_0, opts)
        keys_1 = get_keys(results_1, opts)
        assert keys_0 == keys_1

        with open("extmem-0.json", "r") as fd:
            results_0 = json.load(fd)
        with open("dist-0.json", "r") as fd:
            results_1 = json.load(fd)

        keys_0 = get_keys(results_0, opts)
        keys_1 = get_keys(results_1, opts)
        assert keys_0 == keys_1


@pytest.mark.parametrize("device", devices())
def test_load_json(device: Device) -> None:
    params = {"device": device, "max_bin": 8, "max_depth": 2, "eta": 0.1}

    n_samples_per_batch = 256
    n_features = 128
    n_batches = 8
    sparsity = 0.3

    with TmpDir(n_dirs=2, delete=True) as tmpdirs, Chdir(tmpdirs[0]):
        datagen(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_targets=1,
            n_batches=n_batches,
            assparse=False,
            target_type="reg",
            sparsity=sparsity,
            device=device,
            outdirs=tmpdirs,
            fmt="npy",
        )

        opts = Opts(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_targets=1,
            n_batches=n_batches,
            sparsity=sparsity,
            on_the_fly=True,
            validation=False,
            device=device,
            target_type="reg",
            mr=None,
            cache_host_ratio=None,
        )

        Timer.reset()
        booster_0, results_0 = qdm_train(opts, params, 8, tmpdirs)

        Timer.reset()
        booster_1, results_1 = bench(
            tmpdirs[0],
            8,
            opts,
            params,
            n_workers=2,
            loadfrom=[],
            verbosity=1,
            is_extmem=True,
        )

        keys_0 = get_keys(results_0, opts)
        keys_1 = get_keys(results_1, opts)
        assert keys_0 == keys_1

        with open("extmem-0.json", "r") as fd:
            results_0 = json.load(fd)
        with open("dist-0.json", "r") as fd:
            results_1 = json.load(fd)

        keys_0 = get_keys(results_0, opts)
        keys_1 = get_keys(results_1, opts)
        assert keys_0 == keys_1
