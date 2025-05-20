# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

from itertools import product

import pytest

from dxgb_bench.dxgb_bench import bench, datagen
from dxgb_bench.testing import Device, TmpDir, devices, formats
from dxgb_bench.utils import Timer


def make_data(saveto: list[str], device: Device, fmt: str) -> None:
    n_samples_per_batch = 256
    n_features = 16
    n_batches = 8
    assparse = False
    target_type = "reg"

    # Call the datagen function with test parameters
    datagen(
        n_samples_per_batch=n_samples_per_batch,
        n_features=n_features,
        n_batches=n_batches,
        assparse=assparse,
        target_type=target_type,
        sparsity=0.0,
        device=device,
        outdirs=saveto,
        fmt=fmt,
    )


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_bench_qdm(device: Device, fmt: str) -> None:
    with TmpDir(2, True) as outdirs:
        make_data(outdirs, device, fmt)
        # Call the bench function with test parameters
        params = {"device": device}
        bench(
            task="qdm",
            loadfrom=outdirs,
            params=params,
            n_rounds=8,
            valid=True,
            device=device,
        )
        timer = Timer.global_timer()
        assert timer["Qdm"]["Train-DMatrix"] > 0
        assert timer["Qdm"]["Valid-DMatrix"] > 0
        assert timer["Qdm"]["train"] > 0


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_bench_iter(device: Device, fmt: str) -> None:
    with TmpDir(2, True) as outdirs:
        make_data(outdirs, device, fmt)
        # Call the bench function with test parameters
        params = {"device": device}
        bench(
            task="qdm-iter",
            loadfrom=outdirs,
            params=params,
            n_rounds=8,
            valid=True,
            device=device,
        )
        timer = Timer.global_timer()
        assert timer["Qdm"]["Train-DMatrix-Iter"] > 0
        assert timer["Qdm"]["Valid-DMatrix-Iter"] > 0
        assert timer["Qdm"]["train"] > 0
