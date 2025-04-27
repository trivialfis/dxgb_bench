# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import os
import tempfile
from itertools import product

import cupy as cp
import numpy as np
import pytest
from scipy import sparse
from xgboost.compat import concat

from dxgb_bench.dataiter import (
    BenchIter,
    LoadIterStrip,
    SynIterImpl,
    get_valid_sizes,
    load_all,
)
from dxgb_bench.datasets.generated import make_dense_regression, make_sparse_regression
from dxgb_bench.dxgb_bench import datagen
from dxgb_bench.strip import make_file_name, make_strips
from dxgb_bench.testing import TmpDir, assert_array_allclose, devices, formats, has_cuda


def test_sparse_regressioin() -> None:
    X, y = make_sparse_regression(
        n_samples=3, n_features=2, sparsity=0.6, random_state=0
    )
    assert isinstance(X, sparse.csr_matrix)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 3 and X.shape[1] == 2
    assert y.shape[0] == X.shape[0]
    if len(y.shape) == 2:  # couldn't squeeze vstack result for some reason
        assert y.shape[1] <= 1

    X, y = make_sparse_regression(
        n_samples=1023, n_features=32, sparsity=0.6, random_state=0
    )
    assert X.shape[0] == 1023 and X.shape[1] == 32
    assert y.shape[0] == X.shape[0]
    # 1023 * 32 * 0.6 -> 13094
    assert 13000 < X.nnz < 13110


def test_dense_regression() -> None:
    X, y = make_dense_regression(
        n_samples=3,
        n_features=2,
        sparsity=0.6,
        device="cpu",
        random_state=1,
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 3 and X.shape[1] == 2
    assert y.shape[0] == X.shape[0]

    X, y = make_dense_regression(
        n_samples=2047,
        n_features=16,
        sparsity=0.6,
        device="cpu",
        random_state=1,
    )
    assert X.shape[0] == 2047 and X.shape[1] == 16
    assert y.shape[0] == X.shape[0]
    nnz = np.count_nonzero(~np.isnan(X))
    assert 13000 < nnz < 13230
    nnz = np.count_nonzero(~np.isnan(y))
    assert nnz == 2047


def run_dense_batches(device: str) -> tuple[np.ndarray, np.ndarray]:
    n_features = 3
    n_batches = 12
    nspb = 8

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(
            nspb,
            n_features,
            n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=[path],
            fmt="npy",
        )
        X0, y0 = load_all([path], "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(
            nspb * n_batches,
            n_features,
            1,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=[path],
            fmt="npy",
        )
        X1, y1 = load_all([path], "cpu")

    np.testing.assert_allclose(X0, X1)
    np.testing.assert_allclose(y0, y1)
    return X0, y0


@pytest.mark.skipif(reason="No CUDA.", condition=not has_cuda())
def test_dense_batches() -> None:
    X0, y0 = run_dense_batches("cpu")
    X1, y1 = run_dense_batches("cuda")
    np.testing.assert_allclose(X0, X1, rtol=1e-6)
    np.testing.assert_allclose(y0, y1, rtol=5e-6)


def run_dense_iter(device: str) -> tuple[np.ndarray, np.ndarray]:
    n_features = 4
    n_batches = 12
    nspb = 8

    impl = SynIterImpl(
        nspb,
        n_features,
        n_batches,
        0.0,
        False,
        target_type="reg",
        device=device,
    )
    Xs0, ys0 = [], []
    Xs1, ys1 = [], []
    for i in range(n_batches):
        X, y = impl.get(i)
        Xs0.append(X)
        ys0.append(y)

    for i in range(n_batches):
        X, y = impl.get(i)
        Xs1.append(X)
        ys1.append(y)

    X0 = concat(Xs0)
    y0 = concat(ys0)

    X1 = concat(Xs1)
    y1 = concat(ys1)

    impl = SynIterImpl(
        nspb * n_batches, n_features, 1, 0.0, False, target_type="reg", device=device
    )
    X2, y2 = impl.get(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(
            nspb,
            n_features,
            n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=[path],
            fmt="npy",
        )
        X3, y3 = load_all([path], "cpu")

    assert_array_allclose(X0, X1)
    assert_array_allclose(X0, X2)
    assert_array_allclose(X0, X3)

    assert_array_allclose(y0, y1)
    assert_array_allclose(y0, y2)
    assert_array_allclose(y0, y3)

    return X0, y0


@pytest.mark.skipif(reason="No CUDA.", condition=not has_cuda())
def test_dense_iter() -> None:
    X0, y0 = run_dense_iter("cpu")
    X1, y1 = run_dense_iter("cuda")
    assert_array_allclose(X0, X1, rtol=5e-6)
    assert_array_allclose(y0, y1, rtol=5e-6)


@pytest.mark.parametrize("device", devices())
def test_deterministic(device: str) -> None:
    n_samples_per_batch = 8192
    n_features = 400
    target_type = "bin"
    n_batches = 4

    impl = SynIterImpl(
        n_samples_per_batch,
        n_features,
        n_batches,
        0.0,
        False,
        target_type=target_type,
        device=device,
    )
    it = BenchIter(impl, True, False, device)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    def append(data: np.ndarray, label: np.ndarray) -> None:
        Xs.append(data)
        ys.append(label)

    while it.next(append):
        continue
    it.reset()

    k = 0

    def check(data: np.ndarray, label: np.ndarray) -> None:
        nonlocal k

        if device == "cpu":
            np.testing.assert_allclose(data, Xs[k])
            np.testing.assert_allclose(label, ys[k])
        else:
            cp.testing.assert_allclose(data, Xs[k])
            cp.testing.assert_allclose(label, ys[k])

        k += 1

    while it.next(check):
        continue
    it.reset()


@pytest.mark.parametrize("device", devices())
def test_cv(device: str) -> None:
    with TmpDir(2, True) as outdirs:
        # Within batch read
        n_features = 2
        n_batches = 4
        nspb = 8

        datagen(
            nspb,
            n_features,
            n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )
        n_train, n_valid = get_valid_sizes(n_samples=nspb * n_batches)
        assert n_valid == 6

        # files = get_file_paths(outdirs)
        # impl = LoadIterImpl(list(zip(files[0], files[1])), True, False, device)
        impl = LoadIterStrip(outdirs, is_valid=False, test_size=0.2, device=device)
        # assert len(impl.X_shards) == n_batches

        X, y = load_all(outdirs, device)

        prev = 0
        for i in range(impl.n_batches):
            X_i, y_i = impl.get(i)
            assert_array_allclose(X[prev : prev + nspb - 1], X_i)
            assert_array_allclose(y[prev : prev + nspb - 1], y_i)
            prev += nspb


@pytest.mark.parametrize("device", devices())
def test_datagen(device: str) -> None:
    n_shards = 2
    with TmpDir(n_shards, True) as outdirs:
        n_features = 4
        n_batches = 8
        nspb = 16

        datagen(
            nspb,
            n_features,
            n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )

        for shard_idx, d in enumerate(outdirs):
            Xs, ys = [], []
            for b in range(n_batches):
                fname = make_file_name(
                    (nspb, n_features),
                    "X",
                    "X",
                    batch_idx=b,
                    shard_idx=shard_idx,
                    fmt="npy",
                )
                X = np.load(os.path.join(d, fname))
                assert X.shape == (nspb // n_shards, n_features)
                Xs.append(X)

                fname = make_file_name(
                    (nspb, 1),
                    "y",
                    "y",
                    batch_idx=b,
                    shard_idx=shard_idx,
                    fmt="npy",
                )
                y = np.load(os.path.join(d, fname))
                assert y.shape[0] == nspb // n_shards
                assert y.shape[0] == y.size
                ys.append(y)

            # Must be unique, not guaranteed, just unlikely to have same floating
            # values.
            for i in range(1, n_batches):
                assert not (Xs[0] == Xs[i]).any()
                assert not (ys[0] == ys[i]).any()

        X, y = load_all(outdirs, device)
        assert X.shape[0] == y.shape[0] == nspb * n_batches


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_load_all(device: str, fmt: str) -> None:
    n_shards = 2
    with TmpDir(n_shards, True) as outdirs:
        X_fd, y_fd = make_strips(["X", "y"], outdirs, fmt=fmt, device=device)
        X = np.arange(0, 64, dtype=np.float32).reshape(8, 8)
        y = X.sum(axis=1)
        X_fd.write(X, 0)
        y_fd.write(y, 0)

        X_res, y_res = load_all(outdirs, device=device)
        assert_array_allclose(X, X_res.squeeze())
        assert_array_allclose(y, y_res.squeeze())
