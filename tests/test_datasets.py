# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import os
import tempfile

import cupy as cp
import numpy as np
import pytest
from scipy import sparse
from xgboost.compat import concat

from dxgb_bench.dataiter import (
    LoadIterImpl,
    SynIterImpl,
    get_file_paths,
    get_valid_sizes,
    load_all,
)
from dxgb_bench.datasets.generated import make_dense_regression, make_sparse_regression
from dxgb_bench.dxgb_bench import datagen


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
        datagen(nspb, n_features, n_batches, False, 0.0, device, [path])
        X0, y0 = load_all([path], "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(nspb * n_batches, n_features, 1, False, 0.0, device, [path])
        X1, y1 = load_all([path], "cpu")

    np.testing.assert_allclose(X0, X1)
    np.testing.assert_allclose(y0, y1)
    return X0, y0


def test_dense_batches() -> None:
    X0, y0 = run_dense_batches("cpu")
    X1, y1 = run_dense_batches("cuda")
    np.testing.assert_allclose(X0, X1, rtol=1e-6)
    np.testing.assert_allclose(y0, y1, rtol=5e-6)


def assert_allclose(
    a: np.ndarray | cp.ndarray, b: np.ndarray | cp.ndarray, rtol: float = 1e-7
) -> None:
    if hasattr(a, "get"):
        a = a.get()
    if hasattr(b, "get"):
        b = b.get()
    np.testing.assert_allclose(a, b, rtol=rtol)


def run_dense_iter(device: str) -> tuple[np.ndarray, np.ndarray]:
    n_features = 4
    n_batches = 12
    nspb = 8

    impl = SynIterImpl(nspb, n_features, n_batches, 0.0, False, device)
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

    impl = SynIterImpl(nspb * n_batches, n_features, 1, 0.0, False, device)
    X2, y2 = impl.get(0)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(nspb, n_features, n_batches, False, 0.0, device, [path])
        X3, y3 = load_all([path], "cpu")

    assert_allclose(X0, X1)
    assert_allclose(X0, X2)
    assert_allclose(X0, X3)

    assert_allclose(y0, y1)
    assert_allclose(y0, y2)
    assert_allclose(y0, y3)

    return X0, y0


def test_dense_iter() -> None:
    X0, y0 = run_dense_iter("cpu")
    X1, y1 = run_dense_iter("cuda")
    assert_allclose(X0, X1, rtol=5e-6)
    assert_allclose(y0, y1, rtol=5e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cv(device: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path0 = os.path.join(tmpdir, "data0")
        path1 = os.path.join(tmpdir, "data1")

        # Within batch read
        n_features = 2
        n_batches = 4
        nspb = 8

        datagen(nspb, n_features, n_batches, False, 0.0, device, [path0, path1])
        n_train, n_valid = get_valid_sizes(n_samples=nspb * n_batches)
        assert n_valid == 6

        files = get_file_paths([path0, path1])
        impl = LoadIterImpl(list(zip(files[0], files[1])), True, False, device)
        assert len(impl.X_shards) == n_batches

        X, y = load_all([path0, path1], device)

        prev = 0
        for i in range(impl.n_batches):
            X_i, y_i = impl.get(i)
            if isinstance(X_i, cp.ndarray):
                assert_allclose = cp.testing.assert_allclose
            else:
                assert_allclose = np.testing.assert_allclose
            assert_allclose(X[prev + 1 : prev + nspb], X_i)
            assert_allclose(y[prev + 1 : prev + nspb], y_i)
            prev += nspb
