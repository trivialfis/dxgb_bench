from __future__ import annotations

import os
import tempfile

import cupy as cp
import numpy as np
import pytest
from scipy import sparse
from xgboost.compat import concat

from dxgb_bench.dataiter import BenchIter, SynIterImpl, load_all
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
    assert 13000 < X.nnz < 13100


@pytest.mark.parametrize("force_py", [True, False])
def test_dense_regression(force_py: bool) -> None:
    X, y = make_dense_regression(
        n_samples=3,
        n_features=2,
        sparsity=0.6,
        device="cpu",
        random_state=1,
        _force_py=force_py,
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
        _force_py=force_py,
    )
    assert X.shape[0] == 2047 and X.shape[1] == 16
    assert y.shape[0] == X.shape[0]
    nnz = np.count_nonzero(~np.isnan(X))
    assert 13000 < nnz < 13230
    nnz = np.count_nonzero(~np.isnan(y))
    assert nnz == 2047


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dense_batches(device: str) -> None:
    n_features = 3
    n_batches = 7
    nspb = 64

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(nspb, n_features, n_batches, False, 0.0, device, path)
        X0, y0 = load_all(path, "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(nspb * n_batches, n_features, 1, False, 0.0, device, path)
        X1, y1 = load_all(path, "cpu")

    np.testing.assert_allclose(X0, X1)
    np.testing.assert_allclose(y0, y1)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dense_iter(device: str) -> None:
    n_features = 4
    n_batches = 2
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
        datagen(nspb, n_features, n_batches, False, 0.0, device, path)
        X3, y3 = load_all(path, "cpu")

    def assert_allclose(a: np.ndarray | cp.ndarray, b: np.ndarray | cp.ndarray) -> None:
        if hasattr(a, "get"):
            a = a.get()
        if hasattr(b, "get"):
            b = b.get()
        np.testing.assert_allclose(a, b)

    assert_allclose(X0, X1)
    assert_allclose(X0, X2)
    assert_allclose(X0, X3)

    assert_allclose(y0, y1)
    assert_allclose(y0, y2)
    assert_allclose(y0, y3)
