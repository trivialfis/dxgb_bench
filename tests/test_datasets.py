import numpy as np
import pytest
from scipy import sparse

from dxgb_bench.datasets.generated import make_dense_regression, make_sparse_regression
from dxgb_bench.dxgb_bench import datagen
from dxgb_bench.dataiter import load_all


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


def test_dense_batches() -> None:
    device = "cpu"
    datagen(4, 1, 2, False, 0.0, device, "./data")
    X0, y0 = load_all("./data", "cpu")
    print(X0)
    import shutil
    shutil.rmtree("./data")

    datagen(8, 1, 1, False, 0.0, device, "./data")
    X1, y1 = load_all("./data", "cpu")
    print(X1)
