from __future__ import annotations

import argparse
import ctypes
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy import sparse
from xgboost.compat import concat

from ..utils import DType, div_roundup, fprint
from .dataset import DataSet


def make_reg_c(
    is_cuda: bool, n_samples_per_batch: int, n_features: int, seed: int, sparsity: float
) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(__file__))),
        os.pardir,
        "libdxgbbench.so",
    )
    _lib = ctypes.cdll.LoadLibrary(path)
    if is_cuda:
        import cupy as cp

        X = cp.empty(shape=(n_samples_per_batch, n_features), dtype=np.float32)
        y = cp.empty(shape=(n_samples_per_batch,), dtype=np.float32)
        X_ptr = ctypes.cast(
            X.__cuda_array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
        y_ptr = ctypes.cast(
            y.__cuda_array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
    else:
        X = np.empty(shape=(n_samples_per_batch, n_features), dtype=np.float32)
        y = np.empty(shape=(n_samples_per_batch,), dtype=np.float32)
        X_ptr = ctypes.cast(
            X.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
        y_ptr = ctypes.cast(
            y.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )

    _lib.MakeDenseRegression(
        ctypes.c_bool(is_cuda),
        ctypes.c_int64(n_samples_per_batch),
        ctypes.c_int64(n_features),
        ctypes.c_double(sparsity),
        ctypes.c_int64(seed),
        X_ptr,
        y_ptr,
    )
    return X, y


def make_sparse_regression(
    n_samples: int, n_features: int, *, sparsity: float
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Make dense synthetic data for regression. Result is stored in CSR even if the
    data is dense.

    """

    n_threads_maybe_none = os.cpu_count()
    assert n_threads_maybe_none is not None
    n_threads = min(n_threads_maybe_none, n_samples)
    n_samples_per_batch = div_roundup(n_samples, n_threads)

    def random_csr(t_id: int) -> sparse.csr_matrix:
        rng = np.random.default_rng(1994 * t_id)
        if t_id == n_threads - 1:
            nspb = n_samples - (n_samples_per_batch * (n_threads - 1))
        else:
            nspb = n_samples_per_batch

        csr = sparse.random(
            m=nspb,
            n=n_features,
            density=1.0 - sparsity,
            random_state=rng,
        )
        y = csr.sum(axis=1)
        y += rng.normal(loc=0, scale=0.5, size=y.shape)
        return csr, y

    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            futures.append(executor.submit(random_csr, i))

        X_results = []
        y_results = []
        for f in futures:
            X, y = f.result()
            X_results.append(X)
            y_results.append(y)

        assert len(y_results) == n_threads

        X = sparse.vstack(X_results, format="csr")
        y = np.vstack(y_results)

    assert X.shape[0] == n_samples, X.shape
    assert y.shape[0] == n_samples, y.shape
    return X, y


def psize(X: np.ndarray) -> None:
    n_bytes = X.itemsize * X.size
    if n_bytes < 1024:
        size = f"{n_bytes} B"
    elif n_bytes < 1024**2:
        size = f"{n_bytes / 1024} KB"
    elif n_bytes < 1024**3:
        size = f"{n_bytes / 1024 ** 2} MB"
    else:
        size = f"{n_bytes / 1024 ** 3} GB"
    fprint(f"Estimated Size: {size}")


def make_dense_regression(
    device: str,
    n_samples: int,
    n_features: int,
    *,
    sparsity: float,
    random_state: int,
    _force_py: int = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Make dense synthetic data for regression. Result is stored in arrays even if the
    data is sparse.

    """

    try:
        if _force_py:
            raise ValueError("Force using the Python version for testing.")

        X, y = make_reg_c(
            is_cuda=device == "cuda",
            n_samples_per_batch=n_samples,
            n_features=n_features,
            seed=random_state * n_samples,
            sparsity=sparsity,
        )
        psize(X)
        return X, y
    except Exception as e:
        print(f"Failed to generate using the C extension. {e}")

    n_cpus = os.cpu_count()
    assert n_cpus is not None
    n_threads = min(n_cpus, n_samples)
    start = 0

    def make_regression(
        n_samples_per_batch: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # A custom version of make_regression since sklearn doesn't support np
        # generator.
        rng = np.random.default_rng(seed)
        X = rng.normal(
            loc=0.0, scale=1.5, size=(n_samples_per_batch, n_features)
        ).astype(np.float32)
        err = rng.normal(0.0, scale=0.2, size=(n_samples_per_batch,)).astype(np.float32)
        y = X.sum(axis=1) + err
        mask = rng.binomial(1, sparsity, X.shape).astype(np.bool_)
        X[mask] = np.nan
        return X, y

    futures = []
    n_samples_per_batch = n_samples // n_threads
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            n_samples_cur = n_samples_per_batch
            if i == n_threads - 1:
                n_samples_cur = n_samples - start
            fut = executor.submit(
                make_regression, n_samples_cur, start + random_state * n_samples
            )
            start += n_samples_cur
            futures.append(fut)

    X_arr, y_arr = [], []
    for fut in futures:
        X, y = fut.result()
        X_arr.append(X)
        y_arr.append(y)

    def parallel_concat(
        X_arr: list[np.ndarray], y_arr: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        with ThreadPoolExecutor(max_workers=n_threads) as executor_1:
            while True:
                X_arr_1 = []
                y_arr_1 = []
                X_futures = []

                for i in range(0, len(X_arr), 2):
                    if i + 1 < len(X_arr):
                        X_fut = executor_1.submit(concat, X_arr[i : i + 2])
                        y_fut = executor_1.submit(concat, y_arr[i : i + 2])
                    else:
                        X_fut = executor_1.submit(lambda x: x, X_arr[i])
                        y_fut = executor_1.submit(lambda x: x, y_arr[i])
                    X_futures.append((X_fut, y_fut))

                for X_fut, y_fut in X_futures:
                    X_arr_1.append(X_fut.result())
                    y_arr_1.append(y_fut.result())

                X_arr = X_arr_1
                y_arr = y_arr_1

                if len(X_arr) == 1 and len(y_arr) == 1:
                    break
        assert len(X_arr) == 1 and len(y_arr) == 1
        return X_arr[0], y_arr[0]

    X, y = parallel_concat(X_arr, y_arr)
    if device == "cuda":
        import cupy as cp

        return cp.array(X), cp.array(y)
    psize(X)
    return X, y


class Generated(DataSet):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.backend.find("dask") != -1:
            raise NotImplementedError()
        if args.task is None:
            raise ValueError("`task` is required to generate dataset.")
        if args.n_samples is None:
            raise ValueError("`n_samples` is required to generate dataset.")
        if args.n_features is None:
            raise ValueError("`n_features` is required to generate dataset.")
        if args.sparsity is None:
            raise ValueError("`sparsity` is required to generate dataset.")

        if args.task != "reg":
            raise NotImplementedError()
        self.dirpath = os.path.join(
            args.local_directory, f"{args.n_samples}-{args.n_features}-{args.sparsity}"
        )
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)

        self.X_path = os.path.join(self.dirpath, "X.pkl")
        self.y_path = os.path.join(self.dirpath, "y.pkl")

        self.task: str = "reg:squarederror"

        if os.path.exists(self.X_path) and os.path.exists(self.y_path):
            return

        X, y = make_sparse_regression(
            args.n_samples, args.n_features, sparsity=args.sparsity
        )

        with open(self.X_path, "wb") as fd:
            pickle.dump(X, fd)
        with open(self.y_path, "wb") as fd:
            pickle.dump(y, fd)

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        with open(self.X_path, "rb") as fd:
            X = pickle.load(fd)
        with open(self.y_path, "rb") as fd:
            y = pickle.load(fd)

        return X, y, None
