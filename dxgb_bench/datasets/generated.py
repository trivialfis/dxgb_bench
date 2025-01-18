# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import ctypes
import functools
import os
import pickle
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Tuple

import numpy as np
from scipy import sparse

from ..utils import DType, div_roundup, fprint
from .dataset import DataSet


@functools.cache
def _load_lib() -> ctypes.CDLL:
    path = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(__file__))),
        os.pardir,
        "libdxgbbench.so",
    )
    lib = ctypes.cdll.LoadLibrary(path)
    return lib


def make_reg_c(
    is_cuda: bool, n_samples_per_batch: int, n_features: int, seed: int, sparsity: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Use the C++/CUDA implementation of dense data gen."""
    _lib = _load_lib()

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

    status = _lib.MakeDenseRegression(
        ctypes.c_bool(is_cuda),
        ctypes.c_int64(n_samples_per_batch),
        ctypes.c_int64(n_features),
        ctypes.c_double(sparsity),
        ctypes.c_int64(seed),
        X_ptr,
        y_ptr,
    )
    if status != 0:
        raise ValueError("Native implementation failed.")
    return X, y


def make_sparse_regression(
    n_samples: int, n_features: int, *, sparsity: float, random_state: int
) -> Tuple[sparse.csr_matrix, np.ndarray]:
    """Make sparse synthetic data for regression. Result is stored in CSR even if the
    data is dense.

    """

    n_threads_maybe_none = os.cpu_count()
    assert n_threads_maybe_none is not None
    n_threads = min(n_threads_maybe_none, n_samples)
    n_samples_per_batch = div_roundup(n_samples, n_threads)

    def random_csr(t_id: int, seed: int) -> sparse.csr_matrix:
        rng = np.random.default_rng(seed)
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
            seed = random_state + n_samples_per_batch * (i + 1)
            futures.append(executor.submit(random_csr, i, seed))

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Make dense synthetic data for regression. Result is stored in arrays even if the
    data is sparse.

    """

    X, y = make_reg_c(
        is_cuda=device == "cuda",
        n_samples_per_batch=n_samples,
        n_features=n_features,
        seed=random_state,
        sparsity=sparsity,
    )
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
            args.n_samples, args.n_features, sparsity=args.sparsity, random_state=0
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
