# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import gc
import os
from abc import abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias

import numpy as np
import xgboost as xgb
from scipy import sparse
from typing_extensions import override
from xgboost.compat import concat

from .datasets.generated import make_dense_regression, make_sparse_regression
from .utils import Timer, fprint

if TYPE_CHECKING:
    from cupy import ndarray as cpnd
else:
    cpnd = Any


class IterImpl:
    @abstractmethod
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractproperty
    def n_batches(self) -> int: ...


def load_Xy(
    Xp: str, yp: str, use_mmap: bool
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    if Xp.endswith("npz"):
        X = sparse.load_npz(Xp)
        y = np.load(yp)
    else:
        X = np.load(file=Xp, mmap_mode="r" if use_mmap else None)
        y = np.load(file=yp, mmap_mode="r" if use_mmap else None)

    return X, y


XyPair: TypeAlias = tuple[np.ndarray | sparse.csr_matrix, np.ndarray]


def get_file_paths(loadfrom: str) -> tuple[list[str], list[str]]:
    X_files: list[str] = []
    y_files: list[str] = []
    for root, subdirs, files in os.walk(loadfrom):
        for f in files:
            path = os.path.join(root, f)
            if f.startswith("X-"):
                X_files.append(path)
            else:
                y_files.append(path)

    def key(name: str) -> int:
        i = name.split("-")[1].split(".")[0]
        return int(i)

    X_files = sorted(X_files, key=key)
    y_files = sorted(y_files, key=key)
    return X_files, y_files


def load_batches(
    loadfrom: str, device: str
) -> tuple[list[np.ndarray] | list[sparse.csr_matrix], list[np.ndarray]]:
    """Load all batches."""
    X_files, y_files = get_file_paths(loadfrom)
    with Timer("load-batches", "load"):
        paths = list(zip(X_files, y_files))
        assert paths

        Xs, ys = [], []
        for i in range(len(X_files)):
            # Don't use mmap since we need all the data loaded anyway.
            X, y = load_Xy(X_files[i], y_files[i], False)
            if device == "cuda":
                import cupy as cp

                X = cp.array(X)
                y = cp.array(y)
            Xs.append(X)
            ys.append(y)
    assert len(Xs) == len(ys)
    fprint(f"Total {len(Xs)} batches.")
    return Xs, ys


def load_all(loadfrom: str, device: str) -> XyPair:
    """Load all batches and concatenate them into a single blob.."""
    Xs, ys = load_batches(loadfrom, device)
    with Timer("load-all", "concat"):
        X = concat(Xs)
        y = concat(ys)
    return X, y


class LoadIterImpl(IterImpl):
    """Iterator for loading files."""

    def __init__(self, files: list[tuple[str, str]]) -> None:
        assert files
        self.files = files

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self.files[i]
        # Use mmap since we might only need a portion of data thanks to validation.
        X, y = load_Xy(X_path, y_path, True)
        assert X.shape[0] == y.shape[0]
        return X, y

    @property
    def n_batches(self) -> int:
        return len(self.files)


class SynIterImpl(IterImpl):
    """An iterator for synthesizing dataset on-demand."""

    def __init__(
        self,
        n_samples_per_batch: int,
        n_features: int,
        n_batches: int,
        sparsity: float,
        assparse: bool,
        device: str,
    ) -> None:
        self.n_samples_per_batch = n_samples_per_batch
        self.n_features = n_features
        self._n_batches = n_batches
        self.sparsity = sparsity
        self.assparse = assparse
        self.device = device

        self.sizes: list[int] = []

    @property
    def n_batches(self) -> int:
        return self._n_batches

    def _seed(self, i: int) -> int:
        return sum(self.sizes[:i])

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.assparse:
            assert i == 0, "not implemented"
            X, y = make_sparse_regression(
                n_samples=self.n_samples_per_batch,
                n_features=self.n_features,
                sparsity=self.sparsity,
                random_state=self._seed(i),
            )
        else:
            X, y = make_dense_regression(
                device=self.device,
                n_samples=self.n_samples_per_batch,
                n_features=self.n_features,
                sparsity=self.sparsity,
                random_state=self._seed(i),
            )
        if len(self.sizes) != self._n_batches:
            self.sizes.append(X.size)
        return X, y


TEST_SIZE = 0.2


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float, random_state: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Only used for profiling, not suitable for real world validation.
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    X_train = X[:n_train, ...]
    X_test = X[n_train:, ...]

    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


def get_train(
    X: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    # Only used for profiling, not suitable for real world validation.
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    X_train = X[:n_train, ...]

    y_train = y[:n_train]
    return X_train, y_train


# With the help of mmap, we can read only a portion of data
def get_valid(
    X: np.ndarray, y: np.ndarray, test_size: float
) -> tuple[np.ndarray, np.ndarray]:
    # Only used for profiling, not suitable for real world validation.
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    X_test = X[n_train:, ...]
    y_test = y[n_train:]
    return X_test, y_test


def to_cupy(array: cpnd | np.ndarray) -> cpnd:
    import cupy as cp

    if not isinstance(array, cp.ndarray):
        return cp.array(array)

    return array


class BenchIter(xgb.DataIter):
    """A custom iterator for profiling."""

    def __init__(
        self, it: IterImpl, split: bool, is_ext: bool, is_eval: bool, device: str
    ) -> None:
        self._it = 0
        self._impl = it
        self._split = split
        self._is_eval = is_eval
        self._device = device

        if is_ext:
            super().__init__(cache_prefix="cache", on_host=True)
        else:
            super().__init__()

    def next(self, input_data: Callable) -> bool:
        if self._it == self._impl.n_batches:
            return False

        print("Next:", self._it, flush=True)
        gc.collect()
        X, y = self._impl.get(self._it)

        if self._split:
            if self._is_eval:
                with Timer("BenchIter", "GetValid"):
                    X_valid, y_valid = get_valid(X, y, test_size=TEST_SIZE)
                    if self._device == "cuda":
                        X_valid, y_valid = to_cupy(X_valid), to_cupy(y_valid)
                input_data(data=X_valid, label=y_valid)
            else:
                with Timer("BenchIter", "GetTrain"):
                    X_train, y_train = get_train(X, y, test_size=TEST_SIZE)
                    if self._device == "cuda":
                        X_train, y_train = to_cupy(X_train), to_cupy(y_train)
                input_data(data=X_train, label=y_train)
        else:
            if self._device == "cuda":
                X, y = to_cupy(X), to_cupy(y)
            input_data(data=X, label=y)

        self._it += 1
        return True

    def reset(self) -> None:
        print("Reset:", flush=True)
        self._it = 0
        gc.collect()
