# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import gc
import os
from abc import abstractmethod, abstractproperty
from typing import Callable, TypeAlias

import numpy as np
import xgboost as xgb
from xgboost.compat import concat
from scipy import sparse
from typing_extensions import override

from .datasets.generated import make_dense_regression, make_sparse_regression
from .utils import Timer


class IterImpl:
    @abstractmethod
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractproperty
    def n_batches(self) -> int: ...


def load_Xy(
    Xp: str, yp: str, device: str
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    if Xp.endswith("npz"):
        assert device == "cpu", "not implemented"
        X = sparse.load_npz(Xp)
        y = np.load(yp)
    else:
        X = np.lib.format.open_memmap(filename=Xp, mode="r")
        y = np.lib.format.open_memmap(filename=yp, mode="r")
        if device == "cuda":
            import cupy as cp

            X = cp.asarray(X)
            y = cp.asanyarray(y)

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
            X, y = load_Xy(X_files[i], y_files[i], device)
            Xs.append(X)
            ys.append(y)
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

    def __init__(self, files: list[tuple[str, str]], device: str) -> None:
        assert files
        self.files = files
        self.device = device

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self.files[i]
        X, y = load_Xy(X_path, y_path, self.device)
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

    @property
    def n_batches(self) -> int:
        return self._n_batches

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.assparse:
            assert i == 0, "not implemented"
            X, y = make_sparse_regression(
                n_samples=self.n_samples_per_batch,
                n_features=self.n_features,
                sparsity=self.sparsity,
            )
        else:
            X, y = make_dense_regression(
                device=self.device,
                n_samples=self.n_samples_per_batch,
                n_features=self.n_features,
                sparsity=self.sparsity,
                random_state=i,
            )
        return X, y


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


TEST_SIZE = 0.2


class BenchIter(xgb.DataIter):
    """A custom iterator for profiling."""

    def __init__(self, it: IterImpl, split: bool, is_ext: bool, is_eval: bool) -> None:
        self._it = 0
        self._impl = it
        self._split = split
        self._is_eval = is_eval

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
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42
            )
            if self._is_eval:
                input_data(data=X_valid, label=y_valid)
            else:
                input_data(data=X_train, label=y_train)
        else:
            input_data(data=X, label=y)

        self._it += 1
        return True

    def reset(self) -> None:
        print("Reset:", flush=True)
        self._it = 0
        gc.collect()
