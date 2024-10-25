# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import gc
import os
import re
from abc import abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, Literal, TypeAlias

import kvikio
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


def get_pinfo(Xp: str) -> tuple[str, int, int, int]:
    name = os.path.basename(Xp)
    mat = re.search(fname_pattern, name)
    assert mat is not None
    x, rows_str, cols_str, batch_str = mat.groups()
    n_samples, n_features, batch_idx = int(rows_str), int(cols_str), int(batch_str)
    return x, n_samples, n_features, batch_idx


def load_Xy(
    Xp: str, yp: str, device: str
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    if Xp.endswith("npz"):
        X = sparse.load_npz(Xp)
        y = np.load(yp)
    else:
        _, n_samples, n_features, batch_idx = get_pinfo(Xp)
        X, y = _alloc(n_samples, n_features, device)
        with kvikio.CuFile(Xp, "r") as fd:
            fd.read(X)
        with kvikio.CuFile(yp, "r") as f:
            f.read(y)

    return X, y


XyPair: TypeAlias = tuple[np.ndarray | sparse.csr_matrix, np.ndarray]


def get_file_paths_local(dirname: str) -> tuple[list[str], list[str]]:
    X_files: list[str] = []
    y_files: list[str] = []
    assert os.path.exists(dirname), dirname
    for root, subdirs, files in os.walk(dirname):
        for f in files:
            path = os.path.join(root, f)
            if f.startswith("X"):
                X_files.append(path)
            else:
                y_files.append(path)

    assert len(X_files) == len(y_files)
    return X_files, y_files


fname_pattern = "(\w)_(\d+)_(\d+)-(\d+).npa"


def get_file_paths(loadfrom: list[str]) -> tuple[list[str], list[str]]:
    X_files: list[str] = []
    y_files: list[str] = []
    for d in loadfrom:
        X_fd, y_fd = get_file_paths_local(d)
        X_files.extend(X_fd)
        y_files.extend(y_fd)

    def key(name: str) -> int:
        name = os.path.basename(name)
        mat = re.search(fname_pattern, name)
        assert mat is not None, name
        i = mat.group(4)
        return int(i)

    X_files = sorted(X_files, key=key)
    y_files = sorted(y_files, key=key)

    return X_files, y_files


def load_batches(
    loadfrom: list[str], device: str
) -> tuple[list[np.ndarray] | list[sparse.csr_matrix], list[np.ndarray]]:
    """Load all batches."""
    X_files, y_files = get_file_paths(loadfrom)

    with Timer("load-batches", "load"):
        paths = list(zip(X_files, y_files))
        assert paths

        Xs, ys = [], []
        for i in range(len(X_files)):
            # Don't use mmap since we need all the data loaded anyway.
            X, y = load_Xy(X_files[i], y_files[i], device)
            if device == "cuda":
                import cupy as cp

                X = cp.array(X)
                y = cp.array(y)
            Xs.append(X)
            ys.append(y)
    assert len(Xs) == len(ys)
    fprint(f"Total {len(Xs)} batches.")
    return Xs, ys


def load_all(loadfrom: list[str], device: str) -> XyPair:
    """Load all batches and concatenate them into a single blob.."""
    Xs, ys = load_batches(loadfrom, device)
    with Timer("load-all", "concat"):
        if len(Xs) == 1:
            return Xs[0], ys[0]

        X = concat(Xs)
        y = concat(ys)
    return X, y


def _alloc(
    n_samples: int, n_features: int, device: str
) -> tuple[np.ndarray, np.ndarray]:
    if device == "cpu":
        X = np.empty(shape=(n_samples, n_features), dtype=np.float32)
        y = np.empty(shape=n_samples, dtype=np.float32)
    else:
        import cupy as cp

        X = cp.empty(shape=(n_samples, n_features), dtype=np.float32)
        y = cp.empty(shape=n_samples, dtype=np.float32)
    return X, y


class LoadIterImpl(IterImpl):
    """Iterator for loading files."""

    def __init__(
        self, files: list[tuple[str, str]], split: bool, is_valid: bool, device: str
    ) -> None:
        assert files
        self.files = files
        self.split = split
        self.is_valid = is_valid
        self.device = device

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self.files[i]
        name = os.path.basename(X_path)
        mat = re.search(fname_pattern, name)
        assert mat is not None
        _, rows_str, cols_str, batch_str = mat.groups()
        n_samples, n_features, batch_idx = int(rows_str), int(cols_str), int(batch_str)
        assert batch_idx == i

        if self.split:
            n_valid = int(n_samples * TEST_SIZE)
            n_train = n_samples - n_valid
            if self.is_valid:
                X, y = _alloc(n_valid, n_features, self.device)

                offset_bytes = n_train * n_features * X.itemsize
                with kvikio.CuFile(X_path, "r") as f:
                    f.read(X, file_offset=offset_bytes)

                offset_bytes = n_train * y.itemsize
                with kvikio.CuFile(y_path, "r") as f:
                    f.read(y, file_offset=offset_bytes)
            else:
                X, y = _alloc(n_train, n_features, self.device)
                with kvikio.CuFile(X_path, "r") as f:
                    f.read(X)
                with kvikio.CuFile(y_path, "r") as f:
                    f.read(y)
        else:
            X, y = load_Xy(X_path, y_path, self.device)

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
    with Timer("train_test_split", "train_test_split"):
        n_samples = X.shape[0]
        n_test = int(n_samples * test_size)
        n_train = n_samples - n_test

        X_train = X[:n_train, ...]
        X_test = X[n_train:, ...]

        y_train = y[:n_train]
        y_test = y[n_train:]

        return X_train, X_test, y_train, y_test


class BenchIter(xgb.DataIter):
    """A custom iterator for profiling."""

    def __init__(self, it: IterImpl, is_ext: bool, is_valid: bool, device: str) -> None:
        self._it = 0
        self._impl = it
        self._is_eval = is_valid
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
        with Timer("BenchIter", "GetValid" if self._is_eval else "GetTrain"):
            X, y = self._impl.get(self._it)
        input_data(data=X, label=y)

        self._it += 1
        return True

    def reset(self) -> None:
        print("Reset:", flush=True)
        self._it = 0
        gc.collect()
