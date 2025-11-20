# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import gc
import os
from abc import abstractmethod, abstractproperty
from typing import TYPE_CHECKING, Any, Callable, SupportsFloat, TypeAlias

import numpy as np
import xgboost as xgb
from scipy import sparse
from typing_extensions import override
from xgboost.collective import get_rank
from xgboost.compat import concat

from .datasets.generated import (
    make_dense_binary_classification,
    make_dense_regression,
    make_sparse_regression,
)
from .strip import make_strips
from .utils import TEST_SIZE, Timer, fprint

if TYPE_CHECKING:
    from cupy import ndarray as cpnd
else:
    cpnd = Any


class IterImpl:
    @abstractmethod
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]: ...

    @abstractproperty
    def n_batches(self) -> int: ...


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


def load_batches(
    loadfrom: list[str], device: str
) -> tuple[list[np.ndarray] | list[sparse.csr_matrix], list[np.ndarray]]:
    """Load all batches."""
    X_fd, y_fd = make_strips(["X", "y"], dirs=loadfrom, fmt=None, device=device)

    with Timer("load-batches", "load"):
        Xs, ys = [], []
        for i in range(X_fd.n_batches):
            X = X_fd.read(i, None, None)
            y = y_fd.read(i, None, None)
            assert X.shape[0] == y.shape[0]
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


def get_valid_sizes(
    n_samples: int, test_size: SupportsFloat = TEST_SIZE
) -> tuple[int, int]:
    n_valid = int(n_samples * float(test_size))
    n_train = n_samples - n_valid
    return n_train, n_valid


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


class LoadIterStrip(IterImpl):
    """An iterator for loading dataset from disks.

    Parameters
    ----------
    loadfrom:
        A list of directories containing the data.
    is_valid:
        Whether this is a validation dataset. Ignored when test_size is None.
    test_size:
        Use validation split if it's not None. Specifies the ratio of the test dataset.
    device:
        cpu or cuda.

    """

    def __init__(
        self, loadfrom: list[str], is_valid: bool, test_size: float | None, device: str
    ) -> None:
        self.loadfrom = loadfrom
        self._is_valid = is_valid
        self._test_ratio = test_size

        self.X, self.y = make_strips(["X", "y"], dirs=loadfrom, fmt=None, device=device)

        self.pinfo = self.X.list_file_info()

        if is_valid:
            if test_size is None:
                raise ValueError("Must have a size for train test split.")

    @property
    def n_batches(self) -> int:
        return self.X.n_batches

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        Xinfo = self.X.batch_key[i]

        n_samples = Xinfo.n_samples

        if self._test_ratio is not None:
            n_train, _ = get_valid_sizes(n_samples, self._test_ratio)
            if self._is_valid:
                # This is the validation dataset.
                # similar to X_i[n_train:]
                X = self.X.read(i, n_train, n_samples)
                y = self.y.read(i, n_train, n_samples)
            else:
                # This is the training dataset.
                # similar to X_i[:n_train]
                X = self.X.read(i, 0, n_train)
                y = self.y.read(i, 0, n_train)
        else:
            # Read the full batch
            X = self.X.read(i, None, None)
            y = self.y.read(i, None, None)
        return X, y


class SynIterImpl(IterImpl):
    """An iterator for synthesizing dataset on-demand."""

    def __init__(
        self,
        n_samples_per_batch: int,
        n_features: int,
        n_targets: int,
        n_batches: int,
        sparsity: float,
        assparse: bool,
        target_type: str,
        device: str,
        rs: int = 0,
    ) -> None:
        self.n_samples_per_batch = n_samples_per_batch
        self.n_features = n_features
        self.n_targets = n_targets
        self._n_batches = n_batches
        self.sparsity = sparsity
        self.assparse = assparse
        self.target_type = target_type
        self.device = device

        self.sizes: list[int] = []
        self.rs = rs

        for i in range(self._n_batches):
            size = self.n_samples_per_batch * self.n_features
            self.sizes.append(size)

    @property
    def n_batches(self) -> int:
        assert len(self.sizes) == self._n_batches
        return self._n_batches

    def _seed(self, i: int) -> int:
        return sum(self.sizes[:i]) + self.rs

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        if self.target_type == "bin" and self.assparse:
            raise NotImplementedError(
                "assparse is not supported for binary classification yet."
            )
        if self.target_type == "bin":
            X, y = make_dense_binary_classification(
                device=self.device,
                n_samples=self.n_samples_per_batch,
                n_features=self.n_features,
                n_targets=self.n_targets,
                random_state=self._seed(i),
            )
            fprint("seed:", self._seed(i))
            assert self.sizes[i] == X.size
            return X, y
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
                n_targets=self.n_targets,
                sparsity=self.sparsity,
                random_state=self._seed(i),
            )
        assert self.sizes[i] == X.size, (self.sizes[i], X.size)
        return X, y


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


class DxgbIter(xgb.DataIter):
    @abstractproperty
    def n_batches(self) -> int: ...


def _silent(msg: str) -> None:
    pass


class BenchIter(DxgbIter):
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

    @override
    @property
    def n_batches(self) -> int:
        return self._impl.n_batches

    @override
    def next(self, input_data: Callable) -> bool:
        if self._it == self._impl.n_batches:
            return False

        fprint(f"Next: {self._it}")
        gc.collect()
        with Timer(
            "BenchIter", "GetValid" if self._is_eval else "GetTrain", logger=_silent
        ):
            X, y = self._impl.get(self._it)
        input_data(data=X, label=y)

        self._it += 1
        return True

    @override
    def reset(self) -> None:
        fprint("Reset")
        self._it = 0
        gc.collect()


class StridedIter(DxgbIter):
    """An iterator for loading data with strided iteration."""

    def __init__(
        self,
        it: IterImpl,
        *,
        start: int,
        stride: int,
        is_ext: bool,
        is_valid: bool,
        device: str,
    ) -> None:
        self._start = start
        self._stride = stride
        self._impl = it
        self._is_eval = is_valid

        self._it = start

        if is_ext:
            super().__init__(cache_prefix="cache", on_host=True)
        else:
            super().__init__()

    @override
    @property
    def n_batches(self) -> int:
        n_total_batches = self._impl.n_batches
        return n_total_batches // self._stride

    @override
    def next(self, input_data: Callable) -> bool:
        if self._it >= self._impl.n_batches:
            return False

        if get_rank() == 0:
            fprint(f"Next: {self._it}")

        gc.collect()
        with Timer(
            "BenchIter", "GetValid" if self._is_eval else "GetTrain", logger=_silent
        ):
            X, y = self._impl.get(self._it)
        input_data(data=X, label=y)

        self._it += self._stride
        return True

    @override
    def reset(self) -> None:
        if get_rank() == 0:
            fprint("Reset")
        self._it = self._start
        gc.collect()
