# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import os
from abc import abstractmethod, abstractproperty

import numpy as np
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
        # fixme(jiamingy): Consider using mmap.
        if device == "cpu":
            X = np.load(Xp)
            y = np.load(yp)
        else:
            import cupy as cp

            X = cp.load(Xp)
            y = cp.load(yp)

    return X, y


def load_batches(
    loadfrom: str, device: str
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    """Load all batches and concatenate them into a single pair."""
    X_files: list[str] = []
    y_files: list[str] = []
    with Timer("load-batches", "load"):
        for root, subdirs, files in os.walk(loadfrom):
            for f in files:
                path = os.path.join(root, f)
                if f.startswith("X-"):
                    X_files.append(path)
                else:
                    y_files.append(path)
        paths = list(zip(X_files, y_files))
        assert paths

        Xs, ys = [], []
        for i in range(len(X_files)):
            X, y = load_Xy(X_files[i], y_files[i], device)
            Xs.append(X)
            ys.append(y)

    with Timer("load-batches", "concat"):
        if isinstance(Xs[0], sparse.csr_matrix):
            X = sparse.vstack(Xs)
        else:
            X = np.vstack(Xs)
        y = np.vstack(ys)
    return X, y


class LoadIter(IterImpl):
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


class SynIter(IterImpl):
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
