# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import gc
import os
import re
from abc import abstractmethod, abstractproperty
from bisect import bisect_right
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Callable, TypeAlias

import kvikio
import numpy as np
import xgboost as xgb
from numpy import typing as npt
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


def get_pinfo(Xp: str) -> tuple[str, int, int, int, int]:
    name = os.path.basename(Xp)
    mat = re.search(fname_pattern, name)
    assert mat is not None
    x, rows_str, cols_str, batch_str, shard_str = mat.groups()
    n_samples, n_features, batch_idx, shard_idx = (
        int(rows_str),
        int(cols_str),
        int(batch_str),
        int(shard_str),
    )
    return x, n_samples, n_features, batch_idx, shard_idx


def load_Xy(
    Xp: str, yp: str, device: str
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    if Xp.endswith("npz"):
        X = sparse.load_npz(Xp)
        y = np.load(yp)
    else:
        _, n_samples, n_features, batch_idx, shard = get_pinfo(Xp)
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


# X|y-rows-columns-batch-shard.npa
fname_pattern = r"[X|y]_(\d+)_(\d+)-(\d+)-(\d+).npa"


def get_file_paths(loadfrom: list[str]) -> tuple[list[str], list[str]]:
    X_files: list[str] = []
    y_files: list[str] = []
    for d in loadfrom:
        X_fd, y_fd = get_file_paths_local(d)
        X_files.extend(X_fd)
        y_files.extend(y_fd)

    def key(name: str) -> tuple[int, int]:
        name = os.path.basename(name)
        mat = re.search(fname_pattern, name)
        assert mat is not None, name
        batch = mat.group(4)
        shard = mat.group(5)
        return int(batch), int(shard)

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


def get_valid_sizes(n_samples: int) -> tuple[int, int]:
    n_valid = int(n_samples * TEST_SIZE)
    n_train = n_samples - n_valid
    return n_train, n_valid


def find_shard_ids(
    indptr: npt.NDArray[np.int64], fold: int
) -> tuple[int, int, int, int]:
    n_samples = indptr[-1]
    n_train, n_valid = get_valid_sizes(n_samples)

    begin = n_valid * fold
    end = begin + n_valid

    assert end < n_samples

    beg_idx: int = bisect_right(indptr, begin) - 1
    end_idx: int = bisect_right(indptr, end) - 1

    print(beg_idx, end_idx, indptr, begin)

    beg_in_shard = begin - indptr[beg_idx]
    end_in_shard = end - indptr[end_idx]

    return beg_idx, beg_in_shard, end_idx, end_in_shard


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

        X_shards = defaultdict(list)
        y_shards = defaultdict(list)
        for fname in files:
            name, n_samples, n_features, batch_idx, shard_idx = get_pinfo(fname[0])
            X_shards[batch_idx].append(fname[0])
            y_shards[batch_idx].append(fname[1])
            assert fname[1].startswith("y")

        def key(name: str) -> int:
            name = os.path.basename(name)
            mat = re.search(fname_pattern, name)
            assert mat is not None, name
            shard = mat.group(5)
            return int(shard)

        X_shards_list = []
        for batch_idx, shards in X_shards.items():
            shards = sorted(shards, key=key)
            X_shards_list.append(shards)

        y_shards_list = []
        for batch_idx, shards in y_shards.items():
            shards = sorted(shards, key=key)
            y_shards_list.append(shards)

        self.X_shards = X_shards_list
        self.y_shards = y_shards_list

        assert len(self.X_shards) == len(self.y_shards)

        self.split = split
        self.is_valid = is_valid
        self.device = device

    def get_with_valid(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_shards_i = self.X_shards[i]
        y_shards_i = self.y_shards[i]

        shard_sizes = [0]
        for Xs, ys in zip(X_shards_i, y_shards_i):
            _, n_samples_i, n_features, bidx, sidx = get_pinfo(Xs)
            shard_sizes.append(n_samples_i)
            assert bidx == i

        n_samples = sum(shard_sizes)
        indptr = np.cumsum(shard_sizes)

        n_valid = int(n_samples * TEST_SIZE)
        n_train = n_samples - n_valid

        if self.is_valid:
            X, y = _alloc(n_valid, n_features, self.device)
            beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(
                indptr, 0, True
            )
        else:
            X, y = _alloc(n_train, n_features, self.device)
            beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(
                indptr, 0, False
            )

        prev = 0
        fds = []
        futures = []
        for bidx in range(beg_idx, end_idx):
            Xs = X_shards_i[bidx]
            _, n_samples_i, n_features, bidx, sidx = get_pinfo(Xs)
            if bidx == beg_idx:
                begin = 0
            else:
                begin = beg_in_shard

            if bidx == end_idx - 1:
                end = end_in_shard
            else:
                end = n_samples_i

            out_end = prev + end - begin
            fd_x = kvikio.CuFile(Xs, "r")
            offset_bytes = begin * n_features * X.itemsize
            X_fut = fd_x.pread(
                X[prev:out_end], X[prev:out_end].nbytes, file_offset=offset_bytes
            )

            fd_y = kvikio.CuFile(ys, "r")
            offset_bytes = begin * y.itemsize
            y_fut = fd_y.pread(
                X[prev:out_end], y[prev:out_end].nbytes, file_offset=offset_bytes
            )

            fds.append((fd_x, fd_y))
            futures.append((X_fut, y_fut))

        for X_fut, y_fut in futures:
            X_fut.get()
            y_fut.get()

        for X_fd, y_fd in fds:
            X_fd.close()
            y_fd.close()

        return X, y

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self.X_shards[i], self.y_shards[i]

        if self.split:
            X, y = self.get_with_valid(i)
        else:
            X, y = load_Xy(X_path, y_path, self.device)

        return X, y

    @property
    def n_batches(self) -> int:
        return len(self.X_shards)


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
