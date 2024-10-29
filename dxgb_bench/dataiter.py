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
            n_bytes = fd.read(X)
            assert n_bytes == X.nbytes
        with kvikio.CuFile(yp, "r") as f:
            n_bytes = f.read(y)
            assert n_bytes == y.nbytes

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
fname_pattern = r"([X|y])_(\d+)_(\d+)-(\d+)-(\d+).npa"


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
            # Assuming the paths are sorted.
            xn, xr, _, xbidx, xsidx = get_pinfo(X_files[i])
            yn, yr, _, ybidx, ysidx = get_pinfo(y_files[i])
            assert xr == yr
            assert xbidx == ybidx
            assert xsidx == ysidx
            X, y = load_Xy(X_files[i], y_files[i], device)
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


def finish_futures(
    futures: list[tuple[kvikio.IOFuture, kvikio.IOFuture]],
) -> None:
    for X_fut, y_fut in futures:
        X_fut.get()
        y_fut.get()

def close_fds(fds: list[tuple[kvikio.CuFile, kvikio.CuFile]]) -> None:
    for X_fd, y_fd in fds:
        X_fd.close()
        y_fd.close()



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
            assert os.path.basename(fname[1]).startswith("y"), fname[1]

        def key(name: str) -> int:
            name = os.path.basename(name)
            mat = re.search(fname_pattern, name)
            assert mat is not None, name
            shard = mat.group(5)
            return int(shard)

        X_shards_sorted = {}
        for batch_idx, shards in X_shards.items():
            shards = sorted(shards, key=key)
            X_shards_sorted[batch_idx] = shards

        y_shards_sorted = {}
        for batch_idx, shards in y_shards.items():
            shards = sorted(shards, key=key)
            y_shards_sorted[batch_idx] = shards

        self.X_shards = X_shards_sorted
        self.y_shards = y_shards_sorted

        assert len(self.X_shards) == len(self.y_shards)

        self.split = split
        self.is_valid = is_valid
        self.device = device

    def get_shard_indptr(self, i: int) -> npt.NDArray[np.int64]:
        X_shards_i = self.X_shards[i]
        y_shards_i = self.y_shards[i]

        shard_sizes = [0]
        for Xs, ys in zip(X_shards_i, y_shards_i):
            _, n_samples_i, n_features, bidx, sidx = get_pinfo(Xs)
            shard_sizes.append(n_samples_i)
            assert bidx == i
        indptr = np.cumsum(shard_sizes)
        return indptr

    def get_with_valid(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Function to read the data for training and validation."""
        X_shards_i = self.X_shards[i]
        y_shards_i = self.y_shards[i]

        indptr = self.get_shard_indptr(i)
        n_samples = indptr[-1]

        n_train, n_valid = get_valid_sizes(n_samples)
        # Span of the validation set.
        beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 0)
        _, _, n_features, _, _ = get_pinfo(X_shards_i[0])

        if self.is_valid:
            X, y = _alloc(n_valid, n_features, self.device)
        else:
            X, y = _alloc(n_train, n_features, self.device)

        def read_xy_shard(
            begin: int,
            Xs: str,
            ys: str,
            prev: int,
            out_end: int,
            X: npt.NDArray[np.float32],
            y: npt.NDArray[np.float32],
        ) -> tuple[kvikio.IOFuture, kvikio.CuFile, kvikio.IOFuture, kvikio.CuFile]:
            X_fd = kvikio.CuFile(Xs, "r")
            offset_bytes = int(begin * n_features * X.itemsize)
            X_fut = X_fd.pread(
                X[prev:out_end], X[prev:out_end].nbytes, file_offset=offset_bytes
            )

            y_fd = kvikio.CuFile(ys, "r")
            offset_bytes = int(begin * y.itemsize)
            y_fut = y_fd.pread(
                y[prev:out_end], y[prev:out_end].nbytes, file_offset=offset_bytes
            )
            return X_fut, X_fd, y_fut, y_fd

        prev = 0
        fds = []
        futures = []

        def read_fst_part(Xs: str, ys: str, beg_in_shard: int) -> int:
            begin, end = 0, beg_in_shard
            out_end = prev + end - begin
            diff = out_end - prev
            if diff > 0:
                X_fut, X_fd, y_fut, y_fd = read_xy_shard(
                    begin, Xs, ys, prev, out_end, X, y
                )

                fds.append((X_fd, y_fd))
                futures.append((X_fut, y_fut))
                return out_end - prev
            return 0

        def read_snd_part(Xs: str, ys: str, end_in_shard: int) -> int:
            begin, end = end_in_shard, n_samples_i
            out_end = prev + end - begin
            diff = out_end - prev
            if diff > 0:
                X_fut, X_fd, y_fut, y_fd = read_xy_shard(
                    begin, Xs, ys, prev, out_end, X, y
                )

                fds.append((X_fd, y_fd))
                futures.append((X_fut, y_fut))
                return out_end - prev
            return 0

        # Read each shard
        if self.is_valid:
            for sidx in range(beg_idx, end_idx + 1):
                Xs = X_shards_i[sidx]
                ys = y_shards_i[sidx]
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
                X_fut, X_fd, y_fut, y_fd = read_xy_shard(
                    begin, Xs, ys, prev, out_end, X, y
                )

                fds.append((X_fd, y_fd))
                futures.append((X_fut, y_fut))
                prev += out_end - prev
            finish_futures(futures)
            close_fds(fds)
            return X, y

        for sidx in range(len(X_shards_i)):
            Xs = X_shards_i[sidx]
            ys = y_shards_i[sidx]
            _, n_samples_i, n_features, bidx, sidx = get_pinfo(Xs)

            if sidx >= beg_idx and sidx <= end_idx:
                # May or may not read the full shard
                if beg_idx == end_idx:
                    # need to split up the shard and read it twice
                    # Read first half
                    prev += read_fst_part(Xs, ys, beg_in_shard)
                    # Read second half
                    prev += read_snd_part(Xs, ys, end_in_shard)
                    continue
                # The validation data spans across more than one shards.
                if sidx > beg_idx and sidx < end_idx:
                    # read full shard
                    begin, end = 0, n_samples_i
                    out_end = prev + end - begin
                    X_fut, X_fd, y_fut, y_fd = read_xy_shard(
                        begin, Xs, ys, prev, out_end, X, y
                    )

                    fds.append((X_fd, y_fd))
                    futures.append((X_fut, y_fut))
                    prev += out_end - prev
                    continue
                if sidx == beg_idx:
                    assert sidx <= end_idx
                    # Read first half
                    prev += read_fst_part(Xs, ys, beg_in_shard)
                    continue
                if sidx == end_idx:
                    assert sidx > beg_idx
                    # Read second half
                    prev += read_snd_part(Xs, ys, end_in_shard)
                    continue
            else:
                # Read full shard
                begin, end = 0, n_samples_i
                out_end = prev + end - begin
                X_fut, X_fd, y_fut, y_fd = read_xy_shard(
                    begin, Xs, ys, prev, out_end, X, y
                )

                fds.append((X_fd, y_fd))
                futures.append((X_fut, y_fut))
                prev += out_end - prev

        finish_futures(futures)
        close_fds(fds)

        return X, y

    @override
    def get(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self.X_shards[i], self.y_shards[i]

        if self.split:
            X, y = self.get_with_valid(i)
        else:
            indptr = self.get_shard_indptr(i)
            n_samples = indptr[-1]
            _, n_samples_i, n_features, bidx, sidx = get_pinfo(X_path[0])
            X, y = _alloc(n_samples, n_features, self.device)
            futures = []
            fds = []

            prev = 0
            for Xs, ys in zip(X_path, y_path):
                _, n_samples_i, n_features, bidx, sidx = get_pinfo(Xs)

                out_end = prev + n_samples_i

                X_fd = kvikio.CuFile(Xs, "r")
                X_fut = X_fd.pread(X[prev:out_end], X[prev:out_end].nbytes)

                y_fd = kvikio.CuFile(ys, "r")
                y_fut = y_fd.pread(y[prev:out_end], y[prev:out_end].nbytes)

                prev += n_samples_i

                futures.append((X_fut, y_fut))
                fds.append((X_fd, y_fd))

            finish_futures(futures)
            close_fds(fds)

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
