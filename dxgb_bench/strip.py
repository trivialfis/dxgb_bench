"""Copyright (c) 2025, Jiaming Yuan.  All rights reserved.

Simple implementation of stripping for arrays. We sometimes run benchmarks on devices
that have multiple disks but don't have RAID-0.

"""

from __future__ import annotations

import dataclasses
import math
import os
import re
from abc import abstractmethod
from bisect import bisect_right
from typing import Any

import numpy as np
from numpy import typing as npt
from typing_extensions import override


def divup(a: int, b: int) -> int:
    return math.ceil(a / b)


class Backend:
    """File IO backend. One instance of this class for each batch of the data. Call
    :py:math:`read` for each shard then use the :py:meth:`get` to get the final array.

    """

    def __init__(self, shape: tuple[int, ...] | None, device: str) -> None:
        self.shape = shape
        self.device = device

    @abstractmethod
    def write(self, array: np.ndarray, fname: str) -> None: ...

    @abstractmethod
    def read(self, fname: str, begin: int | None, end: int | None) -> None:
        """Read data.

        Parameters
        ----------
        fname:
            File name.
        begin:
            Optional beginning sample idx.
        end:
            Optional end sample idx.

        """
        ...

    @abstractmethod
    def get(self) -> np.ndarray:
        """Call this when all shards are read."""
        ...

    @abstractmethod
    def __enter__(self) -> Backend: ...

    @abstractmethod
    def __exit__(self, *args: Any) -> None: ...


class _Npy(Backend):
    @override
    def __enter__(self) -> Backend:
        self._arrays = []
        return self

    @override
    def __exit__(self, *args: Any) -> None:
        pass

    @override
    def write(self, array: np.ndarray, fname: str) -> None:
        np.save(fname, array)

    @override
    def read(self, fname: str, begin: int | None, end: int | None) -> None:
        if self.device == "cpu":
            from numpy import load
        else:
            from cupy import load

        if begin is not None and end is not None:
            a = load(fname, mmap_mode="r")[begin:end]
        elif begin is not None:
            a = load(fname, mmap_mode="r")[begin:]
        elif end is not None:
            a = load(fname, mmap_mode="r")[:end]
        else:
            a = load(fname)

        self._arrays.append(a)

    @override
    def get(self) -> np.ndarray:
        if self.device == "cpu":
            return np.concatenate(self._arrays, axis=0)
        else:
            import cupy as cp

            return cp.concatenate(self._arrays, axis=0)


class _Kvikio(Backend):
    def __init__(self, shape: tuple[int, ...] | None, device: str) -> None:
        super().__init__(shape=shape, device=device)

    @override
    def __enter__(self) -> Backend:
        import kvikio

        self._futures: list[kvikio.IOFuture] = []
        # allocate the array buffer
        if self.shape is not None:
            if self.device == "cpu":
                array = np.empty(shape=self.shape, dtype=np.float32)
            else:
                import cupy as cp

                array = cp.empty(shape=self.shape, dtype=np.float32)
        else:
            array = None
        self._array = array

        return self

    @override
    def __exit__(self, *args: Any) -> None:
        pass

    @override
    def write(self, array: np.ndarray, fname: str) -> None:
        import kvikio

        with kvikio.CuFile(fname, "w") as fd:
            n_bytes = fd.write(array)
            assert n_bytes == array.nbytes

    @override
    def read(self, fname: str, begin: int | None, end: int | None) -> None:
        assert self._array is not None
        import kvikio

        with kvikio.CuFile(fname, "r") as fd:
            n_bytes = fd.read(X)
            assert n_bytes == X.nbytes

    @override
    def get(self) -> np.ndarray:
        for fut in self._futures:
            fut.get()
        return self._array


def dispatch_backend(device: str, fmt: str, shape: tuple[int, ...] | None) -> Backend:
    match fmt:
        case "npy":
            return _Npy(shape, device)
        case "kvi":
            return _Kvikio(shape, device)
        case "npz":
            raise NotImplementedError()
        case _:
            raise ValueError("Invalid format.")
    return _Npy()


@dataclasses.dataclass
class PathInfo:
    """Parse result of a path."""

    name: str  # X|y
    n_samples: int
    n_features: int
    batch_idx: int
    shard_idx: int


def make_file_name(
    shape: tuple[int, ...],
    dirname: str,
    name: str,
    batch_idx: int,
    shard_idx: int,
    fmt: str,
) -> str:
    assert name in ("X", "y")
    base = f"{name}-r{shape[0]}"
    if name == "X":
        fname = f"{base}-c{shape[1]}-b{batch_idx}-s{shard_idx}.{fmt}"
    else:
        # Single column
        assert len(shape) == 1 or shape[1] <= 1
        fname = f"{base}-c1-b{batch_idx}-s{shard_idx}.{fmt}"
    return os.path.join(dirname, fname)


# filename pattern
# X|y-rows-columns-batch-shard.npz
FNAME_PAT = r"([X|y])-r(\d+)-c(\d+)-b(\d+)-s(\d+).[npy|npz|kvi]"


def get_pinfo(path: str) -> PathInfo:
    name = os.path.basename(path)
    mat = re.search(FNAME_PAT, name)
    x, rows_str, cols_str, batch_str, shard_str = mat.groups()
    n_samples, n_features, batch_idx, shard_idx = (
        int(rows_str),
        int(cols_str),
        int(batch_str),
        int(shard_str),
    )
    return PathInfo(x, n_samples, n_features, batch_idx, shard_idx)


def get_shard_ids(
    indptr: npt.NDArray[np.int64], begin: int, end: int
) -> tuple[int, int, int, int]:
    n_samples = indptr[-1]
    assert end < n_samples

    beg_idx: int = bisect_right(indptr, begin) - 1
    end_idx: int = bisect_right(indptr, end) - 1

    beg_in_shard = begin - indptr[beg_idx]
    end_in_shard = end - indptr[end_idx]

    return beg_idx, beg_in_shard, end_idx, end_in_shard


class Strip:
    """Class for mimicking RAID-0 with multiple directories. This class can write and
    read numpy arrays in multiple directories with data sharding. Each strip class
    should correspond to one sequence of the same array, either X or y.

    Parameters
    ----------
    name:
        X or y.
    dirs:
        The directories for sharding.
    fmt:
        Output format, be either one of the {npy, npz, kio}.
    device:
        cpu or cuda.

    """

    def __init__(
        self, name: str, dirs: list[str], fmt: str | None, device: str
    ) -> None:
        self._name = name
        assert self._name in ("X", "y")
        self._dirs = [os.path.join(d, name) for d in sorted(dirs)]

        def get_fmt() -> str:
            fst_dir = self._dirs[0]
            for root, subdirs, files in os.walk(fst_dir):
                for f in files:
                    fmt = f.split(".")[-1]
                    assert fmt in ("npy", "npz", "kio")
                    return fmt
            raise ValueError("Failed to infer file format.")

        if fmt is None:
            fmt = get_fmt()
        self._fmt = fmt
        self._device = device

        self._file_info = self.list_file_info()
        self._batch_key = {f.batch_idx: f for f in self._file_info}

        assert self._fmt in ("npy", "npz", "kio")

    def write(self, array: np.ndarray, batch_idx: int) -> int:
        if array.dtype != np.float32:
            raise TypeError("Only f32 is supported.")

        n_dirs = len(self._dirs)
        n_samples = array.shape[0]
        n_samples_per_shard = divup(n_samples, n_dirs)
        prev = 0
        assert len(array.shape) <= 2

        if len(array.shape) == 1:
            shape = (n_samples, 1)
        else:
            shape = array.shape

        with dispatch_backend(device=self._device, fmt=self._fmt, shape=shape) as hdl:
            for shard_idx, dirname in enumerate(self._dirs):

                if not os.path.exists(dirname):
                    os.mkdir(dirname)

                size = min(n_samples_per_shard, n_samples - prev)
                shard = array[prev : prev + size]
                path_shard = make_file_name(
                    shape=shape,
                    dirname=dirname,
                    name=self._name,
                    batch_idx=batch_idx,
                    shard_idx=shard_idx,
                    fmt=self._fmt,
                )
                hdl.write(shard, path_shard)
                prev += size

        return array.size * array.itemsize

    def list_file_info(self) -> list[PathInfo]:
        """Get a list of files for the first shard."""
        fst_dir = self._dirs[0]
        results = []
        for root, subdirs, files in os.walk(fst_dir):
            assert not subdirs
            for f in files:
                path = os.path.join(root, f)
                pinfo = get_pinfo(path)
                results.append(pinfo)
        return results

    def get_shard_indptr(self, batch_idx: int) -> npt.NDArray[np.int64]:
        shard_i = self._batch_key[batch_idx]

        n_dirs = len(self._dirs)
        n_samples = shard_i.n_samples
        n_samples_per_shard = divup(n_samples, n_dirs)

        shard_sizes = [0]
        prev = 0
        for shard_idx, dirname in enumerate(self._dirs):
            size = min(n_samples_per_shard, n_samples - prev)
            shard_sizes.append(size)
            prev += size

        indptr = np.cumsum(shard_sizes)
        assert indptr[-1] == n_samples
        return indptr

    def read(self, batch_idx: int, begin: int | None, end: int | None) -> np.ndarray:
        """Read a batch of data.

        Parameters
        ----------
        batch_idx:
            The batch index.
        begin:
            Optional range begin, for train test split.
        end:
            Optional range end.

        Returns
        -------
        The read array.

        """
        assert self._dirs and self._fmt
        n_dirs = len(self._device)

        if begin is not None:
            assert end is not None
            n_samples = end - begin
            assert n_samples > 0

            indptr = self.get_shard_indptr(batch_idx)
            beg_shard_idx, beg_in_shard, end_shard_idx, end_in_shard = get_shard_ids(
                indptr, begin, end
            )
        else:
            n_samples = self._batch_key[batch_idx].n_samples
            beg_shard_idx, end_shard_idx = 0, n_dirs - 1
            beg_in_shard, end_in_shard = -1, -1

        shape_res = (n_samples, self._batch_key[batch_idx].n_features)
        shape_orig = (
            self._batch_key[batch_idx].n_samples,
            self._batch_key[batch_idx].n_features,
        )

        with dispatch_backend(
            device=self._device, fmt=self._fmt, shape=shape_res
        ) as hdl:
            for shard_idx, dirname in enumerate(self._dirs):
                path_shard = make_file_name(
                    shape=shape_orig,
                    dirname=dirname,
                    name=self._name,
                    batch_idx=batch_idx,
                    shard_idx=shard_idx,
                    fmt=self._fmt,
                )
                if begin is not None:
                    if shard_idx == beg_shard_idx and shard_idx == end_shard_idx:
                        hdl.read(path_shard, beg_in_shard, end_in_shard)
                    elif beg_shard_idx == shard_idx:
                        hdl.read(path_shard, beg_in_shard, None)
                    elif end_shard_idx == shard_idx:
                        hdl.read(path_shard, None, end_in_shard)
                    elif beg_shard_idx < shard_idx < end_shard_idx:
                        hdl.read(path_shard, None, None)
                    else:
                        pass
                else:
                    hdl.read(path_shard, None, None)
            array = hdl.get()
            assert array.shape[0] == n_samples, (array.shape, n_samples)
        return array
