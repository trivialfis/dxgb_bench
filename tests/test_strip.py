"""Copyright (c) 2025, Jiaming Yuan.  All rights reserved."""

from __future__ import annotations

import os
from itertools import product

import numpy as np
import pytest

from dxgb_bench.dataiter import get_valid_sizes
from dxgb_bench.strip import Strip, get_shard_ids
from dxgb_bench.testing import (
    Device,
    assert_array_allclose,
    cleanup_tmp,
    devices,
    formats,
    make_tmp,
)


def assert_dirs_exist(tmpdirs: list[str]) -> None:
    for tmpdir in tmpdirs:
        dirname = os.path.join(tmpdir, "X")
        assert os.path.exists(dirname)


def find_shard_ids(indptr: np.ndarray, fold: int) -> tuple[int, int, int, int]:
    n_train, n_valid = get_valid_sizes(indptr[-1])
    begin = n_valid * fold
    end = begin + n_valid
    return get_shard_ids(indptr, begin, end)


def test_shard_ids() -> None:
    a = [0, 3, 3]  # 2 shards
    indptr = np.cumsum(a)

    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 0)
    assert beg_idx == 0
    assert end_idx == 0
    assert beg_in_shard == 0
    assert end_in_shard == 1

    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 1)
    assert beg_idx == 0
    assert end_idx == 0
    assert beg_in_shard == 1
    assert end_in_shard == 2

    a = [0, 5, 5]  # 2 shards
    indptr = np.cumsum(a)
    n_train, n_valid = get_valid_sizes(indptr[-1])
    assert n_train == 8
    assert n_valid == 2
    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 2)
    assert beg_idx == 0
    assert end_idx == 1
    assert beg_in_shard == 4
    assert end_in_shard == 1

    indptr = np.array([0, 3, 6])
    beg_idx, beg_in_shard, end_idx, end_in_shard = get_shard_ids(indptr, 0, 6)
    assert beg_idx == 0
    assert end_idx == 1
    assert beg_in_shard == 0
    assert end_in_shard == 3


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_single(device: Device, fmt: str) -> None:
    tmpdir0 = make_tmp(0)
    X_out = Strip("X", [tmpdir0], fmt, device)
    for batch_idx in range(2):
        array = np.arange(0, 100, dtype=np.float32)
        X_out.write(array, batch_idx)

    dirname = os.path.join(tmpdir0, "X")
    assert os.path.exists(dirname)
    pinfo = X_out.list_file_info()
    assert len(pinfo) == 2

    X_out = Strip("X", [tmpdir0], fmt, device)
    pinfo = X_out.list_file_info()
    assert len(pinfo) == 2
    for batch_idx in range(2):
        b = X_out.read(batch_idx, None, None).squeeze()
        array = np.arange(0, 100, dtype=np.float32)
        assert_array_allclose(array, b)

    cleanup_tmp([tmpdir0])


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_stripping(device: Device, fmt: str) -> None:
    tmpdirs = [make_tmp(i) for i in range(3)]
    X_out = Strip("X", tmpdirs, fmt, device)
    n_batches = 4
    for batch_idx in range(n_batches):
        array = np.arange(0 + batch_idx, 100 + batch_idx, dtype=np.float32)
        X_out.write(array, batch_idx)

    assert_dirs_exist(tmpdirs)

    pinfo = X_out.list_file_info()
    assert len(pinfo) == n_batches

    for info in pinfo:
        assert info.n_samples == 100
        assert info.n_features == 1

    X_out = Strip("X", tmpdirs, fmt, device)
    for batch_idx in range(n_batches):
        b = X_out.read(batch_idx, None, None).squeeze()
        array = np.arange(0 + batch_idx, 100 + batch_idx, dtype=np.float32)
        assert_array_allclose(array, b)

    cleanup_tmp(tmpdirs)


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_stripping_less(device: Device, fmt: str) -> None:
    n_dirs = 8
    tmpdirs = [make_tmp(i) for i in range(n_dirs)]
    X_out = Strip("X", tmpdirs, fmt, device)

    n_batches = 4
    n_samples = 5
    for batch_idx in range(n_batches):
        array = np.arange(0 + batch_idx, n_samples + batch_idx, dtype=np.float32)
        X_out.write(array, batch_idx)

    assert_dirs_exist(tmpdirs)

    X_out = Strip("X", tmpdirs, fmt, device)
    for batch_idx in range(n_batches):
        b = X_out.read(batch_idx, None, None).squeeze()
        array = np.arange(0 + batch_idx, n_samples + batch_idx, dtype=np.float32)
        assert_array_allclose(array, b)

    cleanup_tmp(tmpdirs)


def run_subset_tests(n_dirs: int, device: Device, fmt: str) -> None:
    tmpdirs = [make_tmp(i) for i in range(n_dirs)]
    X_out = Strip("X", tmpdirs, fmt, device)
    n_batches = 2
    n_samples = 100
    for batch_idx in range(2):
        array = np.arange(0 + batch_idx, n_samples + batch_idx, dtype=np.float32)
        X_out.write(array, batch_idx)

    X_out = Strip("X", tmpdirs, fmt, device)
    pinfo = X_out.list_file_info()
    assert len(pinfo) == n_batches

    for batch_idx in range(n_batches):
        b = X_out.read(batch_idx, n_samples // 4, n_samples // 2).squeeze()
        assert b.size == n_samples // 4
        a = np.arange(
            n_samples // 4 + batch_idx,
            n_samples // 4 + b.size + batch_idx,
            dtype=np.float32,
        )
        assert_array_allclose(a, b)

    cleanup_tmp(tmpdirs)


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_single_subset(device: Device, fmt: str) -> None:
    run_subset_tests(1, device, fmt)


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_stripping_subset(device: Device, fmt: str) -> None:
    run_subset_tests(3, device, fmt)
