"""Copyright (c) 2025, Jiaming Yuan.  All rights reserved."""

import os
import shutil

import numpy as np
import pytest

from dxgb_bench.strip import Strip
from dxgb_bench.testing import Device, assert_array_allclose, devices


def make_tmp(idx: int) -> str:
    """Used for debugging."""
    tmpdir = f"./tmp-{idx}"
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.mkdir(tmpdir)
    return tmpdir


@pytest.mark.parametrize("device", devices())
def test_single_npy(device: Device) -> None:
    tmpdir0 = make_tmp(0)
    X_out = Strip("X", [tmpdir0], "npy", device)
    for batch_idx in range(2):
        array = np.arange(0, 100, dtype=np.float32)
        X_out.write(array, batch_idx)

    dirname = os.path.join(tmpdir0, "X")
    assert os.path.exists(dirname)
    pinfo = X_out.list_file_info()
    assert len(pinfo) == 2

    X_out = Strip("X", [tmpdir0], "npy", device)
    pinfo = X_out.list_file_info()
    assert len(pinfo) == 2
    for batch_idx in range(2):
        b = X_out.read(batch_idx, None, None)
        array = np.arange(0, 100, dtype=np.float32)
        assert_array_allclose(array, b)
