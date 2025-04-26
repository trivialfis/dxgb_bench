# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

from functools import cache
from typing import Literal

import numpy as np


@cache
def has_cuda() -> bool:
    try:
        from cuda import cudart

        status, res = cudart.cudaGetDevice()
        cudart.cudaGetLastError()
        return status == cudart.cudaError_t.cudaSuccess
    except ImportError:
        return False


Device = Literal["cpu", "cuda"]


def devices() -> list[Device]:
    if has_cuda():
        return ["cpu", "cuda"]
    return ["cpu"]


def assert_array_allclose(a: np.ndarray, b: np.ndarray) -> None:
    if has_cuda():
        import cupy as cp

        if isinstance(a, cp.ndarray):
            a = a.get()
        if isinstance(b, cp.ndarray):
            b = b.get()
    np.testing.assert_allclose(a, b)
