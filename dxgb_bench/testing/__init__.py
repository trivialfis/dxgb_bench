# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import os
import shutil
from functools import cache, wraps
from typing import Any, Callable, Literal, ParamSpec, SupportsFloat, TypeVar

import numpy as np


@cache
def has_cuda() -> bool:
    try:
        from cuda.bindings import runtime as cudart

        status, res = cudart.cudaGetDevice()
        cudart.cudaGetLastError()
        return status == cudart.cudaError_t.cudaSuccess
    except (ImportError, RuntimeError):
        return False


Device = Literal["cpu", "cuda"]


def devices() -> list[Device]:
    if has_cuda():
        return ["cpu", "cuda"]
    return ["cpu"]


def formats() -> list[str]:
    if has_cuda():
        return ["npy", "kio"]
    return ["npy"]


def assert_array_allclose(
    a: np.ndarray, b: np.ndarray, rtol: SupportsFloat = 1e-7
) -> None:
    if has_cuda():
        import cupy as cp

        if isinstance(a, cp.ndarray):
            a = a.get()
        if isinstance(b, cp.ndarray):
            b = b.get()
    np.testing.assert_allclose(a, b, rtol=float(rtol))


def make_tmp(idx: int) -> str:
    tmpdir = f"./tmp-{idx}"
    if os.path.exists(tmpdir):
        shutil.rmtree(tmpdir)
    os.mkdir(tmpdir)
    return tmpdir


def cleanup_tmp(tmpdirs: list[str]) -> None:
    for tmpdir in tmpdirs:
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)


class TmpDir:
    """Used for debugging. We can replace it with the :py:mod:`tempfile` if needed."""

    def __init__(self, n_dirs: int, delete: bool) -> None:
        self.n_dirs = n_dirs
        self.delete = delete

    def __enter__(self) -> list[str]:
        self.outdirs = [make_tmp(i) for i in range(self.n_dirs)]
        return self.outdirs

    def __exit__(self, *args: Any) -> None:
        if self.delete:
            cleanup_tmp(self.outdirs)
            del self.outdirs


P = ParamSpec("P")
R = TypeVar("R")


def chdir(path: str) -> Callable:
    """Decorator for changing directory temporarily."""

    def chdir(func: Callable[P, R]) -> Callable[P, R]:
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> R:
            with Chdir(path):
                return func(*args, **kwargs)

        return inner

    return chdir


class Chdir:
    def __init__(self, path: str) -> None:
        self.path = path
        self.curdir = os.path.normpath(os.path.abspath(os.path.curdir))

    def __enter__(self) -> None:
        os.chdir(self.path)

    def __exit__(self, *args: Any) -> None:
        os.chdir(self.curdir)
