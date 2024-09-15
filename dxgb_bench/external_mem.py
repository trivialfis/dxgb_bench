from __future__ import annotations

import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Callable, List, Tuple

import cupy as cp
import numpy as np
import rmm
import xgboost as xgb
from rmm.allocators.cupy import rmm_cupy_allocator
from xgboost.callback import TrainingCheckPoint
from xgboost.compat import concat

from .utils import Progress, Timer

TEST_SIZE = 0.2


def train_test_split(
    X: cp.ndarray, y: cp.ndarray, test_size: float, random_state: int
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    # Only used for profiling, not suitable for real world validation.
    n_samples = X.shape[0]
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test

    X_train = X[:n_train, ...]
    X_test = X[n_train:, ...]

    y_train = y[:n_train]
    y_test = y[n_train:]

    return X_train, X_test, y_train, y_test


class EmTestIterator(xgb.DataIter):
    """A custom iterator for profiling external memory."""

    def __init__(
        self,
        file_paths: List[Tuple[str, str]],
        on_host: bool,
        is_ext: bool,
        device: str,
        split: bool,
        is_eval: bool,
        on_the_fly: bool,
        n_samples_per_batch: int | None = None,
        n_features: int | None = None,
    ) -> None:
        self._file_paths = file_paths
        self._it = 0
        self._device = device
        self._split = split
        self._is_eval = is_eval

        self._fly = on_the_fly
        self._n_samples_per_batch = n_samples_per_batch
        self._n_features = n_features

        if is_ext:
            super().__init__(cache_prefix="cache", on_host=on_host)
        else:
            super().__init__()

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        if self._device != "cpu":
            X = cp.load(X_path)
            y = cp.load(y_path)
        else:
            X = np.lib.format.open_memmap(filename=X_path, mode="r")
            y = np.lib.format.open_memmap(filename=y_path, mode="r")
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        print("Next:", self._it, flush=True)
        if self._it == len(self._file_paths):
            return 0

        gc.collect()

        if self._fly:
            assert self._n_samples_per_batch is not None
            assert self._n_features is not None
            X, y = make_dense_regression_cupy(
                self._n_samples_per_batch, self._n_features, random_state=self._it
            )
        else:
            X, y = self.load_file()

        if self._split:
            X_train, X_valid, y_train, y_valid = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=42
            )
            if self._is_eval:
                input_data(data=X_valid, label=y_valid)
            else:
                input_data(data=X_train, label=y_train)
        else:
            input_data(data=X, label=y)

        self._it += 1
        return 1

    def reset(self) -> None:
        print("Reset:", flush=True)
        self._it = 0


def make_reg_c(n_samples_per_batch: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(__file__))), "libdxgbbench.so"
    )
    _lib = ctypes.cdll.LoadLibrary(path)
    X = np.empty(shape=(n_samples_per_batch, n_features), dtype=np.float32)
    X_ptr = ctypes.cast(
        X.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
    )

    y = np.empty(shape=(n_samples_per_batch,), dtype=np.float32)
    y_ptr = ctypes.cast(
        y.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
    )

    _lib.MakeDenseRegression(
        ctypes.c_int64(n_samples_per_batch),
        ctypes.c_int64(n_features),
        ctypes.c_int64(seed),
        X_ptr,
        y_ptr,
    )
    return X, y


def make_dense_regression(
    n_samples: int, n_features: int, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Make dense synthetic data for regression."""
    n_cpus = os.cpu_count()
    assert n_cpus is not None
    n_threads = min(n_cpus, n_samples)
    start = 0

    def make_regression(
        n_samples_per_batch: int, seed: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        # A custom version of make_regression since sklearn doesn't support np
        # generator.
        rng = np.random.default_rng(seed)
        X = rng.normal(
            loc=0.0, scale=1.5, size=(n_samples_per_batch, n_features)
        ).astype(np.float32)
        err = rng.normal(0.0, scale=0.2, size=(n_samples_per_batch,)).astype(np.float32)
        y = X.sum(axis=1) + err
        return X, y

    futures = []
    n_samples_per_batch = n_samples // n_threads
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            n_samples_cur = n_samples_per_batch
            if i == n_threads - 1:
                n_samples_cur = n_samples - start
            fut = executor.submit(
                make_reg_c, n_samples_cur, start + random_state * n_samples
            )
            start += n_samples_cur
            futures.append(fut)

    X_arr, y_arr = [], []
    for fut in futures:
        X, y = fut.result()
        X_arr.append(X)
        y_arr.append(y)

    def parallel_concat(
        X_arr: List[np.ndarray], y_arr: List[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        with ThreadPoolExecutor(max_workers=n_threads) as executor_1:
            while True:
                X_arr_1 = []
                y_arr_1 = []
                X_futures = []

                for i in range(0, len(X_arr), 2):
                    if i + 1 < len(X_arr):
                        X_fut = executor_1.submit(concat, X_arr[i : i + 2])
                        y_fut = executor_1.submit(concat, y_arr[i : i + 2])
                    else:
                        X_fut = executor_1.submit(lambda x: x, X_arr[i])
                        y_fut = executor_1.submit(lambda x: x, y_arr[i])
                    X_futures.append((X_fut, y_fut))

                for X_fut, y_fut in X_futures:
                    X_arr_1.append(X_fut.result())
                    y_arr_1.append(y_fut.result())

                X_arr = X_arr_1
                y_arr = y_arr_1

                if len(X_arr) == 1 and len(y_arr) == 1:
                    break
        assert len(X_arr) == 1 and len(y_arr) == 1
        return X_arr[0], y_arr[0]

    X, y = parallel_concat(X_arr, y_arr)
    return X, y


def make_dense_regression_cupy(
    n_samples: int, n_features: int, random_state: int
) -> Tuple[cp.ndarray, cp.ndarray]:
    X, y = make_dense_regression(n_samples, n_features, random_state)
    return cp.array(X), cp.array(y)


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    reuse: bool,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []

    if reuse:
        for i in range(n_batches):
            X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
            y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                files = []
                break
            files.append((X_path, y_path))

    if files:
        return files

    assert not files

    for i in range(n_batches):
        X, y = make_dense_regression(
            n_samples_per_batch, n_features=n_features, random_state=i
        )
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))
        print(f"Saved to {X_path} and {y_path}", flush=True)

    gc.collect()

    return files


N_ROUNDS = 128
n_features = 512


def run_external_memory(
    tmpdir: str,
    reuse: bool,
    on_host: bool,
    n_samples_per_batch: int,
    n_batches: int,
) -> xgb.Booster:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=0)

    with Timer("ExtSparse", "make_batches"):
        files = make_batches(n_samples_per_batch, n_features, n_batches, reuse, tmpdir)
        it = EmTestIterator(
            files,
            on_host=on_host,
            is_ext=True,
            device="cpu",
            split=False,
            is_eval=False,
            on_the_fly=False,
        )
    with Timer("ExtSparse", "DMatrix"):
        Xy = xgb.DMatrix(it, missing=np.nan, enable_categorical=False)
    with Timer("ExtSparse", "train"):
        booster = xgb.train(
            {"tree_method": "hist", "max_depth": 6, "device": "cuda"},
            Xy,
            num_boost_round=N_ROUNDS,
            callbacks=[Progress(N_ROUNDS)],
        )
    return booster


def run_over_subscription(
    tmpdir: str,
    reuse: bool,
    n_bins: int,
    n_samples_per_batch: int,
    n_batches: int,
    is_sam: bool,
) -> xgb.Booster:
    if is_sam:
        base_mr = rmm.mr.SamHeadroomMemoryResource(headroom=16 * 1024 * 1024 * 1024)
        mr = rmm.mr.PoolMemoryResource(base_mr)
        rmm.mr.set_current_device_resource(mr)
    else:
        rmm.reinitialize(pool_allocator=True)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    with Timer("OS", "make_batches"):
        files = make_batches(n_samples_per_batch, n_features, n_batches, reuse, tmpdir)
        it = EmTestIterator(
            files,
            is_ext=False,
            on_host=False,
            device="cpu",
            split=False,
            is_eval=False,
            on_the_fly=False,
        )

    with Timer("OS", "QuantileDMatrix"):
        Xy = xgb.QuantileDMatrix(it, max_bin=n_bins)

    with Timer("OS", "Train"):
        booster = xgb.train(
            {
                "tree_method": "hist",
                "max_depth": 6,
                "device": "cuda",
                "max_bin": n_bins,
            },
            Xy,
            num_boost_round=N_ROUNDS,
            callbacks=[Progress(N_ROUNDS)],
        )
    return booster


def run_ext_qdm(
    tmpdir: str,
    reuse: bool,
    n_bins: int,
    n_samples_per_batch: int,
    n_batches: int,
    device: str,
    n_rounds: int,
) -> xgb.Booster:
    base_mr = rmm.mr.CudaAsyncMemoryResource()
    mr = rmm.mr.PoolMemoryResource(base_mr)
    rmm.mr.set_current_device_resource(mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)

    on_the_fly = True

    if not on_the_fly:
        with Timer("ExtQdm", "make_batches"):
            files = make_batches(
                n_samples_per_batch, n_features, n_batches, reuse, tmpdir
            )
    else:
        files = [("", "")] * n_batches

    validation = False

    with Timer("ExtQdm", "ExtMemQuantileDMatrix-Train"):
        it_train = EmTestIterator(
            files,
            is_ext=True,
            on_host=True,
            device=device,
            split=validation,
            is_eval=False,
            on_the_fly=on_the_fly,
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
        )
        Xy_train = xgb.core.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)

    if validation:
        with Timer("ExtQdm", "ExtMemQuantileDMatrix-Valid"):
            it_valid = EmTestIterator(
                files,
                is_ext=True,
                on_host=True,
                device=device,
                split=validation,
                is_eval=True,
                on_the_fly=on_the_fly,
                n_samples_per_batch=n_samples_per_batch,
                n_features=n_features,
            )
            Xy_valid = xgb.core.ExtMemQuantileDMatrix(
                it_train, max_bin=n_bins, ref=Xy_train
            )
        watches = [(Xy_train, "Train"), (Xy_valid, "Valid")]
    else:
        watches = [(Xy_train, "Train")]

    with Timer("ExtQdm", "train"):
        booster = xgb.train(
            {
                "tree_method": "hist",
                "max_depth": 6,
                "max_bin": n_bins,
                "device": device,
            },
            Xy_train,
            num_boost_round=n_rounds,
            evals=watches,
            callbacks=[Progress(n_rounds), TrainingCheckPoint("./models", interval=4)],
        )
    return booster
