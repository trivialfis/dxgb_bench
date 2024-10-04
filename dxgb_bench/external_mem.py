from __future__ import annotations

import argparse
import ctypes
import gc
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Callable, List, Protocol, Tuple

import cupy as cp
import numpy as np
import rmm
import xgboost as xgb
from cuda import cudart
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
        *,
        n_batches: int,
        on_host: bool,
        is_ext: bool,
        device: str,
        split: bool,
        is_eval: bool,
        on_the_fly: bool,
        sparsity: float,
        n_samples_per_batch: int,
        n_features: int,
        tmpdir: str | None = None,
    ) -> None:
        if not on_the_fly:
            assert tmpdir is not None
            with Timer("ExtQdm", "make_batches"):
                self._file_paths = make_batches(
                    n_samples_per_batch, n_features, n_batches, True, tmpdir
                )
        else:
            self._file_paths = [("", "")] * n_batches

        self._it = 0
        self._device = device
        self._split = split
        self._is_eval = is_eval

        self._fly = on_the_fly
        self._n_samples_per_batch = n_samples_per_batch
        self._n_features = n_features
        self._sparsity = sparsity

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
            X, y = make_dense_regression(
                self._device,
                self._n_samples_per_batch,
                self._n_features,
                random_state=self._it,
                sparsity=self._sparsity,
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
        gc.collect()


def make_reg_c(
    is_cuda: bool, n_samples_per_batch: int, seed: int, sparsity: float
) -> Tuple[np.ndarray, np.ndarray]:
    path = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(__file__))), "libdxgbbench.so"
    )
    _lib = ctypes.cdll.LoadLibrary(path)
    if is_cuda:
        X = cp.empty(shape=(n_samples_per_batch, n_features), dtype=np.float32)
        y = cp.empty(shape=(n_samples_per_batch,), dtype=np.float32)
        X_ptr = ctypes.cast(
            X.__cuda_array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
        y_ptr = ctypes.cast(
            y.__cuda_array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
    else:
        X = np.empty(shape=(n_samples_per_batch, n_features), dtype=np.float32)
        y = np.empty(shape=(n_samples_per_batch,), dtype=np.float32)
        X_ptr = ctypes.cast(
            X.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )
        y_ptr = ctypes.cast(
            y.__array_interface__["data"][0], ctypes.POINTER(ctypes.c_float)
        )

    _lib.MakeDenseRegression(
        ctypes.c_bool(is_cuda),
        ctypes.c_int64(n_samples_per_batch),
        ctypes.c_int64(n_features),
        ctypes.c_double(sparsity),
        ctypes.c_int64(seed),
        X_ptr,
        y_ptr,
    )
    return X, y


def make_dense_regression(
    device: str, n_samples: int, n_features: int, *, sparsity: float, random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Make dense synthetic data for regression."""
    try:
        X, y = make_reg_c(
            device == "cuda", n_samples, random_state * n_samples, sparsity=sparsity
        )
        return X, y
    except Exception as e:
        print(f"Failed to generate using the C extension. {e}")

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
                make_regression, n_samples_cur, start + random_state * n_samples
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
    if device == "cuda":
        return cp.array(X), cp.array(y)
    return X, y


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
            "cpu",
            n_samples_per_batch,
            n_features=n_features,
            random_state=i,
            sparsity=0.0,
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


def setup_rmm() -> None:
    print("Use `CudaAsyncMemoryResource`.", flush=True)
    # rmm.reinitialize(pool_allocator=True, initial_pool_size=0)
    use_rmm_pool = False
    if use_rmm_pool:
        mr = rmm.mr.PoolMemoryResource()
    else:
        status, free, total = cudart.cudaMemGetInfo()
        assert status == cudart.cudaError_t.cudaSuccess
        use = int(free * 0.95)
        mr = rmm.mr.CudaAsyncMemoryResource(
            initial_pool_size=use, release_threshold=use, enable_ipc=False
        )
    rmm.mr.set_current_device_resource(mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def run_external_memory(
    tmpdir: str,
    reuse: bool,
    on_host: bool,
    n_samples_per_batch: int,
    n_batches: int,
    sparsity: float,
) -> xgb.Booster:
    setup_rmm()

    with Timer("ExtSparse", "make_batches"):
        it = EmTestIterator(
            n_batches=n_batches,
            on_host=on_host,
            is_ext=True,
            device="cpu",
            split=False,
            is_eval=False,
            on_the_fly=False,
            sparsity=sparsity,
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            tmpdir=tmpdir,
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


def run_ext_qdm(
    tmpdir: str,
    reuse: bool,
    n_bins: int,
    n_samples_per_batch: int,
    n_batches: int,
    device: str,
    n_rounds: int,
    sparsity: float,
    on_the_fly: bool,
    validation: bool,
) -> xgb.Booster:
    setup_rmm()
    with Timer("ExtQdm", "ExtMemQuantileDMatrix-Train"):
        it_train = EmTestIterator(
            n_batches=n_batches,
            is_ext=True,
            on_host=True,
            device=device,
            split=validation,
            is_eval=False,
            on_the_fly=on_the_fly,
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            sparsity=sparsity,
        )
        Xy_train = xgb.core.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)

    if validation:
        with Timer("ExtQdm", "ExtMemQuantileDMatrix-Valid"):
            it_valid = EmTestIterator(
                n_batches=n_batches,
                is_ext=True,
                on_host=True,
                device=device,
                split=validation,
                is_eval=True,
                on_the_fly=on_the_fly,
                n_samples_per_batch=n_samples_per_batch,
                n_features=n_features,
                sparsity=sparsity,
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
            verbose_eval=False,
            callbacks=[Progress(n_rounds)],
        )
    return booster


class TestBody(Protocol):
    @property
    def n_bins(self) -> int: ...
    @property
    def n_samples_per_batch(self) -> int: ...
    @property
    def n_batches(self) -> int: ...
    @property
    def device(self) -> str: ...
    @property
    def on_the_fly(self) -> bool: ...
    @property
    def reuse(self) -> bool: ...
    @property
    def tmpdir(self) -> str: ...


class MakeExtQdmMixIn:
    def make_iter(self: TestBody) -> xgb.DMatrix:
        with Timer("ExtQdm", "ExtMemQuantileDMatrix-Train"):
            it_train = EmTestIterator(
                n_batches=self.n_batches,
                is_ext=True,
                on_host=True,
                device=self.device,
                split=False,
                is_eval=False,
                on_the_fly=self.on_the_fly,
                n_samples_per_batch=self.n_samples_per_batch,
                n_features=n_features,
                sparsity=0.0,
            )
            Xy_train = xgb.ExtMemQuantileDMatrix(it_train, max_bin=self.n_bins)
        return Xy_train


class SetupRmmMixIn:
    def __init__(self) -> None:
        setup_rmm()


class TestInference(MakeExtQdmMixIn, SetupRmmMixIn):
    def __init__(
        self,
        model_path: str,
        predict_type: str,
        n_bins: int,
        n_samples_per_batch: int,
        n_batches: int,
        device: str,
        on_the_fly: bool,
        reuse: bool,
        tmpdir: str,
    ) -> None:
        self.model_path = model_path
        self.predict_type = predict_type

        self.n_bins: int = n_bins
        self.n_samples_per_batch: int = n_samples_per_batch
        self.n_batches: int = n_batches
        self.device: str = device
        self.on_the_fly: bool = on_the_fly

        self.reuse = reuse
        self.tmpdir = tmpdir

        super().__init__()

    def run(self) -> None:
        Xy = self.make_iter()
        booster = xgb.Booster(model_file=self.model_path)
        booster.set_param({"device": self.device})
        with Timer("inference", self.predict_type):
            if self.predict_type == "contribs":
                booster.predict(Xy, pred_contribs=True)
            else:
                booster.predict(Xy, pred_interactions=True)


def run_inference(
    tmpdir: str,
    reuse: bool,
    n_bins: int,
    n_batches: int,
    n_samples_per_batch: int,
    device: str,
    on_the_fly: bool,
    args: argparse.Namespace,
) -> None:
    TestInference(
        model_path=args.model,
        predict_type=args.predict_type,
        n_bins=n_bins,
        n_samples_per_batch=n_samples_per_batch,
        n_batches=n_batches,
        device=device,
        on_the_fly=on_the_fly,
        reuse=reuse,
        tmpdir=tmpdir,
    ).run()
