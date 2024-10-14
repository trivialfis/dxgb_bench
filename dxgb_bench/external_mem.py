from __future__ import annotations

import argparse
import gc
import os
from typing import Callable, List, Protocol, Tuple

import cupy as cp
import numpy as np
import rmm
import xgboost as xgb
from rmm.allocators.cupy import rmm_cupy_allocator
from xgboost.callback import TrainingCheckPoint

from .datasets.generated import make_dense_regression
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

    def next(self, input_data: Callable) -> bool:
        print("Next:", self._it, flush=True)
        if self._it == len(self._file_paths):
            return False

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
        return True

    def reset(self) -> None:
        print("Reset:", flush=True)
        self._it = 0
        gc.collect()


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


n_features = 512


def setup_rmm() -> None:
    print("Use `CudaAsyncMemoryResource`.", flush=True)
    # rmm.reinitialize(pool_allocator=True, initial_pool_size=0)
    use_rmm_pool = False
    if use_rmm_pool:
        mr = rmm.mr.PoolMemoryResource()
    else:
        # status, free, total = cudart.cudaMemGetInfo()
        # assert status == cudart.cudaError_t.cudaSuccess
        # use = int(free * 0.95)
        mr = rmm.mr.CudaAsyncMemoryResource()
        mr = rmm.mr.PoolMemoryResource(mr)
        mr = rmm.mr.LoggingResourceAdaptor(mr, log_file_name="rmm_log")
    rmm.mr.set_current_device_resource(mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def run_external_memory(
    tmpdir: str,
    reuse: bool,
    on_host: bool,
    n_samples_per_batch: int,
    n_bins: int,
    n_batches: int,
    n_rounds: int,
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
            {
                "tree_method": "hist",
                "max_depth": 6,
                "device": "cuda",
                "max_bin": n_bins,
            },
            Xy,
            num_boost_round=n_rounds,
            callbacks=[Progress(n_rounds)],
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
