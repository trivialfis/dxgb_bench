import gc
import os
from concurrent.futures import ThreadPoolExecutor
from time import time
from typing import Any, Callable, List, Tuple

import numpy as np
import rmm
import xgboost as xgb
from sklearn.datasets import make_regression
from xgboost.testing.data import make_dense_regression


class EmTestIterator(xgb.DataIter):
    """A custom iterator for profiling external memory."""

    def __init__(
        self, file_paths: List[Tuple[str, str]], on_host: bool, is_ext: bool
    ) -> None:
        self._file_paths = file_paths
        self._it = 0
        if is_ext:
            super().__init__(cache_prefix="cache", on_host=on_host)
        else:
            super().__init__()

    def load_file(self) -> Tuple[np.ndarray, np.ndarray]:
        X_path, y_path = self._file_paths[self._it]
        X = np.lib.format.open_memmap(filename=X_path, mode="r")
        y = np.lib.format.open_memmap(filename=y_path, mode="r")
        assert X.shape[0] == y.shape[0]
        return X, y

    def next(self, input_data: Callable) -> int:
        print("Next", flush=True)
        if self._it == len(self._file_paths):
            return 0

        X, y = self.load_file()
        input_data(data=X, label=y)
        self._it += 1
        return 1

    def reset(self) -> None:
        self._it = 0


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
        X, y = make_dense_regression(n_samples_per_batch, n_features=n_features)
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))
        print(f"Saved to {X_path} and {y_path}", flush=True)

    gc.collect()

    return files


def run_external_memory(
    tmpdir: str,
    reuse: bool,
    on_host: bool,
    n_samples_per_batch: int,
    n_batches: int,
) -> xgb.Booster:
    rmm.reinitialize(pool_allocator=True, initial_pool_size=0)

    n_features = 512
    files = make_batches(n_samples_per_batch, n_features, n_batches, reuse, tmpdir)
    it = EmTestIterator(files, on_host=on_host, is_ext=True)
    Xy = xgb.DMatrix(it, missing=np.nan, enable_categorical=False)

    booster = xgb.train(
        {"tree_method": "hist", "max_depth": 6, "device": "cuda"},
        Xy,
        num_boost_round=6,
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
        rmm.reinitialize(
            pool_allocator=True,
            system_memory=True,
            system_memory_headroom_size=16 * 1024 * 1024 * 1024,
        )
    else:
        rmm.reinitialize(pool_allocator=True)

    n_features = 512
    files = make_batches(n_samples_per_batch, n_features, n_batches, reuse, tmpdir)
    it = EmTestIterator(files, is_ext=False, on_host=False)

    start = time()
    Xy = xgb.QuantileDMatrix(it, max_bin=n_bins)
    end = time()
    print("QuantileDMatrix duration:", end - start)

    booster = xgb.train(
        {"tree_method": "hist", "max_depth": 6, "device": "cuda", "max_bin": n_bins},
        Xy,
        num_boost_round=6,
    )
    return booster
