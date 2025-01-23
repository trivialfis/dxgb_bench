# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import math
import os

import numpy as np
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client, wait

from ..datasets.generated import make_dense_regression as mdr
from ..datasets.generated import save_Xy


def make_dense_regression(
    device: str,
    n_samples: int,
    n_features: int,
    random_state: int,
) -> tuple[dd.DataFrame, dd.DataFrame]:
    rng = da.random.default_rng(seed=random_state)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)
    if device.startswith("cuda"):
        X = X.to_backend("cupy")
    y = X.sum(axis=1)
    X_df = dd.from_dask_array(X, columns=[f"f{i}" for i in range(n_features)])
    y_df = dd.from_dask_array(y, columns=["t0"])
    if device.startswith("cuda"):
        X_df = X_df.to_backend("cudf")
        y_df = y_df.to_backend("cudf")
    print(X_df.dtypes, y_df.dtypes)
    return X_df, y_df


def make_dense_regression_scatter(
    client: Client,
    device: str,
    n_samples: int,
    n_features: int,
    saveto: str,
    local_test: bool,
) -> None:
    saveto = os.path.expanduser(saveto)
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    def make(n_samples: int, batch_idx: int, seed: int) -> int:
        X, y = mdr(
            device=device,
            n_samples=n_samples,
            n_features=n_features,
            sparsity=0.0,
            random_state=seed,
        )
        if local_test:
            path = os.path.join(saveto, str(batch_idx))
        else:
            path = saveto
        if not os.path.exists(path):
            os.mkdir(path)

        save_Xy(X, y, 0, [path])
        return n_samples

    workers = client.scheduler_info()["workers"]
    n_workers = len(workers)

    n_samples_per_worker = int(math.ceil(n_samples / n_workers))
    last = 0

    futures = []
    for i in range(n_workers):
        batch_size = min(n_samples_per_worker, n_samples - last)
        fut = client.submit(make, batch_size, i, last)
        last += batch_size
        futures.append(fut)

    assert sum(client.gather(futures)) == n_samples


def load_dense_gather(
    client: Client, device: str, loadfrom: str, local_test: bool
) -> tuple[da.Array, da.Array]:
    from ..dataiter import fname_pattern, get_file_paths_local, get_pinfo, load_all

    loadfrom = os.path.expanduser(loadfrom)

    def get_shape(batch_idx: int) -> tuple[int, int]:
        if local_test:
            path = os.path.join(loadfrom, str(batch_idx))
        else:
            path = loadfrom

        X, y = get_file_paths_local(path)
        print(X, y)
        x, n_samples, n_features, batch_idx, shard_idx = get_pinfo(X[0])
        return n_samples, n_features

    def load(batch_idx: int) -> np.ndarray:
        if local_test:
            path = os.path.join(loadfrom, str(batch_idx))
        else:
            path = loadfrom

        X, y = load_all([loadfrom], device)
        y = y.reshape(X.shape[0], 1)
        if device.startswith("cuda"):
            import cupy as cp

            Xy = cp.append(X, y, axis=1)
        else:
            Xy = np.append(X, y, axis=1)

        assert Xy.shape[0] == X.shape[0]
        assert Xy.shape[1] == X.shape[1] + 1
        return Xy

    workers = client.scheduler_info()["workers"]
    n_workers = len(workers)
    print(f"n_workers: {n_workers}")

    futures = []
    for i in range(n_workers):
        fut = client.submit(get_shape, i)
        futures.append(fut)
    shapes = client.gather(futures)
    print(shapes)
    arrays = []
    for i in range(n_workers):
        fut = client.submit(load, i)
        daarr = da.from_delayed(
            fut, shape=(shapes[i][0], shapes[i][1] + 1), dtype=np.float32
        )
        arrays.append(daarr)
    Xy = da.concatenate(arrays, axis=0)
    [Xy] = client.persist([Xy])
    wait([Xy])
    return Xy[:, :-1], Xy[:, -1]
