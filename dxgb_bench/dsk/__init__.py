# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import math
import os

import numpy as np
from dask import array as da
from dask import dataframe as dd
from dask.distributed import Client, Future, wait
from dask.distributed.comm import parse_host_port

from ..datasets.generated import make_dense_regression as mdr
from ..datasets.generated import save_Xy
from ..utils import fprint


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
    fmt: str,
) -> None:
    saveto = os.path.abspath(os.path.expanduser(saveto))
    client.restart()
    if not os.path.exists(saveto):
        os.mkdir(saveto)

    workers = list(client.scheduler_info()["workers"].keys())
    n_workers = len(workers)

    def rmtree(h: str) -> None:
        fprint(f"rmtree: {saveto}")
        if os.path.exists(saveto):
            import shutil

            fprint(f"removing: {saveto}")
            shutil.rmtree(saveto)
        os.mkdir(saveto)

    futures: list[Future] = []

    hosts = set()
    for w in workers:
        host, _ = parse_host_port(w)
        hosts.add(host)

    for i in range(n_workers):
        host, _ = parse_host_port(workers[i])
        if host in hosts:
            print("submit:", host, workers[i])
            fut = client.submit(rmtree, host, workers=workers[i])
            futures.append(fut)
            hosts.remove(host)

    client.gather(futures)

    def make(n_samples: int, batch_idx: int, seed: int) -> int:
        X, y = mdr(
            device=device,
            n_samples=n_samples,
            n_features=n_features,
            sparsity=0.0,
            random_state=seed,
        )
        save_Xy(X, y, batch_idx, fmt, [saveto])
        return n_samples

    if n_samples > n_workers * 8:
        n_tasks = n_workers * 8
    else:
        n_tasks = n_workers

    n_samples_per_task = int(math.ceil(n_samples / n_tasks))
    last = 0

    futures = []
    for i in range(n_tasks):
        batch_size = min(n_samples_per_task, n_samples - last)
        fut = client.submit(make, batch_size, i, last, workers=[workers[i % n_workers]])
        last += batch_size
        futures.append(fut)

    assert sum(client.gather(futures)) == n_samples


def load_dense_gather(
    client: Client, device: str, loadfrom: str, local_test: bool
) -> tuple[da.Array, da.Array]:
    from ..dataiter import get_file_paths_local, get_pinfo

    loadfrom = os.path.expanduser(loadfrom)

    def get_shape(batch_idx: int) -> list[tuple[str, str, tuple[int, int]]]:
        if local_test:
            path = os.path.join(loadfrom, str(batch_idx))
        else:
            path = loadfrom

        X, y = get_file_paths_local(path)
        print(X, y)
        shapes = []
        for xp in X:
            pinfo = get_pinfo(xp)
            shapes.append((pinfo.n_samples, pinfo.n_features))
        return list(zip(X, y, shapes))

    workers = list(client.scheduler_info()["workers"].keys())
    n_workers = len(workers)
    print(f"n_workers: {n_workers}")

    futures = []
    for i in range(n_workers):
        fut = client.submit(get_shape, i, workers=[workers[i]])
        futures.append(fut)
    # workers[local_files[X, y, shape]]
    names_shapes_local: list[list[tuple[str, str, tuple[int, int]]]] = client.gather(
        futures
    )

    def load1(X_path: str, y_path: str) -> np.ndarray:
        from ..dataiter import load_Xy

        X, y = load_Xy(X_path, y_path, device)
        y = y.reshape(X.shape[0], 1)

        # merge X and y for easier return.
        if device.startswith("cuda"):
            import cupy as cp

            Xy = cp.append(X, y, axis=1)
        else:
            Xy = np.append(X, y, axis=1)
        return Xy

    arrays = []
    for i in range(len(names_shapes_local)):
        for j in range(len(names_shapes_local[i])):
            Xp: str = names_shapes_local[i][j][0]
            yp: str = names_shapes_local[i][j][1]
            s: tuple[int, int] = names_shapes_local[i][j][2]
            fut = client.submit(load1, Xp, yp, workers=[workers[i]])
            arrays.append(
                da.from_delayed(fut, shape=(s[0], s[1] + 1), dtype=np.float32)
            )

    Xy = da.concatenate(arrays, axis=0)
    [Xy] = client.persist([Xy])
    wait([Xy])
    res = client.compute(Xy.shape)
    if hasattr(res, "result"):
        print("Shape:", res.result())
    else:
        print("Shape:", res)
    return Xy[:, :-1], Xy[:, -1]
