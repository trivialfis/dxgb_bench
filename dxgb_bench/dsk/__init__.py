# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

from dask import array as da
from dask import dataframe as dd


def make_dense_regression(
    device: str, n_samples: int, n_features: int, random_state: int
) -> tuple[dd.DataFrame, dd.DataFrame]:
    rng = da.random.default_rng(seed=random_state)
    X = rng.normal(size=(n_samples, n_features))
    if device.startswith("cuda"):
        X = X.to_backend("cupy")
    y = X.sum(axis=1)
    X_df = dd.from_dask_array(X, columns=[f"f{i}" for i in range(n_features)])
    y_df = dd.from_dask_array(y, columns=["t0"])
    if device.startswith("cuda"):
        X_df = X_df.to_backend("cudf")
        y_df = y_df.to_backend("cudf")
    return X_df, y_df
