# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

from dask import array as da


def make_dense_regression(
    device: str, n_samples: int, n_features: int, random_state: int
) -> tuple[da.Array, da.Array]:
    rng = da.random.default_rng(seed=random_state)
    X = rng.normal(size=(n_samples, n_features))
    if device.startswith("cuda"):
        X = X.to_backend("cupy")
    y = X.sum(axis=1)
    return X, y
