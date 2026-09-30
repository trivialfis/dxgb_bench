# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import ctypes
import functools
import os
import platform
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType

import numpy as np
from scipy import sparse

from ..strip import make_strips
from ..utils import Timer, div_roundup, fprint

DEFAULT_DATA_SEED = 2026
DEFAULT_TARGET_SEED = 2027


def _array_module(device: str) -> ModuleType:
    if device == "cpu":
        return np
    if device == "cuda":
        import cupy

        return cupy
    raise ValueError("device must be cpu or cuda.")


def _array_ptr(array: np.ndarray) -> int:
    if hasattr(array, "__cuda_array_interface__"):
        return array.__cuda_array_interface__["data"][0]
    return array.__array_interface__["data"][0]


@functools.cache
def _load_lib() -> ctypes.CDLL:
    if platform.system() == "Windows":
        name = "dxgbbench.dll"
    else:
        name = "libdxgbbench.so"
    path = os.path.join(
        os.path.normpath(os.path.abspath(os.path.dirname(__file__))),
        os.pardir,
        name,
    )
    lib = ctypes.cdll.LoadLibrary(path)
    lib.MakeDenseRegression.argtypes = [
        ctypes.c_bool,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_double,
        ctypes.c_int64,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.MakeDenseRegression.restype = ctypes.c_int
    lib.MakeImbalancedFeatures.argtypes = [
        ctypes.c_bool,
        *([ctypes.c_int64] * 5),
        ctypes.c_void_p,
    ]
    lib.MakeImbalancedFeatures.restype = ctypes.c_int
    lib.MakeRegressionTargets.argtypes = [
        ctypes.c_bool,
        *([ctypes.c_int64] * 5),
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    lib.MakeRegressionTargets.restype = ctypes.c_int
    return lib


def make_dense_regression(
    device: str,
    n_samples: int,
    n_features: int,
    n_targets: int,
    *,
    sparsity: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate dense regression arrays, using NaN for missing feature values.

    The native generator interprets random_state as an offset in the feature stream.
    """
    xp = _array_module(device)
    X = xp.empty((n_samples, n_features), dtype=np.float32)
    y = xp.empty((n_samples, n_targets), dtype=np.float32)
    if device == "cuda":
        xp.cuda.get_current_stream().synchronize()
    status = _load_lib().MakeDenseRegression(
        device == "cuda",
        n_samples,
        n_features,
        n_targets,
        sparsity,
        random_state,
        _array_ptr(X),
        _array_ptr(y),
    )
    if status != 0:
        raise ValueError("Native implementation failed.")
    return X, y


def make_sparse_regression(
    n_samples: int, n_features: int, *, sparsity: float, random_state: int
) -> tuple[sparse.csr_matrix, np.ndarray]:
    """Make sparse synthetic data for regression. Result is stored in CSR even if the
    data is dense.

    """

    n_threads_maybe_none = os.cpu_count()
    assert n_threads_maybe_none is not None
    n_threads = min(n_threads_maybe_none, n_samples)
    n_samples_per_batch = div_roundup(n_samples, n_threads)

    def random_csr(t_id: int, seed: int) -> tuple[sparse.csr_matrix, np.ndarray]:
        rng = np.random.default_rng(seed)
        if t_id == n_threads - 1:
            nspb = n_samples - (n_samples_per_batch * (n_threads - 1))
        else:
            nspb = n_samples_per_batch

        csr = sparse.random(
            m=nspb,
            n=n_features,
            density=1.0 - sparsity,
            random_state=rng,
        )
        y = csr.sum(axis=1)
        y += rng.normal(loc=0, scale=0.5, size=y.shape)
        return csr, y

    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            seed = random_state + n_samples_per_batch * (i + 1)
            futures.append(executor.submit(random_csr, i, seed))

        X_results = []
        y_results = []
        for f in futures:
            X, y = f.result()
            X_results.append(X)
            y_results.append(y)

        assert len(y_results) == n_threads

        X = sparse.vstack(X_results, format="csr")
        y = np.vstack(y_results)

    assert X.shape[0] == n_samples, X.shape
    assert y.shape[0] == n_samples, y.shape
    return X, y


def psize(X: np.ndarray) -> str:
    """Print the size into a string."""
    n_bytes = X.itemsize * X.size
    if n_bytes < 1024:
        size = f"{n_bytes} B"
    elif n_bytes < 1024**2:
        size = f"{n_bytes / 1024} KB"
    elif n_bytes < 1024**3:
        size = f"{n_bytes / 1024**2} MB"
    else:
        size = f"{n_bytes / 1024**3} GB"
    return size


def make_dense_binary_classification(
    device: str, n_samples: int, n_features: int, n_targets: int, random_state: int
) -> tuple[np.ndarray, np.ndarray]:
    X, y_sum = make_dense_regression(
        device,
        n_samples,
        n_features,
        n_targets,
        sparsity=0.0,
        random_state=random_state,
    )
    return X, (y_sum > 0).astype(np.float32)


def make_regression_targets(
    X: np.ndarray,
    n_targets: int,
    *,
    random_state: int = DEFAULT_TARGET_SEED,
    row_offset: int = 0,
) -> np.ndarray:
    """Apply one fixed linear model with unit normal noise across all batches.

    ``row_offset`` is the number of rows preceding X in the full dataset. Coefficients
    depend only on the feature/target counts and seed, never on the batch shape.
    """
    if X.ndim != 2 or min(X.shape) <= 0 or n_targets <= 0:
        raise ValueError(
            "X must be nonempty and two-dimensional; n_targets must be positive."
        )
    if X.dtype != np.float32:
        raise TypeError("X must have dtype float32.")
    if random_state < 0 or row_offset < 0:
        raise ValueError("random_state and row_offset must be nonnegative.")
    is_cuda = hasattr(X, "__cuda_array_interface__")
    xp = _array_module("cuda" if is_cuda else "cpu")
    X = xp.ascontiguousarray(X)
    coef = np.random.default_rng(random_state).normal(size=(X.shape[1], n_targets))
    coef = xp.asarray(coef, dtype=np.float32)
    y = xp.empty((X.shape[0], n_targets), dtype=np.float32)
    if is_cuda:
        xp.cuda.get_current_stream().synchronize()
    status = _load_lib().MakeRegressionTargets(
        is_cuda,
        *X.shape,
        n_targets,
        random_state,
        row_offset,
        _array_ptr(X),
        _array_ptr(coef),
        _array_ptr(y),
    )
    if status != 0:
        raise ValueError("Native target generation failed.")
    return y


def make_imbalanced_regression(
    device: str,
    n_samples: int,
    n_features: int,
    n_targets: int,
    *,
    n_binary: int,
    random_state: int = DEFAULT_DATA_SEED,
    row_offset: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate binary columns followed by standard normal columns and linear targets.

    The imbalance is in feature bin counts, not target classes. Use the same seed
    and cumulative row offsets to reproduce the same dataset with any batch size.
    """
    if min(n_samples, n_features, n_targets) <= 0:
        raise ValueError("n_samples, n_features, and n_targets must be positive.")
    if not 0 <= n_binary <= n_features:
        raise ValueError("n_binary must be between zero and n_features.")
    if random_state < 0 or row_offset < 0:
        raise ValueError("random_state and row_offset must be nonnegative.")
    xp = _array_module(device)
    X = xp.empty((n_samples, n_features), dtype=np.float32)
    if device == "cuda":
        xp.cuda.get_current_stream().synchronize()
    status = _load_lib().MakeImbalancedFeatures(
        device == "cuda",
        n_samples,
        n_features,
        n_binary,
        random_state,
        row_offset,
        _array_ptr(X),
    )
    if status != 0:
        raise ValueError("Native feature generation failed.")
    y = make_regression_targets(
        X, n_targets, random_state=random_state, row_offset=row_offset
    )
    return X, y


def _validate_generation(
    n_samples: int,
    n_features: int,
    n_targets: int,
    *,
    assparse: bool,
    target_type: str,
    sparsity: float,
    device: str,
    n_binary: int | None,
    random_state: int | None,
) -> None:
    if min(n_samples, n_features, n_targets) <= 0:
        raise ValueError("Sample, feature, and target counts must be positive.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if target_type not in ("reg", "bin"):
        raise ValueError("target_type must be reg or bin.")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError("sparsity must be between zero and one.")
    if random_state is not None and random_state < 0:
        raise ValueError("random_state must be nonnegative.")
    if n_binary is not None:
        if assparse or sparsity != 0.0 or target_type != "reg":
            raise ValueError(
                "n_binary requires dense regression data with zero sparsity."
            )
        if not 0 <= n_binary <= n_features:
            raise ValueError("n_binary must be between zero and n_features.")
    if assparse and (target_type != "reg" or n_targets != 1):
        raise ValueError("Sparse generation supports single-target regression only.")


def make_batch(
    n_samples: int,
    n_features: int,
    n_targets: int,
    *,
    assparse: bool,
    target_type: str,
    sparsity: float,
    device: str,
    n_binary: int | None = None,
    random_state: int | None = None,
    row_offset: int = 0,
) -> tuple[np.ndarray | sparse.csr_matrix, np.ndarray]:
    """Generate a batch for stored datasets or an on-demand iterator.

    Dense batches use global row offsets, independent of batch size and access order.
    The original dense generator starts at feature offset zero; mixed features use
    DEFAULT_DATA_SEED unless an explicit random_state is supplied.
    """
    _validate_generation(
        n_samples,
        n_features,
        n_targets,
        assparse=assparse,
        target_type=target_type,
        sparsity=sparsity,
        device=device,
        n_binary=n_binary,
        random_state=random_state,
    )
    if row_offset < 0:
        raise ValueError("row_offset must be nonnegative.")
    if n_binary is not None:
        return make_imbalanced_regression(
            device,
            n_samples,
            n_features,
            n_targets,
            n_binary=n_binary,
            random_state=DEFAULT_DATA_SEED if random_state is None else random_state,
            row_offset=row_offset,
        )
    seed = (0 if random_state is None else random_state) + row_offset * n_features
    if assparse:
        return make_sparse_regression(
            n_samples, n_features, sparsity=sparsity, random_state=seed
        )
    if target_type == "bin":
        return make_dense_binary_classification(
            device, n_samples, n_features, n_targets, random_state=seed
        )
    return make_dense_regression(
        device, n_samples, n_features, n_targets, sparsity=sparsity, random_state=seed
    )


def datagen(
    n_samples_per_batch: int,
    n_features: int,
    n_targets: int,
    n_batches: int,
    *,
    assparse: bool,
    target_type: str,
    sparsity: float,
    device: str,
    outdirs: list[str],
    fmt: str,
    n_binary: int | None = None,
    random_state: int | None = None,
) -> None:
    """Write synthetic batches using the same generator as the data iterator."""
    _validate_generation(
        n_samples_per_batch,
        n_features,
        n_targets,
        assparse=assparse,
        target_type=target_type,
        sparsity=sparsity,
        device=device,
        n_binary=n_binary,
        random_state=random_state,
    )
    if n_batches <= 0:
        raise ValueError("Batch count must be positive.")
    if n_binary is not None:
        _check_new_dataset(outdirs)
    if fmt == "auto":
        fmt = "npz" if assparse else "kio"
    if fmt not in (("npz",) if assparse else ("npy", "kio")):
        raise ValueError(
            "Sparse data requires npz; dense data requires npy or kio storage."
        )
    if not outdirs:
        raise ValueError("Output directories must be nonempty.")

    for d in outdirs:
        Path(d).mkdir(parents=True, exist_ok=True)

    with Timer("datagen", "gen"):
        # Retain the historical CSR seed sequence, which advances by stored nnz.
        size = 0
        if not assparse:
            X_fd, y_fd = make_strips(["X", "y"], outdirs, fmt=fmt, device=device)

        for i in range(n_batches):
            X, y = make_batch(
                n_samples_per_batch,
                n_features,
                n_targets,
                assparse=assparse,
                target_type=target_type,
                sparsity=sparsity,
                device=device,
                n_binary=n_binary,
                random_state=size if assparse else random_state,
                row_offset=0 if assparse else i * n_samples_per_batch,
            )
            if assparse:
                out = outdirs[i % len(outdirs)]
                sparse.save_npz(
                    os.path.join(out, f"X_{X.shape[0]}_{X.shape[1]}-{i}.npz"), X
                )
                np.save(os.path.join(out, f"y_{y.shape[0]}_1-{i}.npz"), y)
                size += X.size
            else:
                fprint(
                    f"Batch:{i}, estimated size: {psize(X)}. {i * 100 / n_batches:.2f}%",
                    end="\r",
                )
                X_fd.write(X, batch_idx=i)
                y_fd.write(y, batch_idx=i)

    print(Timer.global_timer())


def _check_new_dataset(outdirs: list[str]) -> None:
    if not outdirs or len({Path(d).resolve() for d in outdirs}) != len(outdirs):
        raise ValueError("Output directories must be nonempty and distinct.")
    for d in outdirs:
        for name in ("X", "y"):
            if os.path.lexists(Path(d) / name):
                raise ValueError(f"Output already contains {name}: {d}")


def regenerate_targets(
    loadfrom: list[str],
    outdirs: list[str],
    n_targets: int,
    *,
    device: str = "cpu",
    random_state: int = DEFAULT_TARGET_SEED,
) -> None:
    """Write new regression targets, sharing the source feature strips by symlink."""
    if n_targets <= 0 or random_state < 0:
        raise ValueError("n_targets must be positive and random_state nonnegative.")
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be cpu or cuda.")
    if not loadfrom or len(loadfrom) != len(outdirs):
        raise ValueError(
            "Source and output must have the same number of shard directories."
        )
    if len({Path(d).resolve() for d in loadfrom}) != len(loadfrom):
        raise ValueError("Source directories must be distinct.")
    _check_new_dataset(outdirs)
    for d in loadfrom:
        if not (Path(d) / "X").is_dir():
            raise ValueError(f"Source has no feature directory: {d}")
    (X_fd,) = make_strips(["X"], loadfrom, fmt=None, device=device)
    batches = sorted(X_fd.batch_key)
    if not batches or batches != list(range(len(batches))):
        raise ValueError("Source batches must be consecutive, starting at zero.")
    n_features = X_fd.batch_key[0].n_features
    if n_features <= 0 or any(
        p.n_features != n_features or p.n_samples <= 0 for p in X_fd.batch_key.values()
    ):
        raise ValueError(
            "Source batches must be nonempty with a consistent feature count."
        )
    if X_fd.fmt not in ("npy", "kio"):
        raise ValueError("Target generation requires dense npy or kio features.")
    for src, dst in zip(sorted(loadfrom), sorted(outdirs)):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "X").symlink_to(
            (Path(src) / "X").resolve(), target_is_directory=True
        )
    (y_fd,) = make_strips(["y"], outdirs, fmt=X_fd.fmt, device=device)
    row_offset = 0
    with Timer("datagen", "targets"):
        for i in batches:
            X = X_fd.read(i, None, None)
            y = make_regression_targets(
                X, n_targets, random_state=random_state, row_offset=row_offset
            )
            y_fd.write(y, i)
            row_offset += X.shape[0]
