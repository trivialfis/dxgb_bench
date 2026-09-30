# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import os
import tempfile
from itertools import product
from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from scipy import sparse
from xgboost import QuantileDMatrix
from xgboost.compat import concat

from dxgb_bench.dataiter import (
    BenchIter,
    LoadIterStrip,
    SynIterImpl,
    get_valid_sizes,
    load_all,
)
from dxgb_bench.datasets.generated import (
    datagen,
    make_dense_regression,
    make_imbalanced_regression,
    make_regression_targets,
    make_sparse_regression,
    regenerate_targets,
)
from dxgb_bench.dxgb_bench import cli_main
from dxgb_bench.strip import make_file_name, make_strips
from dxgb_bench.testing import TmpDir, assert_array_allclose, devices, formats, has_cuda


def test_sparse_regressioin() -> None:
    X, y = make_sparse_regression(
        n_samples=3, n_features=2, sparsity=0.6, random_state=0
    )
    assert isinstance(X, sparse.csr_matrix)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 3 and X.shape[1] == 2
    assert y.shape[0] == X.shape[0]
    if len(y.shape) == 2:  # couldn't squeeze vstack result for some reason
        assert y.shape[1] <= 1

    X, y = make_sparse_regression(
        n_samples=1023, n_features=32, sparsity=0.6, random_state=0
    )
    assert X.shape[0] == 1023 and X.shape[1] == 32
    assert y.shape[0] == X.shape[0]
    # 1023 * 32 * 0.6 -> 13094
    assert 13000 < X.nnz < 13110


def test_dense_regression() -> None:
    X, y = make_dense_regression(
        n_samples=3,
        n_features=2,
        n_targets=1,
        sparsity=0.6,
        device="cpu",
        random_state=1,
    )
    assert isinstance(X, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert X.shape[0] == 3 and X.shape[1] == 2
    assert y.shape[0] == X.shape[0]

    X, y = make_dense_regression(
        n_samples=2047,
        n_features=16,
        n_targets=1,
        sparsity=0.6,
        device="cpu",
        random_state=1,
    )
    assert X.shape[0] == 2047 and X.shape[1] == 16
    assert y.shape[0] == X.shape[0]
    nnz = np.count_nonzero(~np.isnan(X))
    assert 13000 < nnz < 13230
    nnz = np.count_nonzero(~np.isnan(y))
    assert nnz == 2047


def run_dense_batches(device: str, n_targets: int) -> tuple[np.ndarray, np.ndarray]:
    """Compare results between multiple batches and single batch."""
    n_features = 3
    n_batches = 12
    nspb = 16

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(
            nspb,
            n_features,
            n_targets=n_targets,
            n_batches=n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=[path],
            fmt="npy",
        )
        X0, y0 = load_all([path], "cpu")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data")
        datagen(
            nspb * n_batches,
            n_features,
            n_targets=n_targets,
            n_batches=1,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=[path],
            fmt="npy",
        )
        X1, y1 = load_all([path], "cpu")

    np.testing.assert_allclose(X0, X1)
    np.testing.assert_allclose(y0, y1)
    return X0, y0


@pytest.mark.skipif(reason="No CUDA.", condition=not has_cuda())
@pytest.mark.parametrize("n_targets", [1, 3])
def test_dense_batches(n_targets: int) -> None:
    X0, y0 = run_dense_batches("cpu", n_targets)
    X1, y1 = run_dense_batches("cuda", n_targets)
    np.testing.assert_allclose(X0, X1, rtol=1e-6)
    np.testing.assert_allclose(y0, y1, rtol=1e-5)


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize(
    "target_type,n_binary", [("reg", None), ("bin", None), ("reg", 2)]
)
@pytest.mark.parametrize("seed", [None, 19])
def test_dense_iter(
    tmp_path: Path,
    device: str,
    target_type: str,
    n_binary: int | None,
    seed: int | None,
) -> None:
    nspb, n_batches = 7, 3
    args = dict(
        n_features=4,
        n_targets=3,
        sparsity=0.0,
        assparse=False,
        target_type=target_type,
        device=device,
        n_binary=n_binary,
    )
    impl = SynIterImpl(nspb, n_batches=n_batches, rs=seed, **args)
    batches = [impl.get(i) for i in range(n_batches)]
    X = concat([batch[0] for batch in batches])
    y = concat([batch[1] for batch in batches])
    for i in reversed(range(n_batches)):
        X_i, y_i = impl.get(i)
        assert_array_allclose(X_i, batches[i][0])
        assert_array_allclose(y_i, batches[i][1])

    single = SynIterImpl(nspb * n_batches, n_batches=1, rs=seed, **args)
    X_single, y_single = single.get(0)
    assert_array_allclose(X, X_single)
    assert_array_allclose(y, y_single)

    outdirs = [str(tmp_path / "data")]
    datagen(
        nspb, n_batches=n_batches, outdirs=outdirs, fmt="npy", random_state=seed, **args
    )
    stored_X, stored_y = load_all(outdirs, "cpu")
    assert_array_allclose(X, stored_X)
    assert_array_allclose(y, stored_y)
    if target_type == "bin":
        assert set(np.unique(stored_y)) == {0.0, 1.0}


@pytest.mark.parametrize("device,n_targets", product(devices(), [1, 3]))
def test_deterministic(device: str, n_targets: int) -> None:
    n_samples_per_batch = 8192
    n_features = 400
    target_type = "reg"
    n_batches = 4

    print()
    impl = SynIterImpl(
        n_samples_per_batch,
        n_features,
        n_targets=n_targets,
        n_batches=n_batches,
        sparsity=0.0,
        assparse=False,
        target_type=target_type,
        device=device,
    )
    it = BenchIter(impl, True, False, min_cache_page_bytes=None, device=device)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []

    def append(data: np.ndarray, label: np.ndarray) -> None:
        Xs.append(data)
        ys.append(label)

    while it.next(append):
        continue
    it.reset()

    k = 0

    def check(data: np.ndarray, label: np.ndarray) -> None:
        nonlocal k

        if device == "cpu":
            np.testing.assert_allclose(data, Xs[k])
            np.testing.assert_allclose(label, ys[k])
        else:
            cp.testing.assert_allclose(data, Xs[k])
            cp.testing.assert_allclose(label, ys[k])

        k += 1

    while it.next(check):
        continue
    it.reset()


@pytest.mark.parametrize("device", devices())
def test_cv(device: str) -> None:
    with TmpDir(2, True) as outdirs:
        # Within batch read
        n_features = 2
        n_batches = 4
        nspb = 8

        datagen(
            nspb,
            n_features,
            1,
            n_batches=n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )
        n_train, n_valid = get_valid_sizes(n_samples=nspb * n_batches)
        assert n_valid == 6

        impl = LoadIterStrip(outdirs, is_valid=False, test_size=0.2, device=device)
        X, y = load_all(outdirs, device)

        prev = 0
        for i in range(impl.n_batches):
            X_i, y_i = impl.get(i)
            assert_array_allclose(X[prev : prev + nspb - 1], X_i)
            assert_array_allclose(y[prev : prev + nspb - 1], y_i)
            prev += nspb


@pytest.mark.parametrize("device", devices())
def test_datagen(device: str) -> None:
    n_shards = 2
    with TmpDir(n_shards, True) as outdirs:
        n_features = 4
        n_batches = 8
        nspb = 16

        datagen(
            nspb,
            n_features,
            1,
            n_batches=n_batches,
            assparse=False,
            target_type="reg",
            sparsity=0.0,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )

        for shard_idx, d in enumerate(outdirs):
            Xs, ys = [], []
            for b in range(n_batches):
                fname = make_file_name(
                    (nspb, n_features),
                    "X",
                    "X",
                    batch_idx=b,
                    shard_idx=shard_idx,
                    fmt="npy",
                )
                X = np.load(os.path.join(d, fname))
                assert X.shape == (nspb // n_shards, n_features)
                Xs.append(X)

                fname = make_file_name(
                    (nspb, 1),
                    "y",
                    "y",
                    batch_idx=b,
                    shard_idx=shard_idx,
                    fmt="npy",
                )
                y = np.load(os.path.join(d, fname))
                assert y.shape[0] == nspb // n_shards
                assert y.shape[0] == y.size
                ys.append(y)

            # Must be unique, not guaranteed, just unlikely to have same floating
            # values.
            for i in range(1, n_batches):
                assert not (Xs[0] == Xs[i]).any()
                assert not (ys[0] == ys[i]).any()

        X, y = load_all(outdirs, device)
        assert X.shape[0] == y.shape[0] == nspb * n_batches


@pytest.mark.parametrize("device,fmt", product(devices(), formats()))
def test_load_all(device: str, fmt: str) -> None:
    n_shards = 2
    with TmpDir(n_shards, True) as outdirs:
        X_fd, y_fd = make_strips(["X", "y"], outdirs, fmt=fmt, device=device)
        X = np.arange(0, 64, dtype=np.float32).reshape(8, 8)
        y = X.sum(axis=1)
        X_fd.write(X, 0)
        y_fd.write(y, 0)

        X_res, y_res = load_all(outdirs, device=device)
        assert_array_allclose(X, X_res.squeeze())
        assert_array_allclose(y, y_res.squeeze())


@pytest.mark.parametrize(
    "device,n_targets,n_binary", list(product(devices(), [1, 4], [0, 3, 8]))
)
def test_imbalanced_batches(device: str, n_targets: int, n_binary: int) -> None:
    kwargs = dict(
        n_features=8, n_targets=n_targets, n_binary=n_binary, random_state=2026
    )
    X, y = make_imbalanced_regression(device, 35, **kwargs)
    assert X.dtype == y.dtype == np.float32
    assert X.shape == (35, 8) and y.shape == (35, n_targets)
    cpu_X, cpu_y = make_imbalanced_regression("cpu", 35, **kwargs)
    assert_array_allclose(X, cpu_X, rtol=1e-5)
    np.testing.assert_allclose(
        cp.asnumpy(y) if device == "cuda" else y, cpu_y, rtol=1e-5, atol=1e-6
    )
    assert np.isin(cpu_X[:, :n_binary], [0.0, 1.0]).all()
    if n_binary < 8:
        assert np.unique(cpu_X[:, n_binary:]).size > 2
    # Access batches out of order, including a repeated batch and odd row counts.
    for begin, end in [(10, 35), (0, 3), (3, 10), (10, 35)]:
        X_i, y_i = make_imbalanced_regression(
            device, end - begin, row_offset=begin, **kwargs
        )
        assert_array_allclose(X_i, X[begin:end])
        assert_array_allclose(y_i, y[begin:end])
    other_X, _ = make_imbalanced_regression(
        device, 35, 8, n_targets + 1, n_binary=n_binary, random_state=2026
    )
    assert_array_allclose(X, other_X)


def test_imbalanced_model() -> None:
    X, y = make_imbalanced_regression("cpu", 2048, 8, 3, n_binary=5)
    cuts, _ = QuantileDMatrix(X, y, max_bin=32).get_quantile_cut()
    n_bins = np.diff(cuts)
    assert (n_bins[:5] <= 3).all()
    assert (n_bins[5:] > n_bins[:5].max()).all()
    # A model fitted to one batch must also explain the other batches with unit noise.
    coef = np.linalg.lstsq(X[:1024], y[:1024], rcond=None)[0]
    residual = y[1024:] - X[1024:] @ coef
    assert 0.8 < np.std(residual) < 1.2
    other_X, other_y = make_imbalanced_regression(
        "cpu", 2048, 8, 3, n_binary=5, random_state=2027
    )
    assert not np.array_equal(X, other_X)
    assert not np.array_equal(y, other_y)


@pytest.mark.parametrize("device,fmt", list(product(devices(), formats())))
def test_imbalanced_datagen(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, device: str, fmt: str
) -> None:
    if fmt == "kio":
        pytest.importorskip("kvikio")
    monkeypatch.chdir(tmp_path)
    cli_main(
        [
            "datagen",
            "--n_samples_per_batch=7",
            "--n_batches=3",
            "--n_features=8",
            "--n_binary=5",
            "--n_targets=4",
            "--data_seed=19",
            f"--device={device}",
            f"--fmt={fmt}",
            "--saveto=source-b,source-a",
        ]
    )
    X, y = load_all(["source-b", "source-a"], "cpu")
    expected_X, expected_y = make_imbalanced_regression(
        "cpu", 21, 8, 4, n_binary=5, random_state=19
    )
    np.testing.assert_allclose(X, expected_X, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(y, expected_y, rtol=1e-5, atol=1e-6)
    cli_main(
        [
            "datagen",
            "--loadfrom=source-b,source-a",
            "--saveto=targets-b,targets-a",
            "--n_targets=3",
            "--data_seed=29",
            f"--device={device}",
        ]
    )
    for suffix in ("a", "b"):
        shared = Path(f"targets-{suffix}/X")
        assert shared.is_symlink()
        assert shared.resolve() == Path(f"source-{suffix}/X").resolve()
    shared_X, new_y = load_all(["targets-b", "targets-a"], "cpu")
    np.testing.assert_array_equal(X, shared_X)
    expected_y = make_regression_targets(X, 3, random_state=29)
    np.testing.assert_allclose(new_y, expected_y, rtol=1e-5, atol=1e-6)
    _, original_y = load_all(["source-b", "source-a"], "cpu")
    np.testing.assert_array_equal(y, original_y)
    # Reject existing labels and mismatched feature links before modifying either.
    with pytest.raises(ValueError, match="already contains"):
        regenerate_targets(["source-b", "source-a"], ["targets-b", "targets-a"], 2)
    _, intact_y = load_all(["targets-b", "targets-a"], "cpu")
    np.testing.assert_array_equal(new_y, intact_y)


@pytest.mark.parametrize(
    "args,message",
    [
        ([], "are required"),
        (["--n_binary=-1"], "n_binary must be"),
        (["--n_binary=9"], "n_binary must be"),
        (["--n_binary=3", "--n_targets=0"], "must be positive"),
        (["--n_binary=3", "--data_seed=-1"], "must be nonnegative"),
        (["--n_binary=3", "--sparsity=0.1"], "zero sparsity"),
        (["--n_binary=3", "--assparse"], "dense regression"),
        (["--n_binary=3", "--target_type=bin"], "dense regression"),
        (["--n_binary=3", "--fmt=npz"], "npy or kio"),
        (["--loadfrom=source"], "infers feature shapes"),
    ],
)
def test_imbalanced_invalid_args(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str], message: str
) -> None:
    shape = ["--n_samples_per_batch=7", "--n_features=8"] if args else []
    with pytest.raises(SystemExit) as exc:
        cli_main(
            ["datagen", "--device=cpu", f"--saveto={tmp_path / 'out'}", *shape, *args]
        )
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
