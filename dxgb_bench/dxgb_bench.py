# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import xgboost as xgb
from scipy import sparse
from xgboost import QuantileDMatrix

from .dataiter import (
    TEST_SIZE,
    BenchIter,
    LoadIterImpl,
    get_file_paths,
    load_all,
    train_test_split,
)
from .datasets.generated import make_dense_regression, make_sparse_regression
from .strip import Strip
from .utils import (
    DFT_OUT,
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_target_type,
    make_params_from_args,
    mkdirs,
    split_path,
)


def datagen(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    *,
    assparse: bool,
    target_type: str,
    sparsity: float,
    device: str,
    outdirs: list[str],
    fmt: str,
) -> None:
    if assparse and fmt == "auto":
        fmt = "npz"

    if target_type != "reg":
        raise NotImplementedError()

    mkdirs(outdirs)

    with Timer("datagen", "gen"):
        size = 0

        X_fd = Strip("X", dirs=outdirs, fmt=fmt, device=device)
        y_fd = Strip("y", dirs=outdirs, fmt=fmt, device=device)

        for i in range(n_batches):
            assert n_samples_per_batch >= 1
            if not assparse:  # default
                X, y = make_dense_regression(
                    device=device,
                    n_samples=n_samples_per_batch,
                    n_features=n_features,
                    sparsity=sparsity,
                    random_state=size,
                )

                if device == "cuda":
                    import cupy as cp

                    assert isinstance(X, cp.ndarray)

                X_fd.write(X, batch_idx=i)
                y_fd.write(y, batch_idx=i)
            else:
                out = outdirs[i % len(outdirs)]
                X, y = make_sparse_regression(
                    n_samples=n_samples_per_batch,
                    n_features=n_features,
                    sparsity=sparsity,
                    random_state=size,
                )
                sparse.save_npz(
                    os.path.join(out, f"X_{X.shape[0]}_{X.shape[1]}-{i}.npz"), X
                )
                np.save(os.path.join(out, f"y_{y.shape[0]}_1-{i}.npz"), y)
            size += X.size

    print(Timer.global_timer())


def bench(
    task: str,
    loadfrom: list[str],
    params: dict[str, Any],
    n_rounds: int,
    valid: bool,
    device: str,
) -> None:
    for d in loadfrom:
        assert os.path.exists(d)

    if task == "qdm":
        X, y = load_all(loadfrom, device)
        if valid:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=2024
            )
            with Timer("Qdm", "Train-DMatrix"):
                Xy = QuantileDMatrix(X_train, y_train)
            with Timer("Qdm", "Valid-DMatrix"):
                Xy_valid = QuantileDMatrix(X_test, y_test, ref=Xy)
            watches = [(Xy, "Train"), (Xy_valid, "Valid")]
        else:
            with Timer("Qdm", "Train-DMatrix"):
                Xy = QuantileDMatrix(X, y)
                Xy_valid = None
            watches = [(Xy, "Train")]
    else:
        assert task == "qdm-iter"
        X_files, y_files = get_file_paths(loadfrom)
        paths = list(zip(X_files, y_files))
        if valid:
            it_impl = LoadIterImpl(paths, split=True, is_valid=False, device=device)
            it_train = BenchIter(it_impl, is_ext=False, is_valid=False, device=device)

            it_impl = LoadIterImpl(paths, split=True, is_valid=True, device=device)
            it_valid = BenchIter(it_impl, is_ext=False, is_valid=True, device=device)
        else:
            it_impl = LoadIterImpl(paths, split=False, is_valid=False, device=device)
            it_train = BenchIter(it_impl, is_ext=False, is_valid=False, device=device)
            it_valid = None

        with Timer("Qdm", "Train-DMatrix"):
            Xy = QuantileDMatrix(it_train)
            watches = [(Xy, "Train")]
        if valid:
            with Timer("Qdm", "Valid-DMatrix"):
                Xy_valid = QuantileDMatrix(it_valid, ref=Xy)
                watches.append((Xy_valid, "Valid"))

    with Timer("Qdm", "train"):
        booster = xgb.train(
            params,
            Xy,
            num_boost_round=n_rounds,
            evals=watches,
            verbose_eval=True,
        )

    assert booster.num_boosted_rounds() == n_rounds
    print(f"Trained for {n_rounds} iterations.")
    print(Timer.global_timer())


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    subsparsers = parser.add_subparsers(dest="command")
    dg_parser = subsparsers.add_parser("datagen")
    bh_parser = subsparsers.add_parser("bench")

    # Datagen parser
    dg_parser = add_data_params(dg_parser, True)
    dg_parser = add_device_param(dg_parser)
    dg_parser.add_argument(
        "--saveto",
        type=str,
        default=DFT_OUT,
        help="Comma separated list of output directories. Poor man's raid0.",
    )

    # Benchmark parser
    bh_parser = add_device_param(bh_parser)
    bh_parser.add_argument(
        "--loadfrom",
        type=str,
        required=False,
        default=DFT_OUT,
        help="Load data from a directory instead of generating it.",
    )
    bh_parser.add_argument(
        "--task",
        choices=["qdm", "qdm-iter"],
        help=(
            "qdm is to use the `QuantileDMatrix` with a single blob of data, "
            + "whereas the `qdm-iter` uses the `QuantileDMatrix` with an iterator."
        ),
        required=True,
    )
    bh_parser = add_hyper_param(bh_parser)
    bh_parser = add_target_type(bh_parser)
    bh_parser.add_argument(
        "--valid", action="store_true", help="Split for the validation dataset."
    )

    args = parser.parse_args()

    if args.command == "datagen":
        saveto = split_path(args.saveto)
        datagen(
            n_samples_per_batch=args.n_samples_per_batch,
            n_features=args.n_features,
            n_batches=args.n_batches,
            assparse=args.assparse,
            target_type=args.target_type,
            sparsity=args.sparsity,
            device=args.device,
            outdirs=saveto,
            fmt=args.fmt,
        )
    else:
        assert args.command == "bench"
        loadfrom = split_path(args.loadfrom)
        params = make_params_from_args(args)
        bench(args.task, loadfrom, params, args.n_rounds, args.valid, args.device)


if __name__ == "__main__":
    cli_main()
