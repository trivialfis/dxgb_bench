# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import os

import numpy as np
import xgboost as xgb
from scipy import sparse
from xgboost import DataIter, QuantileDMatrix

from .dataiter import (
    TEST_SIZE,
    BenchIter,
    LoadIterImpl,
    get_file_paths,
    load_all,
    load_batches,
    train_test_split,
)
from .datasets.generated import make_dense_regression, make_sparse_regression
from .utils import Timer, add_data_params


def datagen(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    assparse: bool,
    sparsity: float,
    device: str,
    out: str,
) -> None:
    if not os.path.exists(out):
        os.mkdir(out)

    with Timer("datagen", "gen"):
        size = 0
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
                np.save(os.path.join(out, f"X-{i}.npy"), X)
                np.save(os.path.join(out, f"y-{i}.npy"), y)
            else:
                X, y = make_sparse_regression(
                    n_samples=n_samples_per_batch,
                    n_features=n_features,
                    sparsity=sparsity,
                    random_state=size,
                )
                sparse.save_npz(os.path.join(out, f"X-{i}.npz"), X)
                np.save(os.path.join(out, f"y-{i}.npy"), y)
            size += X.size

    print(Timer.global_timer())


def bench(task: str, loadfrom: str, n_rounds: int, valid: bool, device: str) -> None:
    assert os.path.exists(loadfrom)

    if task == "qdm":
        X, y = load_all(loadfrom, device)
        if valid:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=2024
            )
            with Timer("Qdm", "Train"):
                Xy = QuantileDMatrix(X_train, y_train)
            with Timer("Qdm", "Valid"):
                Xy_valid = QuantileDMatrix(X_test, y_test, ref=Xy)
            watches = [(Xy, "Train"), (Xy_valid, "Valid")]
        else:
            with Timer("Qdm", "Train"):
                Xy = QuantileDMatrix(X, y)
                Xy_valid = None
            watches = [(Xy, "Train")]
    else:
        assert task == "qdm-iter"
        X_files, y_files = get_file_paths(loadfrom)
        paths = list(zip(X_files, y_files))
        it_impl = LoadIterImpl(paths, device=device)
        it_train = BenchIter(it_impl, split=valid, is_ext=False, is_eval=False)
        with Timer("Qdm", "Train"):
            Xy = QuantileDMatrix(it_train)
            watches = [(Xy, "Train")]

        if valid:
            it_valid = BenchIter(it_impl, split=valid, is_ext=False, is_eval=True)
            with Timer("Qdm", "valid"):
                Xy_valid = QuantileDMatrix(it_valid, ref=Xy)
                watches.append((Xy_valid, "Valid"))

    with Timer("Qdm", "train"):
        booster = xgb.train(
            {"tree_method": "hist", "device": device, "max_depth": 6},
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

    dft_out = os.path.join(os.curdir, "data")

    # Datagen parser
    dg_parser = add_data_params(dg_parser, True)
    dg_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    dg_parser.add_argument("--saveto", type=str, default=dft_out)

    # Benchmark parser
    bh_parser.add_argument(
        "--device", choices=["cpu", "cuda"], default="cpu", required=False
    )
    bh_parser.add_argument(
        "--loadfrom",
        type=str,
        required=False,
        default=dft_out,
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
    bh_parser.add_argument("--n_rounds", type=int, default=128)
    bh_parser.add_argument(
        "--valid", action="store_true", help="Split for the validation dataset."
    )

    args = parser.parse_args()

    if args.command == "datagen":
        datagen(
            n_samples_per_batch=args.n_samples_per_batch,
            n_features=args.n_features,
            n_batches=args.n_batches,
            assparse=args.assparse,
            sparsity=args.sparsity,
            device=args.device,
            out=args.saveto,
        )
    else:
        assert args.command == "bench"
        bench(args.task, args.loadfrom, args.n_rounds, args.valid, args.device)


if __name__ == "__main__":
    cli_main()
