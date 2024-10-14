# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import os

import numpy as np
import xgboost as xgb
from scipy import sparse
from xgboost import DataIter, QuantileDMatrix

from .datasets.generated import make_dense_regression, make_sparse_regression
from .utils import Timer
from .dataiter import load_batches

def datagen(
    n_samples: int,
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
        for i in range(n_batches):
            if not assparse:  # default
                X, y = make_dense_regression(
                    device=device,
                    n_samples=n_samples,
                    n_features=n_features,
                    sparsity=sparsity,
                    random_state=i,
                )
                np.save(os.path.join(out, f"X-{i}.npy"), X)
                np.save(os.path.join(out, f"y-{i}.npy"), y)
            else:
                assert n_batches == 1, "not implemented"
                X, y = make_sparse_regression(
                    n_samples=n_samples, n_features=n_features, sparsity=sparsity
                )
                sparse.save_npz(os.path.join(out, f"X-{i}.npz"), X)
                np.save(os.path.join(out, f"y-{i}.npy"), y)

    print(Timer.global_timer())


def bench(task: str, loadfrom: str, n_rounds: int, device: str) -> None:
    assert os.path.exists(loadfrom)

    X_files: list[str] = []
    y_files: list[str] = []
    for root, subdirs, files in os.walk(loadfrom):
        for f in files:
            path = os.path.join(root, f)
            if f.startswith("X-"):
                X_files.append(path)
            else:
                y_files.append(path)
    paths = list(zip(X_files, y_files))
    assert paths

    if task == "qdm":
        with Timer("Qdm", "train"):
            X, y = load_batches(loadfrom, device)
            Xy = QuantileDMatrix(X, y)
            booster = xgb.train(
                {"tree_method": "hist", "device": device}, Xy, num_boost_round=n_rounds
            )
    else:
        assert task == "qdm-iter"

    assert booster.num_boosted_rounds() == n_rounds


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    subsparsers = parser.add_subparsers(dest="command")
    dg_parser = subsparsers.add_parser("datagen")
    bh_parser = subsparsers.add_parser("bench")

    dft_out = os.path.join(os.curdir, "data")

    # Datagen parser
    dg_parser.add_argument("--n_samples", type=int, required=True)
    dg_parser.add_argument("--n_features", type=int, required=True)
    dg_parser.add_argument("--assparse", action="store_true")
    dg_parser.add_argument("--sparsity", type=float, default=0.0)
    dg_parser.add_argument("--n_batches", type=int, default=1)
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

    args = parser.parse_args()

    if args.command == "datagen":
        datagen(
            args.n_samples,
            args.n_features,
            args.n_batches,
            args.assparse,
            args.sparsity,
            args.device,
            out=args.output,
        )
    else:
        assert args.command == "bench"
        bench(args.task, args.loadfrom, args.n_rounds, args.device)


if __name__ == "__main__":
    cli_main()
