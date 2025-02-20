"""
# datagen
{
  "sparsity": 0.0,
  "n_features": 256,
  "n_samples_per_batch": 4096,
  "n_batches": 1,
  "outdirs": ["./data-0", "./data-1"]
  "format": "kvikio",		// one of the npy, kvikio, npz,
  "assparse": true,
  "device": "cpu"
}
# bench
{
  "dmatrix": "qdm_iter",
  "device": "cpu",
  "num_boost_round": 10
}
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import Any, Callable, Generator, Type

import numpy as np
import xgboost as xgb
from scipy import sparse
from typing_extensions import override
from xgboost.compat import concat

from .datasets.generated import make_sparse_regression
from .utils import Timer


def make_sparse_regression_batches(config: dict[str, Any]) -> dict[str, Any]:
    n_batches: int = config["n_batches"]
    n_samples_per_batch: int = config["n_samples_per_batch"]
    n_features: int = config["n_features"]
    outdirs: list[str] = config["outdirs"]
    if config["format"] != "npy":
        raise ValueError("Only npz is supported for sparse output.")

    file_info = []
    n_samples = 0

    for i in range(n_batches):
        X, y = make_sparse_regression(
            n_samples=n_samples_per_batch,
            n_features=n_features,
            sparsity=config["sparsity"],
            random_state=i * n_samples_per_batch,
        )
        outdir = outdirs[i % len(outdirs)]
        outdir = os.path.abspath(os.path.expanduser(outdir))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        X_path = os.path.join(outdir, f"X-{i}.npz")
        y_path = os.path.join(outdir, f"y-{i}.npy")
        sparse.save_npz(X_path, X)
        np.save(y_path, y)

        meta = {"shape": X.shape, "X": X_path, "y": y_path, "batch_idx": i}
        file_info.append(meta)
        n_samples += n_samples_per_batch

    meta = {"shape": (n_samples, n_features), "files": file_info, "format": "csr"}
    return meta


def load_sparse_it(
    meta: dict[str, Any]
) -> Generator[tuple[sparse.csr_matrix, np.ndarray]]:
    n_batches = len(meta["files"])
    files = meta["files"]

    for i in range(n_batches):
        X_path = files[i]["X"]
        y_path = files[i]["y"]
        X = sparse.load_npz(X_path)
        y = np.load(y_path)

        yield X, y


class BenchIter(xgb.DataIter):
    def __init__(self, meta: dict[str, Any]) -> None:
        self.meta = meta
        self.gen: Generator[tuple[sparse.csr_matrix, np.ndarray]] | None = None
        super().__init__()

    @override
    def next(self, input_data: Callable) -> bool:
        if self.gen is None:
            self.gen = load_sparse_it(self.meta)
        try:
            X, y = next(self.gen)
            input_data(data=X, label=y)
            return True
        except StopIteration:
            return False

    @override
    def reset(self) -> None:
        self.gen = None


def loaddata(use_batches: bool, dm_type: Callable) -> xgb.DMatrix:
    with open("data.json", "r") as fd:
        meta = json.load(fd)

    if meta["format"] == "csr":
        # load sparse data
        n_batches = len(meta["files"])
        if use_batches:
            it = BenchIter(meta)
            return dm_type(it)
        else:
            Xs = []
            ys = []
            for X, y in load_sparse_it(meta):
                Xs.append(X)
                ys.append(y)

            X = concat(Xs)
            y = concat(ys)
            return dm_type(X, y)
    else:
        pass


def datagen(args: argparse.Namespace) -> None:
    with open(args.config, "r") as fd:
        config = json.load(fd)

    if config["assparse"]:
        with Timer("datagen", "sparse"):
            outdirs = config["outdirs"]
            for outdir in outdirs:
                outdir = os.path.abspath(os.path.expanduser(outdir))
                if os.path.exists(outdir):
                    shutil.rmtree(outdir)

            meta = make_sparse_regression_batches(config)

        with open("data.json", "w") as fd:
            json.dump(meta, fd)
    else:
        pass


def runbench(config: dict[str, Any]) -> None:
    dmatrix = config["dmatrix"]
    with Timer("bench", "DMatrix"):
        Xy = loaddata(dmatrix in ["extmem", "qdm_iter"], map_dm(dmatrix))
    with Timer("bench", "Train"):
        xgb.train(
            {"device": config["device"]},
            num_boost_round=config["num_boost_round"],
            dtrain=Xy,
            evals=[(Xy, "Train")],
        )

def map_dm(name: str) -> Type:
    match name:
        case "qdm":
            return xgb.QuantileDMatrix
        case "qdm_iter":
            return xgb.QuantileDMatrix
        case "extmem":
            return xgb.ExtMemQuantileDMatrix
        case "dm":
            return xgb.DMatrix
        case _:
            raise ValueError("Invalid DMatrix type.")


def cli_main() -> None:
    parser = argparse.ArgumentParser()

    subsparsers = parser.add_subparsers(dest="command")
    dg_parser = subsparsers.add_parser("datagen")
    dg_parser.add_argument("--config", type=str, required=True)

    bh_parser = subsparsers.add_parser("bench")
    bh_parser.add_argument("--config", type=str, required=True)

    args = parser.parse_args()

    assert args.command in ("datagen", "bench")

    if args.command == "datagen":
        datagen(args)
    else:
        with open(args.config, "r") as fd:
            config = json.load(fd)
        runbench(config)
