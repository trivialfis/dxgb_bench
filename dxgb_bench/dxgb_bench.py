# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import json
import os
from time import time
from typing import Any

import numpy as np
import xgboost as xgb
from scipy import sparse
from xgboost import QuantileDMatrix

from .dataiter import (
    TEST_SIZE,
    load_all,
    train_test_split,
)
from .datasets.generated import make_dense_regression, make_sparse_regression, psize
from .external_mem import make_iter
from .strip import make_strips
from .utils import (
    DFT_OUT,
    EvalsLog,
    Opts,
    Timer,
    __version__,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_rmm_param,
    device_attributes,
    fill_opts_shape,
    fprint,
    machine_info,
    make_params_from_args,
    merge_opts,
    mkdirs,
    peak_rmm_memory_bytes,
    save_booster,
    save_results,
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
    if fmt == "auto":
        fmt = "kio"

    if target_type != "reg":
        raise NotImplementedError()

    mkdirs(outdirs)

    with Timer("datagen", "gen"):
        size = 0

        X_fd, y_fd = make_strips(["X", "y"], outdirs, fmt=fmt, device=device)

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
                size_str = psize(X)
                fprint(
                    f"Batch:{i}, estimated size: {size_str}. {i * 100 / n_batches:.2f}%",
                    end="\r",
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
    model_path: str | None,
    params: dict[str, Any],
    opts: Opts,
    n_rounds: int,
) -> None:
    if not opts.on_the_fly:
        for d in loadfrom:
            assert os.path.exists(d), d

    with Timer("Train", "Total"):
        if task == "qdm":
            X, y = load_all(loadfrom, opts.device)
            if opts.validation:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=TEST_SIZE, random_state=2024
                )
                with Timer("Train", "DMatrix-Train"):
                    Xy = QuantileDMatrix(X_train, y_train, max_bin=params["max_bin"])
                with Timer("Train", "DMatrix-Valid"):
                    Xy_valid = QuantileDMatrix(X_test, y_test, ref=Xy)
                watches = [(Xy, "Train"), (Xy_valid, "Valid")]
            else:
                with Timer("Train", "DMatrix-Train"):
                    Xy = QuantileDMatrix(X, y, max_bin=params["max_bin"])
                    Xy_valid = None
                watches = [(Xy, "Train")]

            evals_result: EvalsLog = {}
            with Timer("Train", "Train"):
                booster = xgb.train(
                    params,
                    Xy,
                    num_boost_round=n_rounds,
                    evals=watches,
                    verbose_eval=True,
                    evals_result=evals_result,
                )

            opts = fill_opts_shape(opts, Xy, Xy_valid, 1)
        else:
            assert task == "qdm-iter"

            it_train, it_valid = make_iter(opts, loadfrom, is_ext=False)
            with Timer("Train", "DMatrix-Train"):
                Xy_train = QuantileDMatrix(it_train, max_bin=params["max_bin"], max_quantile_batches=2)
                watches = [(Xy_train, "Train")]

            if opts.validation:
                with Timer("Train", "DMatrix-Valid"):
                    Xy_valid = QuantileDMatrix(it_valid, ref=Xy_train)
                    watches.append((Xy_valid, "Valid"))
            else:
                Xy_valid = None

            evals_result = {}
            with Timer("Train", "Train"):
                booster = xgb.train(
                    params,
                    Xy_train,
                    num_boost_round=n_rounds,
                    evals=watches,
                    verbose_eval=True,
                    evals_result=evals_result,
                )

            if len(watches) >= 2:
                assert watches[1][1] == "Valid"
                opts = fill_opts_shape(
                    opts, Xy_train, watches[1][0], it_train.n_batches
                )
            else:
                opts = fill_opts_shape(opts, Xy_train, None, it_train.n_batches)

        print(f"Trained for {n_rounds} iterations.")
        print(Timer.global_timer())
        assert booster.num_boosted_rounds() == n_rounds
        machine = machine_info(opts.device)
        opts_dict = merge_opts(opts, params)
        opts_dict["n_rounds"] = n_rounds
        opts_dict["n_workers"] = 1
        results = {
            "opts": opts_dict,
            "timer": Timer.global_timer(),
            "evals": evals_result,
            "machine": machine,
        }
        save_results(results, "incore")

        if model_path is not None:
            save_booster(booster, model_path)


# https://github.com/dmlc/xgboost/pull/11058
def quick_inference(model_path: str) -> None:
    def train_model(model_path: str) -> None:
        from sklearn.datasets import load_digits
        from sklearn.model_selection import train_test_split

        data = load_digits()
        X_train, X_test, y_train, y_test = train_test_split(
            data["data"], data["target"], test_size=0.2
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        params = {
            "max_depth": 3,
            "eta": 1,
            "objective": "multi:softprob",
            "num_class": 10,
        }
        bst = xgb.train(params, dtrain, 128, [(dtrain, "train")])
        bst.save_model(model_path)

    def predict_np_array(model_path: str) -> None:
        bst = xgb.Booster()
        bst.set_param({"nthread": 1})
        bst.load_model(fname=model_path)
        times = []
        np.random.seed(7)
        iterations = 1000
        for _ in range(iterations):
            sample = np.random.uniform(-1, 10, size=(1, 64))
            start = time()
            bst.inplace_predict(sample)
            times.append(time() - start)
        iter_time = sum(times[iterations // 2 :]) / iterations / 2
        print("np.array iter_time: ", iter_time * 1000, "ms")

    def predict_sklearn(model_path: str) -> None:
        import pandas as pd

        clf = xgb.XGBClassifier()
        clf.set_params(n_jobs=1)
        clf.load_model(fname=model_path)
        times = []
        np.random.seed(7)
        iterations = 1000
        attrs = {f"{i}" for i in range(64)}
        for _ in range(iterations):
            sample = pd.DataFrame({ind: [np.random.uniform(-1, 10)] for ind in attrs})

            start = time()
            clf.predict_proba(sample)
            times.append(time() - start)

        iter_time = sum(times[iterations // 2 :]) / iterations / 2
        print("DataFrame iter_time: ", iter_time * 1000, "ms")

    train_model(model_path)
    predict_sklearn(model_path)
    predict_np_array(model_path)


def bench_inference(
    task: str,
    loadfrom: list[str],
    model_path: str,
    n_repeats: int,
    device: str,
) -> None:
    with Timer("Inference", "Total"):
        booster = xgb.Booster(model_file=model_path)
        booster.set_param({"device": device})

        X, y = load_all(loadfrom, device)

        with Timer("Inference", "Inference"):
            for i in range(n_repeats):
                booster.inplace_predict(X)

    results = {"timer": Timer.global_timer()}
    save_results(results, "infer")


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", action="store_true")

    subsparsers = parser.add_subparsers(dest="command")
    dg_parser = subsparsers.add_parser("datagen")
    if_parser = subsparsers.add_parser("infer")
    bh_parser = subsparsers.add_parser("bench")
    mi_parser = subsparsers.add_parser("mi", description="Print machine information.")
    di_parser = subsparsers.add_parser("di", description="Print device attributes.")
    rmm_peak_parser = subsparsers.add_parser(
        "rmmpeak", description="Get the peak memory usage from a RMM log."
    )

    # machine info parser
    mi_parser = add_device_param(mi_parser)

    # rmm peak parser
    rmm_peak_parser.add_argument("--path", type=str, required=True)

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
    bh_parser = add_data_params(bh_parser, False)
    bh_parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    bh_parser.add_argument(
        "--loadfrom",
        type=str,
        required=False,
        default=DFT_OUT,
        help="Load data from a directory instead of generating it.",
    )
    bh_parser.add_argument(
        "--task",
        choices=["qdm", "qdm-iter", "machine"],
        help=(
            "qdm is to use the `QuantileDMatrix` with a single blob of data, "
            + "whereas the `qdm-iter` uses the `QuantileDMatrix` with an iterator."
        ),
        required=True,
    )
    parser = add_rmm_param(parser)
    bh_parser = add_hyper_param(bh_parser)
    bh_parser.add_argument(
        "--valid", action="store_true", help="Split for the validation dataset."
    )
    bh_parser.add_argument(
        "--model_path",
        type=str,
        help="Save the booster object if a path is provided.",
        default=None,
    )

    # Inference parser
    if_parser.add_argument("--model_path", type=str, required=True)
    if_parser.add_argument("--n_repeats", type=int, default=10)
    if_parser = add_device_param(if_parser)
    if_parser.add_argument(
        "--loadfrom",
        type=str,
        required=False,
        default=DFT_OUT,
        help="Load data from a directory instead of generating it.",
    )
    if_parser.add_argument(
        "--task",
        choices=["inplace_np", "quick"],
        default="inplace_np",
    )

    args = parser.parse_args()
    if args.version is True:
        fprint(__version__)
        return

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
    elif args.command == "mi":
        mi = machine_info(device=args.device)
        print(json.dumps(mi, indent=2))
    elif args.command == "di":
        device_attributes()
    elif args.command == "rmmpeak":
        path = os.path.expanduser(args.path)
        assert os.path.exists(path)
        peak = peak_rmm_memory_bytes(path)
        print("Peak memory usage:", peak)
    elif args.command == "infer":
        if args.task == "quick":
            quick_inference(args.model_path)
            return
        loadfrom = split_path(args.loadfrom)
        bench_inference(
            args.task, loadfrom, args.model_path, args.n_repeats, args.device
        )
    else:
        assert args.command == "bench"
        loadfrom = split_path(args.loadfrom)
        params = make_params_from_args(args)

        n_batches = args.n_batches
        if args.fly:
            n = args.n_samples_per_batch * n_batches
        else:
            n = 0

        n_features = args.n_features
        opts = Opts(
            n_samples_per_batch=n // n_batches,
            n_features=n_features,
            n_batches=n_batches,
            sparsity=args.sparsity,
            on_the_fly=args.fly,
            validation=args.valid,
            device=args.device,
            mr=args.mr,
            target_type=args.target_type,
            cache_host_ratio=args.cache_host_ratio,
        )

        bench(
            args.task,
            loadfrom,
            args.model_path,
            params,
            opts,
            args.n_rounds,
        )


if __name__ == "__main__":
    cli_main()
