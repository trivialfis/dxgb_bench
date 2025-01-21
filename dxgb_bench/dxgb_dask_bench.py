#!/usr/bin/env python
import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import dask
import dask_cuda
import distributed
import numpy
import pandas
import psutil
import pynvml
import xgboost
from dask.distributed import Client, LocalCluster, wait
from dask_cuda import LocalCUDACluster

from .datasets import factory as data_factory
from .dsk import algorithm
from .utils import TEST_SIZE, TemporaryDirectory, Timer, fprint, add_device_param, add_hyper_param

try:
    import cudf
    import dask_cudf
except ImportError:
    cudf = None
    dask_cudf = None
try:
    import cupy
except ImportError:
    cupy = None


def packages_version() -> Dict[str, Optional[str]]:
    packages = {
        "numpy": numpy.__version__,
        "dask": dask.__version__,
        "pandas": pandas.__version__,
        "distributed": distributed.__version__,
        "cudf": cudf.__version__ if cudf else None,
        "dask_cudf": dask_cudf.__version__ if dask_cudf else None,
        "dask_cuda": dask_cuda.__version__,
        "xgboost": xgboost.__version__,
        "cupy": cupy.__version__ if cupy else None,
        "pynvml": pynvml.__version__,
    }
    return packages


def print_version() -> None:
    fprint("Package version:")
    packages = packages_version()
    for name, version in packages.items():
        fprint("- " + name + ":", version)
    fprint()


def main(args: argparse.Namespace) -> None:
    print_version()

    if not os.path.exists(args.temporary_directory):
        os.mkdir(args.temporary_directory)

    def cluster_type(*user_args: Any, **kwargs: Any) -> LocalCluster:
        if args.device == "CPU":
            return LocalCluster(*user_args, **kwargs)
        else:
            total_gpus = dask_cuda.utils.get_n_gpus()
            assert args.workers is None or args.workers <= total_gpus
            return LocalCUDACluster(*user_args, **kwargs)

    def run_benchmark(client: Client) -> None:
        from .dsk import make_dense_regression

        X, y = make_dense_regression(
            args.device, args.n_samples, args.n_features, random_state=1994
        )

        if args.valid:
            from dask_ml.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, random_state=1994, test_size=TEST_SIZE
            )
            with Timer("dask", "wait"):
                X_train, y_train, X_test, y_test = client.persist(
                    [X_train, y_train, X_test, y_test]
                )
                wait([X_train, y_train, X_test, y_test])
        else:
            X_train, X_test, y_train, y_test = X, None, y, None
            with Timer("dask", "wait"):
                X_train, y_train = client.persist([X_train, y_train])
                wait([X_train, y_train])

        algo = algorithm.factory(args.tree_method, "reg:squarederror", client, args)
        eval_results: xgboost.callback.TrainingCallback.EvalsLog = algo.fit(X, y)
        if args.model_out is not None:
            algo.booster.save_model(args.model_out)

    with TemporaryDirectory(args.temporary_directory):
        if args.scheduler is not None:
            with Client(scheduler_file=args.scheduler) as client:
                run_benchmark(client)
        else:
            with cluster_type(
                n_workers=args.workers, threads_per_worker=args.cpus
            ) as cluster:
                print("dashboard link:", cluster.dashboard_link)
                with Client(cluster) as client:
                    run_benchmark(client)

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    if args.device == "cuda":
        pynvml.nvmlInit()
        devices: List[str] = []
        n_devices = pynvml.nvmlDeviceGetCount()
        for i in range(n_devices):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name: bytes = pynvml.nvmlDeviceGetName(handle)
            devices.append(name.decode("utf-8"))

    # Don't override the previous result.
    i = 0
    while True:
        f = args.tree_method + "-rounds:" + str(args.rounds) + "-" + str(i) + ".json"
        path = os.path.join(args.output_directory, f)
        if os.path.exists(path):
            i += 1
            continue
        with open(path, "w") as fd:
            timer = Timer.global_timer()
            json.dump(timer, fd, indent=2)
            break


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Arguments for benchmarking with XGBoost dask."
    )

    subsparsers = parser.add_subparsers(dest="command")
    dg_parser = subsparsers.add_parser("datagen")
    bh_parser = subsparsers.add_parser("bench")

    dg_parser.add_argument(
        "--n_samples", type=int, help="Number of samples for generated dataset."
    )
    dg_parser.add_argument(
        "--n_features", type=int, help="Number of features for generated dataset."
    )
    dg_parser.add_argument("--sparsity", type=float, help="Sparsity of generated dataset.")
    dg_parser = add_device_param(dg_parser)

    bh_parser.add_argument(
        "--temporary-directory",
        type=str,
        help="Temporary directory used for dask.",
        default="dask_workspace",
    )
    bh_parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory storing benchmark results.",
        default="benchmark_outputs",
    )
    bh_parser.add_argument("--valid", action="store_true")
    bh_parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler address.  Use local cluster by default.",
        default=None,
    )
    bh_parser = add_device_param(bh_parser)
    bh_parser.add_argument("--workers", type=int, help="Number of Workers", default=None)
    bh_parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUs, used for setting number of threads.",
        default=psutil.cpu_count(logical=False),
    )
    bh_parser = add_hyper_param(bh_parser)

    # output model
    parser.add_argument(
        "--model-out",
        type=str,
        default=None,
    )
    args = parser.parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        sys.exit(0)
