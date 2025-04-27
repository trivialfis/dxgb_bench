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
from cuda import cudart
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster, wait
from dask_cuda import LocalCUDACluster

from .dsk import (
    algorithm,
    load_dense_gather,
    make_dense_regression,
    make_dense_regression_scatter,
)
from .utils import (
    DFT_OUT,
    TEST_SIZE,
    TemporaryDirectory,
    Timer,
    add_device_param,
    add_hyper_param,
    fprint,
)

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


def cluster_type(
    args: argparse.Namespace, *user_args: Any, **kwargs: Any
) -> LocalCluster:
    if args.device == "cpu":
        return LocalCluster(*user_args, **kwargs)
    else:
        total_gpus = dask_cuda.utils.get_n_gpus()
        assert args.workers is None or args.workers <= total_gpus
        status, free, total = cudart.cudaMemGetInfo()
        if status != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(cudart.cudaGetErrorString(status))
        size = str(int(total * 0.9) / 1024**3) + "G"
        return LocalCUDACluster(*user_args, **kwargs, rmm_pool_size=size)


def datagen(args: argparse.Namespace) -> None:

    def doit_unified(client: Client) -> None:
        saveto = os.path.expanduser(args.saveto)
        X_path = os.path.join(saveto, "X")
        y_path = os.path.join(saveto, "y")

        if os.path.exists(X_path) or os.path.exists(y_path):
            raise ValueError("Please remove the old data first.")

        X, y = make_dense_regression(
            args.device, args.n_samples, args.n_features, random_state=1994
        )
        X, y = client.persist([X, y])

        X.to_parquet(X_path)
        y.to_parquet(y_path)

    def doit_scatter(client: Client) -> None:
        make_dense_regression_scatter(
            client,
            args.device,
            args.n_samples,
            args.n_features,
            saveto=args.saveto,
            local_test=args.local_test_fs,
            fmt="npy",
        )

    if args.scheduler is not None:
        with Client(scheduler_file=args.scheduler) as client:
            if args.scatter:
                doit_scatter(client)
            else:
                doit_unified(client)
    else:
        with cluster_type(
            args, n_workers=args.workers, threads_per_worker=args.cpus
        ) as cluster:
            with Client(cluster) as client:
                if args.scatter:
                    doit_scatter(client)
                else:
                    doit_unified(client)


def load_data(args: argparse.Namespace) -> tuple[dd.DataFrame, dd.DataFrame]:

    saveto = os.path.expanduser(args.loadfrom)
    X_path = os.path.join(saveto, "X")
    y_path = os.path.join(saveto, "y")

    X, y = dd.read_parquet(X_path), dd.read_parquet(y_path)
    if args.device.startswith("cuda"):
        X, y = X.to_backend("cudf"), y.to_backend("cudf")
    return X, y


def main(args: argparse.Namespace) -> None:
    print_version()

    if not os.path.exists(args.temporary_directory):
        os.mkdir(args.temporary_directory)

    def run_benchmark(client: Client) -> None:
        client.restart()
        with Timer("dask", "load"):
            if args.gather:
                device = "cpu"
                # device = args.device
                X, y = load_dense_gather(
                    client, device, args.loadfrom, args.local_test_fs
                )
                if not args.disable_device_qdm:
                    X = X.to_backend("cupy")
                    y = y.to_backend("cupy")
                    X, y = client.persist([X, y])
                wait([X, y])
            else:
                X, y = load_data(args)

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
        eval_results: xgboost.callback.TrainingCallback.EvalsLog = algo.fit(
            X_train, y_train, None, X_test, y_test
        )
        if args.model_out is not None:
            algo.booster.save_model(args.model_out)

    with TemporaryDirectory(args.temporary_directory):
        if args.scheduler is not None:
            with Client(scheduler_file=args.scheduler) as client:
                run_benchmark(client)
        else:
            with cluster_type(
                args, n_workers=args.workers, threads_per_worker=args.cpus
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
        f = args.tree_method + "-rounds:" + str(args.n_rounds) + "-" + str(i) + ".json"
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

    def add_sched(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument(
            "--scheduler",
            type=str,
            help="Scheduler address.  Use local cluster by default.",
            default=None,
        )
        parser.add_argument(
            "--workers", type=int, help="Number of Workers", default=None
        )
        parser.add_argument(
            "--cpus",
            type=int,
            help="Number of CPUs, used for setting number of threads.",
            default=psutil.cpu_count(logical=False),
        )
        return parser

    dg_parser.add_argument(
        "--n_samples", type=int, help="Number of samples for generated dataset."
    )
    dg_parser.add_argument(
        "--n_features", type=int, help="Number of features for generated dataset."
    )
    dg_parser.add_argument(
        "--sparsity", type=float, help="Sparsity of generated dataset."
    )
    dg_parser.add_argument("--saveto", type=str, default=DFT_OUT)
    dg_parser.add_argument(
        "--scatter",
        action="store_true",
        help="Scatter the generated files to individual workers. This is to handler clusters that don't have a unified file system view.",
    )
    dg_parser.add_argument("--local_test_fs", action="store_true")
    dg_parser = add_device_param(dg_parser)
    dg_parser = add_sched(dg_parser)

    bh_parser.add_argument(
        "--temporary-directory",
        type=str,
        help="Temporary directory used for dask.",
        default="dask_workspace",
    )
    bh_parser.add_argument("--loadfrom", type=str, default=DFT_OUT)
    bh_parser.add_argument("--gather", action="store_true")
    bh_parser.add_argument("--local_test_fs", action="store_true")

    bh_parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory storing benchmark results.",
        default="benchmark_outputs",
    )
    bh_parser.add_argument("--valid", action="store_true")
    bh_parser.add_argument("--disable-device-qdm", action="store_true")
    bh_parser = add_sched(bh_parser)
    bh_parser = add_device_param(bh_parser)
    bh_parser = add_hyper_param(bh_parser)

    # output model
    bh_parser.add_argument(
        "--model-out",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    try:
        if args.command == "datagen":
            datagen(args)
        else:
            main(args)
    except KeyboardInterrupt:
        sys.exit(0)
