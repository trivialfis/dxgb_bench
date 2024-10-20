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

from . import algorihm
from .datasets import factory as data_factory
from .utils import TemporaryDirectory, Timer, fprint

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
    args.local_directory = os.path.expanduser(args.local_directory)

    if not os.path.exists(args.temporary_directory):
        os.mkdir(args.temporary_directory)

    def cluster_type(*user_args: Any, **kwargs: Any) -> LocalCluster:
        if args.device == "CPU":
            return LocalCluster(*user_args, **kwargs)
        else:
            total_gpus = dask_cuda.utils.get_n_gpus()
            assert args.workers is None or args.workers <= total_gpus
            return LocalCUDACluster(*user_args, **kwargs)

    def run_benchmark(client: Optional[Client]) -> None:
        d, task = data_factory(args.data, args)
        (X, y, w) = d.load(args)
        extra_args = d.extra_args()

        if args.backend.find("dask") != -1:
            with Timer(args.backend, "dask-wait"):
                X = X.persist()
                y = y.persist()
                wait(X)
                wait(y)
        else:
            cupy.cuda.runtime.deviceSynchronize()

        algo = algorihm.factory(args.algo, task, client, args, extra_args)
        eval_results: xgboost.callback.TrainingCallback.EvalsLog = algo.fit(X, y, w)
        print("Evaluation results:", eval_results)
        if args.model_out is not None:
            algo.booster.save_model(args.model_out)

        timer = Timer.global_timer()
        dataset_results = list(eval_results.values())
        if args.eval:
            assert len(dataset_results) == 1

            for k, v in dataset_results[0].items():
                timer[k] = v[-1]

    if args.backend.find("dask") == -1:
        run_benchmark(None)
    else:
        with TemporaryDirectory(args.temporary_directory):
            # race condition for creating directory.
            # dask.config.set({'temporary_directory': args.temporary_directory})
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

    if args.device == "GPU":
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
        f = (
            args.algo
            + "-rounds:"
            + str(args.rounds)
            + "-data:"
            + args.data
            + "-"
            + str(i)
            + ".json"
        )
        path = os.path.join(args.output_directory, f)
        if os.path.exists(path):
            i += 1
            continue
        with open(path, "w") as fd:
            timer = Timer.global_timer()
            timer["packages"] = packages_version()
            timer["args"] = args.__dict__
            timer["devices"] = devices
            json.dump(timer, fd, indent=2)
            break


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        description="Arguments for benchmarking with XGBoost dask."
    )
    parser.add_argument(
        "--local-directory",
        type=str,
        help="Local directory for storing the dataset.",
        default="dxgb_bench_workspace",
    )
    parser.add_argument(
        "--temporary-directory",
        type=str,
        help="Temporary directory used for dask.",
        default="dask_workspace",
    )
    parser.add_argument(
        "--output-directory",
        type=str,
        help="Directory storing benchmark results.",
        default="benchmark_outputs",
    )
    parser.add_argument(
        "--eval",
        type=int,
        choices=[0, 1],
        help="Whether XGBoost should run evaluation at each iteration.",
        default=0,
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        help="Scheduler address.  Use local cluster by default.",
        default=None,
    )
    parser.add_argument("--device", type=str, help="CPU or GPU", default="GPU")
    parser.add_argument("--workers", type=int, help="Number of Workers", default=None)
    parser.add_argument(
        "--cpus",
        type=int,
        help="Number of CPUs, used for setting number of threads.",
        default=psutil.cpu_count(logical=False),
    )
    parser.add_argument(
        "--algo", type=str, help="Used algorithm", default="xgboost-gpu-hist"
    )
    parser.add_argument(
        "--rounds", type=int, default=1000, help="Number of boosting rounds."
    )
    # data
    parser.add_argument("--data", type=str, help="Name of dataset.", required=True)
    parser.add_argument(
        "--n_samples", type=int, help="Number of samples for generated dataset."
    )
    parser.add_argument(
        "--n_features", type=int, help="Number of features for generated dataset."
    )
    parser.add_argument("--sparsity", type=float, help="Sparsity of generated dataset.")
    parser.add_argument(
        "--task",
        type=str,
        help="Type of generated dataset.",
        choices=["reg", "cls", "aft", "rank"],
    )
    # tree parameters
    parser.add_argument(
        "--backend", type=str, help="Data loading backend.", default="dask_cudf"
    )
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument(
        "--policy", type=str, default="depthwise", choices=["lossguide", "depthwise"]
    )
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample_bynode", type=float, default=None)
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
