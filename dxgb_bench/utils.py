# Copyright (c) 2020-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import warnings
from dataclasses import asdict, dataclass
from functools import cache
from typing import TYPE_CHECKING, Any, Callable, Dict, TypeAlias, Union

from packaging.version import parse as parse_version

try:
    import nvtx
except ImportError as e:
    warnings.warn(str(e), UserWarning)
    nvtx = None

import pandas
import tqdm
import xgboost as xgb
from xgboost.compat import import_cupy

if TYPE_CHECKING:
    import cudf
    from dask import array as da
    from dask import dataframe as dd

    DC: TypeAlias = Union[da.Array, dd.DataFrame, dd.Series]  # dask collection
    ID = Union[
        cudf.DataFrame, pandas.DataFrame, cudf.Series, pandas.Series
    ]  # input data
    DType = Union[
        cudf.DataFrame,
        pandas.DataFrame,
        cudf.Series,
        pandas.Series,
        da.Array,
        dd.DataFrame,
        dd.Series,
    ]


def fprint(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


fprint.__doc__ = print.__doc__

EvalsLog: TypeAlias = xgb.callback.TrainingCallback.EvalsLog


def read_csv(
    path: str,
    sep: str,
    dtype,
    header,
    names,
    backend,
    skiprows=0,
    blocksize=None,
) -> DType:
    if backend == "dask_cudf":
        import dask_cudf

        df = dask_cudf.read_csv(
            path, delimiter=sep, dtype=dtype, header=None, names=names
        )
    elif backend == "dask":
        if blocksize is None:
            blocksize = dd.io.csv.AUTO_BLOCKSIZE
        df = dd.read_csv(
            path,
            names=names,
            blocksize=blocksize,
            engine="python",
            sep=sep,
            skiprows=skiprows,
            dtype=dtype,
            header=header,
        )
    elif backend == "cudf":
        import cudf

        df = cudf.read_csv(path, delimiter=sep, dtype=dtype, header=None, names=names)
    else:
        df = None
        raise ValueError("Unknown read_csv backend:", backend)
    return df


pbar = None


def show_progress(block_num: int, block_size: int, total_size: int) -> None:
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None


GlobalTimer: TypeAlias = Dict[str, Dict[str, float]]

global_timer: GlobalTimer = {}


class Timer:
    def __init__(self, name: str, proc: str, logger: Callable = fprint) -> None:
        self.name = name
        self.proc = proc
        self.proc_name = proc
        self.range_id = None
        self.logger = logger

    def __enter__(self) -> "Timer":
        if nvtx is not None:
            self.range_id = nvtx.start_range(self.name + "-" + self.proc)
        self.start = time.time()
        msg = f"{self.name} {self.proc_name} started: {time.ctime()}"
        self.logger(msg)
        return self

    def __exit__(self, t: None, value: None, traceback: None) -> None:
        if self.range_id is not None:
            nvtx.end_range(self.range_id)
        end = time.time()
        if self.name not in global_timer.keys():
            s = time.time()
            global_timer[self.name] = {self.proc_name: s - s}
        if self.proc_name not in global_timer[self.name]:
            s = time.time()
            global_timer[self.name][self.proc_name] = s - s
        global_timer[self.name][self.proc_name] += end - self.start
        msg = f"{self.name} {self.proc_name} ended in: {end - self.start} seconds."
        self.logger(msg)

    @staticmethod
    def global_timer() -> GlobalTimer:
        return global_timer

    @staticmethod
    def reset() -> None:
        global global_timer
        global_timer = {}


class Progress(xgb.callback.TrainingCallback):
    """A callback function for displaying training progress."""

    def __init__(self, n_rounds: int) -> None:
        super().__init__()
        self.n_rounds = n_rounds

    def before_training(self, model: xgb.Booster) -> xgb.Booster:
        self.start = time.time()
        self.pbar = tqdm.tqdm(total=self.n_rounds, unit="iter")
        return model

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        self.end = time.time()
        self.pbar.close()
        return model

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: EvalsLog
    ) -> bool:
        progress_desc = []
        for data, metrics in evals_log.items():
            for m, res in metrics.items():
                desc = f"{data}-{m}:{res[-1]:.4f}"
                progress_desc.append(desc)
        self.pbar.set_description("|".join(progress_desc), refresh=True)
        self.pbar.update(1)
        return False


def div_roundup(a: int, b: int) -> int:
    """Round up for division."""
    return math.ceil(a / b)


def add_target_type(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--target_type",
        choices=["bin", "reg"],
        default="reg",
        help="Type of the target, either binary classification or regression.",
    )
    return parser


def add_data_params(
    parser: argparse.ArgumentParser,
    required: bool,
    n_features: int | None = None,
) -> argparse.ArgumentParser:
    parser.add_argument("--n_samples_per_batch", type=int, required=required)
    if n_features is not None:
        parser.add_argument(
            "--n_features", type=int, required=required, default=n_features
        )
    else:
        parser.add_argument("--n_features", type=int, required=required)
    parser.add_argument("--n_batches", type=int, default=1)
    parser.add_argument("--assparse", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.0)
    parser.add_argument("--fmt", choices=["auto", "npy", "npz", "kio"], default="auto")
    parser = add_target_type(parser)
    return parser


def add_device_param(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu"],
        help="cpu or cuda",
        default="cuda",
    )
    return parser


def add_rmm_param(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--mr",
        choices=["arena", "binning", "pool"],
        default="arena",
        help="Name of the RMM memory resource.",
    )
    return parser


def add_hyper_param(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--tree_method", type=str, help="Used algorithm", default="hist"
    )
    parser.add_argument(
        "--n_rounds", type=int, default=128, help="Number of boosting rounds."
    )
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument(
        "--policy", type=str, default="depthwise", choices=["lossguide", "depthwise"]
    )
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--subsample", type=float, default=None)
    parser.add_argument("--colsample_bynode", type=float, default=None)
    parser.add_argument("--colsample_bytree", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--min_child_weight", type=float, default=None)
    parser.add_argument("--reg_lambda", type=float, default=None)
    parser.add_argument("--verbosity", choices=[0, 1, 2, 3], default=1, type=int)
    # DMatrix
    parser.add_argument("--cache_host_ratio", type=float, required=False)
    return parser


def make_params_from_args(args: argparse.Namespace) -> dict[str, Any]:
    params = {
        "tree_method": args.tree_method,
        "max_depth": args.max_depth,
        "grow_policy": args.policy,
        "subsample": args.subsample,
        "colsample_bynode": args.colsample_bynode,
        "colsample_bytree": args.colsample_bytree,
        "max_bin": args.n_bins,
        "lambda": args.reg_lambda,
        "gamma": args.gamma,
        "eta": args.eta,
        "min_child_weight": args.min_child_weight,
        "device": args.device,
        "verbosity": args.verbosity,
        "objective": "binary:logistic" if args.target_type == "bin" else None,
    }
    fprint(params)
    return params


def split_path(path: str) -> list[str]:
    if path.find(",") != -1:
        path_ls = path.split(",")
    else:
        path_ls = [path]
    return path_ls


TEST_SIZE = 0.2  # Emulate to 5-fold CV
DFT_OUT = os.path.join(os.curdir, "data")


def setup_rmm(mr_name: str) -> None:
    import rmm
    from cuda.bindings import runtime as cudart
    from rmm.allocators.cupy import rmm_cupy_allocator

    cp = import_cupy()

    status, free, total = cudart.cudaMemGetInfo()
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))
    print("Setup rmm, total:", total, "free:", free)

    match mr_name:
        case "arena":
            fprint("Use `ArenaMemoryResource`.")
            mr = rmm.mr.CudaMemoryResource()
            mr = rmm.mr.ArenaMemoryResource(mr, arena_size=int(total * 0.9))
            status, free, total = cudart.cudaMemGetInfo()
        case "binning":
            fprint("Use `BinningMemoryResource`.")
            mr = rmm.mr.CudaAsyncMemoryResource(
                initial_pool_size=int(total * 0.8),
                release_threshold=int(total * 0.95),
                enable_ipc=False,
            )
            mr = rmm.mr.BinningMemoryResource(mr, 21, 25)
        case "pool":
            fprint("Use `PoolMemoryResource`.")
            mr = rmm.mr.CudaMemoryResource()
            mr = rmm.mr.PoolMemoryResource(
                mr,
                initial_pool_size=int(total * 0.8),
                release_threshold=int(total * 0.95),
            )

    mr = rmm.mr.LoggingResourceAdaptor(mr, log_file_name="rmm_log")
    rmm.mr.set_current_device_resource(mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)


def peak_rmm_memory_bytes(path: str = "rmm_log.dev0") -> int:
    import pandas as pd

    current = 0  # current memory usge
    peak = 0  # peak memory usage

    # The pointers that are currently in use
    recs: dict[str, int] = {}

    df = pd.read_csv(path)

    for i in range(df.shape[0]):
        row = df.iloc[i, :]
        if row.loc["Action"] == "allocate":
            current += row["Size"]
            assert row["Pointer"] not in recs
            recs[row["Pointer"]] = i
        elif row.loc["Action"] == "free":
            current -= row["Size"]
            recs.pop(row["Pointer"])
        if current > peak:
            peak = current
    return peak


def c2cinfo() -> int | None:
    import pynvml as nm  # mamba install nvidia-ml-py -c rapidsai -c nvidia

    # Or just run `nvidia-smi c2c -i 0 -s`. If there's no C2C device, it returns 3.
    # This script uses nvml to do the same thing.
    nm.nvmlInit()

    hdl = nm.nvmlDeviceGetHandleByIndex(0)
    try:
        info = nm.nvmlDeviceGetC2cModeInfoV1(hdl)
    except nm.NVMLError_NotSupported:
        info = None

    # NVML_FI_DEV_C2C_LINK_GET_MAX_BW: C2C Link Speed in MBps for active links.
    if info is not None and info.isC2cEnabled == 1:
        lc, bw = nm.nvmlDeviceGetFieldValues(
            hdl, [nm.NVML_FI_DEV_C2C_LINK_COUNT, nm.NVML_FI_DEV_C2C_LINK_GET_MAX_BW]
        )
        print("Link count:", lc.value.siVal, "Bandwidth:", bw.value.sllVal)
    else:
        lc, bw = None, None

    nm.nvmlShutdown()

    if lc is not None:
        return int(lc.value.siVal)
    return lc


def machine_info(device: str) -> dict:
    system = platform.system()
    machine = platform.machine()
    cpus = os.cpu_count()
    assert cpus

    info: dict[str, Any] = {"system": system, "arch": machine, "cpus": cpus}

    def query_smi(what: str) -> list[str]:
        # We can query mutiple fields in one go, but I don't want to parse csv files.
        r = subprocess.run(
            f"nvidia-smi --query-gpu={what} --format=csv".split(" "),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert r.returncode == 0, r.stdout
        lines = r.stdout.decode("utf-8").splitlines()
        assert lines[0] == what
        return lines[1:]

    if device != "cpu":
        info["gpus"] = query_smi("name")
        info["drivers"] = query_smi("driver_version")
        info["c2c"] = c2cinfo()
    else:
        info["gpus"] = None
        info["drivers"] = None
        info["c2c"] = None

    return info


def mkdirs(outdirs: list[str]) -> None:
    for d in outdirs:
        if not os.path.exists(d):
            os.mkdir(d)


@dataclass
class Opts:
    n_samples_per_batch: int
    n_features: int
    n_batches: int
    sparsity: float
    on_the_fly: bool
    validation: bool
    device: str
    mr: str | None
    target_type: str
    cache_host_ratio: float | None


def merge_opts(opts: Opts, params: dict[str, Any]) -> dict[str, Any]:
    opts_dict = asdict(opts)
    # merge with checks.
    for k, v in params.items():
        if k in opts_dict:
            assert v == opts_dict[k]
        else:
            opts_dict[k] = v
    return opts_dict


@cache
def has_chr() -> bool:
    ver = parse_version(xgb.__version__)
    return (ver.major == 3 and ver.minor > 0) or ver.major > 3


def save_results(results: dict[str, Any], prefix: str) -> None:
    k = 0
    path = prefix + f"-{k}.json"
    while os.path.exists(path):
        k += 1
        path = prefix + f"-{k}.json"

    print(f"saving results to: {path}")
    with open(path, "w") as fd:
        json.dump(results, fd, indent=2)


def fill_opts_shape(
    opts: Opts, Xy: xgb.DMatrix, Xy_valid: xgb.DMatrix | None, n_batches: int
) -> Opts:
    def get_valid(field: str) -> int:
        if Xy_valid is not None:
            v = getattr(Xy_valid, field)()
        else:
            v = 0
        return v

    n_samples = Xy.num_row() + get_valid("num_row")
    assert n_samples % n_batches == 0
    density = (Xy.num_nonmissing() + get_valid("num_nonmissing")) / (
        n_samples * Xy.num_col()
    )
    sparsity = 1.0 - density
    opts.n_samples_per_batch = n_samples // n_batches
    opts.n_features = Xy.num_col()
    opts.sparsity = sparsity
    opts.n_batches = n_batches
    return opts
