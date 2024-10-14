# Copyright (c) 2020-2024, Jiaming Yuan.  All rights reserved.
import math
import os
import shutil
import sys
import time
import warnings
from typing import Any, Dict, TypeAlias, Union

try:
    import nvtx
except ImportError as e:
    warnings.warn(str(e), UserWarning)
    nvtx = None

import pandas
import tqdm
import xgboost as xgb

try:
    import cudf
    import dask_cudf
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

except ImportError:
    da = None
    dd = None
    cudf = None
    dask_cudf = None

    DC: TypeAlias = Any
    ID: TypeAlias = Any
    DType: TypeAlias = Any


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
    def __init__(self, name: str, proc: str):
        self.name = name
        self.proc = proc
        self.proc_name = proc + " (sec)"
        self.range_id = None

    def __enter__(self) -> "Timer":
        if nvtx is not None:
            self.range_id = nvtx.start_range(self.name + "-" + self.proc)
        self.start = time.time()
        fprint(self.name, self.proc_name, "started: ", time.ctime())
        return self

    def __exit__(self, t: None, value: None, traceback: None) -> None:
        print(traceback, type(traceback))
        if self.range_id is not None:
            nvtx.end_range(self.range_id)
        end = time.time()
        if self.name not in global_timer.keys():
            global_timer[self.name] = {}
        global_timer[self.name][self.proc_name] = end - self.start
        fprint(self.name, self.proc_name, "ended in: ", end - self.start, "seconds.")

    @staticmethod
    def global_timer() -> GlobalTimer:
        return global_timer


class TemporaryDirectory:
    def __init__(self, path: str) -> None:
        self.path = path

    def __enter__(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

    def __exit__(self, *args):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)


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
