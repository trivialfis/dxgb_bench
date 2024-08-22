import argparse
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple, TypeAlias, Union
from urllib.request import urlretrieve

import pandas
import tqdm
import xgboost as xgb

try:
    import cudf
    import dask_cudf
    from dask import array as da
    from dask import dataframe as dd

    DC: TypeAlias = Union[da.Array, dd.DataFrame, dd.Series]  # dask collection
    ID = Union[cudf.DataFrame, pandas.DataFrame, cudf.Series, pandas.Series]  # input data
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
    blocksize=dd.io.csv.AUTO_BLOCKSIZE,
) -> DType:
    if backend == "dask_cudf":
        df = dask_cudf.read_csv(
            path, delimiter=sep, dtype=dtype, header=None, names=names
        )
    elif backend == "dask":
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


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(block_size / 1024)
    else:
        pbar.close()
        pbar = None


class DataSet:
    uri: Optional[str] = None

    def retrieve(self, local_directory: str) -> str:
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
        assert self.uri
        filename = os.path.join(local_directory, os.path.basename(self.uri))
        if not os.path.exists(filename):
            fprint(
                "Retrieving from {uri} to {filename}".format(
                    uri=self.uri, filename=filename
                )
            )
            urlretrieve(self.uri, filename, show_progress)
        return filename

    def extra_args(self) -> Dict[str, Any]:
        return {}

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        raise NotImplementedError()


global_timer: Dict[str, Dict[str, float]] = {}


class Timer:
    def __init__(self, name: str, proc: str):
        self.name = name
        self.proc = proc + " (sec)"

    def __enter__(self) -> "Timer":
        self.start = time.time()
        fprint(self.name, self.proc, "started: ", time.ctime())
        return self

    def __exit__(self, type, value, traceback):
        end = time.time()
        if self.name not in global_timer.keys():
            global_timer[self.name] = {}
        global_timer[self.name][self.proc] = end - self.start
        fprint(self.name, self.proc, "ended in: ", end - self.start, "seconds.")

    @staticmethod
    def global_timer() -> Dict[str, Dict[str, float]]:
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

    def after_iteration(
        self, model: xgb.Booster, epoch: int, evals_log: EvalsLog
    ) -> bool:
        self.pbar.update(1)
        return False

    def after_training(self, model: xgb.Booster) -> xgb.Booster:
        self.end = time.time()
        self.pbar.close()
        return model
