import tqdm
from urllib.request import urlretrieve
import sys
import time
import shutil
import argparse
import os
from typing import Union, Any, Dict, Tuple, Optional

from dask import dataframe as dd
from dask import array as da
import pandas

try:
    import cudf
    import dask_cudf
except ImportError:
    cudf = None
    dask_cudf = None


def fprint(*args: Any, **kwargs: Any) -> None:
    print(*args, **kwargs)
    sys.stdout.flush()


fprint.__doc__ = print.__doc__


DC = Union[da.Array, dd.DataFrame, dd.Series]  # dask collection
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
    uri = None

    def retrieve(self, local_directory: str) -> str:
        if not os.path.exists(local_directory):
            os.makedirs(local_directory)
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


global_timer = {}


class Timer:
    def __init__(self, name: str, proc: str):
        self.name = name
        self.proc = proc + " (sec)"

    def __enter__(self) -> None:
        self.start = time.time()
        fprint(self.name, self.proc, "started: ", time.ctime())

    def __exit__(self, type, value, traceback):
        end = time.time()
        if self.name not in global_timer.keys():
            global_timer[self.name] = {}
        global_timer[self.name][self.proc] = end - self.start
        fprint(self.name, self.proc, "ended in: ", end - self.start, "seconds.")

    @staticmethod
    def global_timer():
        return global_timer


class TemporaryDirectory:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.mkdir(self.path)

    def __exit__(self, *args):
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
