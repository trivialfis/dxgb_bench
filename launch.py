#!/usr/bin/env python
import subprocess
import psutil
import argparse
from typing import Dict, Union, List, Any, Tuple


def check_call(*args: Any, **kwargs: Any) -> None:
    print("running: ", *args)
    subprocess.check_call(*args, **kwargs)


Args = Dict[str, Union[List[Union[str, int]], str, int]]

gpu_hist = "xgboost-gpu-hist"

mortgage: Args = {
    "data": "mortgage",
    "algo": "xgboost-gpu-hist",
    "cpus": psutil.cpu_count(logical=True),
    "rounds": 200,
    "backend": "cudf",
    "max-depth": 16,
    "policy": ["depthwise", "lossguide"],
    "workers": 1,
    "f32-hist": 0,
}

mortgage_distributed = mortgage.copy()
mortgage_distributed["workers"] = 2
mortgage_distributed["backend"] = "dask_cudf"

mortgage_2y = mortgage.copy()
mortgage_2y["data"] = "mortgage:2"
mortgage_2y["backend"] = "dask_cudf"
mortgage_2y["workers"] = 2


higgs = mortgage.copy()
higgs["data"] = "higgs"

higgs_distributed = higgs.copy()
higgs_distributed["backend"] = "dask_cudf"
higgs_distributed["workers"] = 2

covtype = mortgage.copy()
covtype["data"] = "covtype"

covtype_distributed = covtype.copy()
covtype_distributed["backend"] = "dask_cudf"
covtype_distributed["workers"] = 2

year = mortgage.copy()
year["data"] = "year"

year_distributed = year.copy()
year_distributed["backend"] = "dask_cudf"
year_distributed["workers"] = 2


def rec(v_i: int, variables: list, spec: list) -> None:
    if v_i == len(variables):
        cmd = ["dxgb-bench"] + spec
        check_call(cmd)
        return

    n_varients = len(variables[v_i])
    for i in range(n_varients):
        k, v = variables[v_i][i]
        appended = spec.copy()
        appended.append(k + "=" + str(v))
        rec(v_i + 1, variables, appended)


def launch(dirpath: str, parameters: Args) -> None:
    variables = []
    constants: List[Tuple[str, Union[str, int]]] = [("--local-directory", dirpath)]
    for key, value in parameters.items():
        prefix = "--" + key
        if isinstance(value, list):
            var = []
            for v in value:
                item = (prefix, v)
                var.append(item)
            variables.append(var)
        else:
            item = (prefix, value)
            constants.append(item)

    spec = [k + "=" + str(v) for k, v in constants]
    rec(0, variables, spec)


def main(local_directory: str) -> None:
    launch(local_directory, mortgage)
    launch(local_directory, mortgage_distributed)

    launch(local_directory, mortgage_2y)

    launch(local_directory, higgs)
    launch(local_directory, higgs_distributed)

    launch(local_directory, covtype)
    launch(local_directory, covtype_distributed)

    launch(local_directory, year)
    launch(local_directory, year_distributed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-directory",
        type=str,
        help="Local directory for storing the dataset.",
        required=True,
    )
    args = parser.parse_args()
    main(args.local_directory)
