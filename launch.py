#!/usr/bin/env python
import argparse
import json
import os
import subprocess
from time import time
from typing import Any, Dict, List, Tuple, Union

import psutil


def check_call(*args: Any, **kwargs: Any) -> None:
    print("running: ", *args)
    subprocess.check_call(*args, **kwargs)


Args = Dict[str, Union[List[Union[str, int, float]], str, int]]

gpu_hist = "xgboost-gpu-hist"
cpu_hist = "xgboost-cpu-hist"

mortgage: Args = {
    "data": "mortgage",
    "algo": [gpu_hist],
    "colsample_bynode": [0.6, 1.0],
    "cpus": psutil.cpu_count(logical=True),
    "rounds": [200, 500],
    "backend": "cudf",
    "max-depth": [8, 16],
    "policy": ["depthwise", "lossguide"],
    "workers": 1,
    "f32-hist": 0,
    "eval": 0,
}

mortgage_2y = mortgage.copy()
mortgage_2y["data"] = "mortgage:2"
mortgage_2y["backend"] = "dask_cudf"
mortgage_2y["workers"] = 2


higgs = mortgage.copy()
higgs["data"] = "higgs"

covtype = mortgage.copy()
covtype["data"] = "covtype"

year = mortgage.copy()
year["data"] = "year"

airline = mortgage.copy()
airline["backend"] = "cudf"
airline["data"] = "airline"

generated = mortgage.copy()
generated["data"] = "generated"
generated["n_samples"] = int(2e7)
generated["n_features"] = [256, 64]
generated["sparsity"] = [0.8, 0.4, 0.1]
generated["task"] = "reg"

epsilon = mortgage.copy()
epsilon["data"] = "epsilon"

history = []


def rec(v_i: int, variables: list, spec: list, resume: bool) -> None:
    if v_i == len(variables):
        cmd = ["dxgb-bench"] + spec

        if resume and cmd in history:
            print(f"Skipping: {cmd}")
            return

        check_call(cmd)

        history.append(tuple(cmd))
        with open("./history.json", "w") as fd:
            json.dump(history, fd)

        return

    n_varients = len(variables[v_i])
    for i in range(n_varients):
        k, v = variables[v_i][i]
        appended = spec.copy()
        appended.append(k + "=" + str(v))
        rec(v_i + 1, variables, appended, resume)


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
    resume = args.resume == 1
    rec(0, variables, spec, resume)


def main(local_directory: str) -> None:
    global history
    if os.path.exists("./history.json"):
        with open("./history.json", "r") as fd:
            history = json.load(fd)

    local_directory = os.path.expanduser(local_directory)
    launch(local_directory, generated)

    launch(local_directory, mortgage)

    launch(local_directory, mortgage_2y)

    launch(local_directory, higgs)

    launch(local_directory, covtype)

    launch(local_directory, year)

    launch(local_directory, airline)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--local-directory",
        type=str,
        help="Local directory for storing the dataset.",
        required=True,
    )
    parser.add_argument(
        "--resume",
        type=int,
        help="Resume from last session.",
        default=0,
    )
    args = parser.parse_args()
    start = time()
    main(args.local_directory)
    end = time()
    print("Total duration:", end - start)
