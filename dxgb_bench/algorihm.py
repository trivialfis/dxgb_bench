import argparse
from time import time
from typing import Any, Dict, Optional, Union

import tqdm
import xgboost as xgb
from distributed import Client
from typing_extensions import TypeAlias
from xgboost import dask as dxgb

from .utils import DC, ID, Progress, Timer

EvalsLog: TypeAlias = xgb.callback.TrainingCallback.EvalsLog


class XgbDaskBase:
    def __init__(
        self, parameters: dict, rounds: int, client: Client, eval: bool
    ) -> None:
        self.parameters = parameters
        self.client = client
        self.num_boost_round = rounds
        self.name: str = "base"
        self.should_eval = eval

    def fit(self, X: DC, y: DC, weight: Optional[DC] = None) -> dict:
        with Timer(self.name, "DMatrix"):
            dtrain = dxgb.DaskDMatrix(self.client, data=X, label=y, weight=weight)

        if self.should_eval:
            callbacks = []
            evals = [(dtrain, "Train")]
        else:
            callbacks = [Progress(self.num_boost_round)]
            evals = []

        with Timer(self.name, "train"):
            output = dxgb.train(
                client=self.client,
                params=self.parameters,
                dtrain=dtrain,
                evals=evals,
                num_boost_round=self.num_boost_round,
                callbacks=callbacks,
            )
            self.booster = output["booster"]
            return output["history"]

    def predict(self, X: DC) -> DC:
        with Timer(self.name, "predict"):
            predictions = dxgb.inplace_predict(self.client, self.booster, X)
            return predictions


class XgbDaskCpuHist(XgbDaskBase):
    def __init__(
        self, parameters: dict, rounds: int, client: Client, eval: bool
    ) -> None:
        super().__init__(parameters, rounds, client, eval)
        self.name = "xgboost-dask-cpu-hist"
        self.parameters["tree_method"] = "hist"

    def fit(self, X: DC, y: DC, weight: Optional[DC] = None) -> EvalsLog:
        with xgb.config_context(verbosity=1):
            with Timer(self.name, "DaskQuantileDMatrix"):
                dtrain = dxgb.DaskQuantileDMatrix(
                    self.client, data=X, label=y, weight=weight, enable_categorical=True
                )

            if self.should_eval:
                callbacks = []
                evals = [(dtrain, "Train")]
            else:
                callbacks = [Progress(self.num_boost_round)]
                evals = []

            with Timer(self.name, "train"):
                output = dxgb.train(
                    client=self.client,
                    params=self.parameters,
                    dtrain=dtrain,
                    evals=evals,
                    num_boost_round=self.num_boost_round,
                    callbacks=callbacks,
                )
                self.booster = output["booster"]
                return output["history"]


class XgbDaskGpuHist(XgbDaskCpuHist):
    def __init__(
        self, parameters: dict, rounds: int, client: Client, should_eval: bool
    ) -> None:
        XgbDaskBase.__init__(self, parameters, rounds, client, should_eval)
        self.name = "xgboost-dask-gpu-hist"
        self.parameters["tree_method"] = "hist"
        self.parameters["device"] = "cuda"


class XgbDaskCpuApprox(XgbDaskBase):
    def __init__(
        self, parameters: dict, rounds: int, client: Client, eval: bool
    ) -> None:
        super().__init__(parameters, rounds, client, eval)
        self.name = "xgboost-dask-cpu-approx"
        self.parameters["tree_method"] = "approx"


class XgbDaskGpuApprox(XgbDaskBase):
    def __init__(
        self, parameters: dict, rounds: int, client: Client, eval: bool
    ) -> None:
        super().__init__(parameters, rounds, client, eval)
        self.name = "xgboost-dask-gpu-approx"
        self.parameters["tree_method"] = "approx"
        self.parameters["device"] = "cuda"


def factory(
    name: str,
    task: str,
    client: Client,
    args: argparse.Namespace,
    extra_args: Dict[str, Any],
) -> XgbDaskBase:
    parameters = {
        "max_depth": args.max_depth,
        "grow_policy": args.policy,
        "objective": task,
        "subsample": args.subsample,
        "colsample_bynode": args.colsample_bynode,
    }
    parameters.update(extra_args)

    if args.backend.find("dask") == -1:
        parameters["nthread"] = args.cpus

    print("parameters:", parameters, flush=True)
    should_eval = args.eval == 1

    assert client is not None
    if name == "xgboost-gpu-hist":
        return XgbDaskGpuHist(parameters, args.rounds, client, should_eval)
    elif name == "xgboost-gpu-approx":
        return XgbDaskGpuApprox(parameters, args.rounds, client, should_eval)
    elif name == "xgboost-cpu-hist":
        return XgbDaskCpuHist(parameters, args.rounds, client, should_eval)
    elif name == "xgboost-cpu-approx":
        return XgbDaskCpuApprox(parameters, args.rounds, client, should_eval)

    raise ValueError(
        "Unknown algorithm: ",
        name,
        " expecting one of the: {",
        "xgboost-gpu-hist",
        "xgboost-cpu-approx",
        "xgboost-cpu-hist",
        "}",
    )
