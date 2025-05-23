import argparse
from time import time
from typing import Any, Dict, Optional, Union

import tqdm
import xgboost as xgb
from distributed import Client
from typing_extensions import TypeAlias
from xgboost import dask as dxgb
from xgboost.collective import Config as CollCfg

from ..utils import DC, ID, Progress, Timer, fprint

EvalsLog: TypeAlias = xgb.callback.TrainingCallback.EvalsLog


class XgbDaskBase:
    def __init__(
        self, parameters: dict, rounds: int, client: Client, should_eval: bool
    ) -> None:
        self.parameters = parameters
        self.client = client
        self.num_boost_round = rounds
        self.name: str = "base"
        self.should_eval = should_eval

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
        self, parameters: dict, rounds: int, n_bins: int, client: Client, eval: bool
    ) -> None:
        super().__init__(parameters, rounds, client, eval)
        self.name = "xgboost-dask-cpu-hist"
        self.parameters["tree_method"] = "hist"
        self.parameters["max_bin"] = n_bins

    def fit(self, X: DC, y: DC, weight: Optional[DC] = None) -> EvalsLog:
        with xgb.config_context(verbosity=1):
            with Timer(self.name, "DaskQuantileDMatrix"):
                dtrain = dxgb.DaskQuantileDMatrix(
                    self.client,
                    data=X,
                    label=y,
                    weight=weight,
                    enable_categorical=True,
                    max_bin=self.parameters["max_bin"],
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
        self,
        parameters: dict,
        rounds: int,
        n_bins: int,
        client: Client,
        should_eval: bool,
    ) -> None:
        XgbDaskBase.__init__(self, parameters, rounds, client, should_eval)
        self.name = "xgboost-dask-gpu-hist"
        self.parameters["tree_method"] = "hist"
        self.parameters["max_bin"] = n_bins
        self.parameters["device"] = "cuda"

    def fit(
        self,
        X: DC,
        y: DC,
        weight: Optional[DC] = None,
        # validation
        X_valid: DC | None = None,
        y_valid: DC | None = None,
    ) -> EvalsLog:
        with xgb.config_context(verbosity=1):
            with Timer(self.name, "DaskQuantileDMatrix"):
                dtrain = dxgb.DaskQuantileDMatrix(
                    self.client,
                    data=X,
                    label=y,
                    weight=weight,
                    enable_categorical=True,
                    max_bin=self.parameters["max_bin"],
                )

            if self.should_eval:
                callbacks = []
                with Timer(self.name, "DaskQuantileDMatrix-1"):
                    dvalid = dxgb.DaskQuantileDMatrix(
                        self.client,
                        data=X_valid,
                        label=y_valid,
                        enable_categorical=True,
                        ref=dtrain,
                    )
                evals = [(dtrain, "Train"), (dvalid, "Valid")]
            else:
                callbacks = [Progress(self.num_boost_round)]
                evals = []

            with xgb.config_context(use_rmm=True, verbosity=2):
                with Timer(self.name, "train"):
                    output = dxgb.train(
                        client=self.client,
                        params=self.parameters,
                        dtrain=dtrain,
                        evals=evals,
                        num_boost_round=self.num_boost_round,
                        callbacks=callbacks,
                        coll_cfg=CollCfg(timeout=60),
                    )
                    self.booster = output["booster"]
                    return output["history"]


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
) -> XgbDaskBase:
    parameters = {
        "max_depth": args.max_depth,
        "grow_policy": args.policy,
        "objective": task,
        "subsample": args.subsample,
        "colsample_bynode": args.colsample_bynode,
    }

    fprint("parameters:", parameters)
    should_eval = args.valid == 1

    assert client is not None
    if name == "hist" and args.device == "cuda":
        return XgbDaskGpuHist(
            parameters, args.n_rounds, args.n_bins, client, should_eval
        )
    elif name == "approx" and args.device == "cuda":
        return XgbDaskGpuApprox(parameters, args.n_rounds, client, should_eval)
    elif name == "hist" and args.device == "cpu":
        return XgbDaskCpuHist(
            parameters, args.n_rounds, args.n_bins, client, should_eval
        )
    elif name == "approx" and args.device == "cpu":
        return XgbDaskCpuApprox(parameters, args.n_rounds, client, should_eval)

    raise ValueError(f"Unknown tree method: {name}")
