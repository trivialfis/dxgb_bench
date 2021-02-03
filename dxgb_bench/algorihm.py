from xgboost import dask as dxgb
from .utils import Timer, fprint


class XgbDaskBase:
    def __init__(self, parameters, rounds, client):
        self.parameters = parameters
        self.client = client
        self.num_boost_round = rounds

    def fit(self, X, y, weight=None):
        with Timer(self.name, "DMatrix"):
            dtrain = dxgb.DaskDMatrix(self.client, data=X, label=y, weight=weight)
        with Timer(self.name, "train"):
            output = dxgb.train(
                client=self.client,
                params=self.parameters,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
            )
            self.output = output
            return output

    def predict(self, X):
        with Timer(self.name, "predict"):
            predictions = dxgb.inplace_predict(self.client, self.output, X)
            return predictions


class XgbDaskGpuHist(XgbDaskBase):
    def __init__(self, parameters, rounds, client):
        super().__init__(parameters, rounds, client)
        self.name = "xgboost-dask-gpu-hist"
        self.parameters["tree_method"] = "gpu_hist"

    def fit(self, X, y, weight=None):
        with Timer(self.name, "DMatrix"):
            dtrain = dxgb.DaskDeviceQuantileDMatrix(
                self.client, data=X, label=y, weight=weight
            )
        with Timer(self.name, "train"):
            output = dxgb.train(
                client=self.client,
                params=self.parameters,
                dtrain=dtrain,
                num_boost_round=self.num_boost_round,
            )
            self.output = output
            return output


class XgbDaskCpuHist(XgbDaskBase):
    def __init__(self, parameters, rounds, client):
        super().__init__(parameters, rounds, client)
        self.name = "xgboost-dask-cpu-hist"
        self.parameters["tree_method"] = "hist"


class XgbDaskCpuApprox(XgbDaskBase):
    def __init__(self, parameters, rounds, client):
        super().__init__(parameters, rounds, client)
        self.name = "xgboost-dask-cpu-approx"
        self.parameters["tree_method"] = "approx"


def factory(name, task, client, args):
    parameters = {"max_depth": 8, "objective": task}
    if name == "xgboost-dask-gpu-hist":
        return XgbDaskGpuHist(parameters, args.rounds, client)
    elif name == "xgboost-dask-cpu-hist":
        return XgbDaskCpuHist(parameters, args.rounds, client)
    elif name == "xgboost-dask-cpu-approx":
        return XgbDaskCpuApprox(parameters, args.rounds, client)
