import os

import xgboost as xgb
from sklearn.datasets import load_svmlight_file

from dxgb_bench.utils import DataSet, Timer, read_csv

with Timer("loading_dataset") as t:
    if os.path.exists("dtrain.bin"):
        print("Loading existing DMatrix.")
        dtrain = xgb.DMatrix("dtrain.bin")
        dtest = xgb.DMatrix("dtest.bin")
    else:
        print("Generating DMatrix.")
        X, y = load_svmlight_file("./all.svm")


class URL(DataSet):
    def __init__(self, args):
        self.uri = (
            "http://archive.ics.uci.edu/ml/machine-learning-"
            "databases/url/url_svmlight.tar.gz"
        )
        self.local_directory = os.path.join(args.local_directory, "url")
        self.retrieve(self.local_directory)
        # https://docs.dask.org/en/latest/array-sparse.html
        raise NotImplementedError(
            """I don't know how to use dask with sparse dataset."""
        )

    def load(self, args):
        path = None
        url = read_csv(path)
        y = url.iloc[:, 0]
        X = url.iloc[:, 1:]
        return X, y, None
