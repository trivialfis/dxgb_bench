import argparse
import os
import pickle
from typing import Optional, Tuple

import numpy as np
from sklearn.datasets import load_svmlight_file

from dxgb_bench.utils import DataSet, DType


class Epsilon(DataSet):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.backend.find("dask") != -1:
            raise ValueError("Unspported backend for Epsilon")

        url_train = (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
            "/epsilon_normalized.bz2"
        )
        url_test = (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
            "/epsilon_normalized.t.bz2"
        )
        self.local_directory = os.path.join(args.local_directory, "epsilon")
        self.uri = url_train
        train_path = self.retrieve(self.local_directory)
        self.uri = url_test
        test_path = self.retrieve(self.local_directory)

        self.pickle_path = os.path.join(self.local_directory, "epsilon-dxgb.pkl")
        if not os.path.exists(self.pickle_path):
            X_train, y_train = load_svmlight_file(train_path, dtype=np.float32)
            X_test, y_test = load_svmlight_file(test_path, dtype=np.float32)
            X_train = X_train.toarray()
            X_test = X_test.toarray()
            y_train[y_train <= 0] = 0
            y_test[y_test <= 0] = 0

            X_train = np.vstack((X_train, X_test))
            y_train = np.append(y_train, y_test)

            with open(os.path.join(self.pickle_path), "wb") as fd:
                pickle.dump((X_train, y_train), fd)

        self.task = "binary:logistic"

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        with open(self.pickle_path, "rb") as fd:
            X, y = pickle.load(fd)
        return X, y, None
