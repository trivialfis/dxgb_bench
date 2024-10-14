import argparse
import os
import zipfile
from typing import Optional, Tuple

from ..utils import DType, fprint, read_csv
from .dataset import DataSet


class YearPrediction(DataSet):
    def __init__(self, args: argparse.Namespace) -> None:
        self.uri = (
            "https://archive.ics.uci.edu/ml/machine-learning-"
            "databases/00203/YearPredictionMSD.txt.zip"
        )
        self.local_directory = os.path.join(args.local_directory, "year_prediction")
        self.retrieve(self.local_directory)
        zip_path = os.path.join(self.local_directory, "YearPredictionMSD.txt.zip")
        self.csv_path = os.path.join(self.local_directory, "YearPredictionMSD.txt")
        if not os.path.exists(self.csv_path):
            with zipfile.ZipFile(zip_path, "r") as z:
                fprint("Extracting", zip_path)
                z.extractall(self.local_directory)

        self.task = "reg:squarederror"

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        year = read_csv(
            self.csv_path,
            header=None,
            names=None,
            backend=args.backend,
            dtype=None,
            sep=",",
        )
        X = year.iloc[:, 1:]
        y = year.iloc[:, 0]
        return X, y, None
