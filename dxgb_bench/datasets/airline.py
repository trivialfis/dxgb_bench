import argparse
import bz2
import os
from typing import Optional, Tuple

import numpy as np

from ..utils import DType, fprint, read_csv
from .dataset import DataSet


class Airline(DataSet):
    def __init__(self, args: argparse.Namespace):
        local_directory = os.path.join(args.local_directory, "airline14")
        self.uri = "https://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"
        self.retrieve(local_directory)

        self.extract_path = os.path.join(local_directory, "airline_14col.data")
        extracted = os.path.exists(self.extract_path)

        if not extracted:
            filename = os.path.join(local_directory, "airline_14col.data.bz2")
            assert os.path.exists(filename)
            with bz2.open(filename) as z:
                fprint("Extracting", filename)
                data = z.read()
            with open(self.extract_path, "wb") as fd:
                fd.write(data)

        self.task = "binary:logistic"

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        header = [
            "Year",
            "Month",
            "DayofMonth",
            "DayofWeek",
            "CRSDepTime",
            "CRSArrTime",
            "UniqueCarrier",
            "FlightNum",
            "ActualElapsedTime",
            "Origin",
            "Dest",
            "Distance",
            "Diverted",
            "ArrDelay",
        ]
        df = read_csv(
            self.extract_path,
            sep=",",
            backend=args.backend,
            dtype=np.float32,
            header=None,
            names=header,
        )

        df["ArrDelayBinary"] = 1 * (df["ArrDelay"] > 0)
        X = df[df.columns.difference(["ArrDelay", "ArrDelayBinary"])]
        y = df["ArrDelayBinary"]
        print(
            "Positive:", (y == 1).sum(), "Negative:", (y == 0).sum(), "Shape:", X.shape
        )
        return X, y, None
