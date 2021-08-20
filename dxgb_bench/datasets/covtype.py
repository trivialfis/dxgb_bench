import argparse
import os
import gzip
from typing import Tuple, Optional, Dict, Any

import numpy as np
from dxgb_bench.utils import DataSet, fprint, read_csv, DType


class Covtype(DataSet):
    def __init__(self, args: argparse.Namespace):
        local_directory = os.path.join(args.local_directory, "covtype")
        self.uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        self.retrieve(local_directory)

        self.extract_path = os.path.join(local_directory, "covtype.data")
        extracted = os.path.exists(self.extract_path)

        if not extracted:
            filename = os.path.join(local_directory, "covtype.data.gz")
            assert os.path.exists(filename)
            with gzip.open(filename) as z:
                fprint("Extracting", filename)
                data = z.read()
            with open(self.extract_path, "wb") as fd:
                fd.write(data)

        self.task = "multi:softprob"

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        Xy = read_csv(
            self.extract_path,
            sep=",",
            backend=args.backend,
            header=None,
            names=None,
            dtype=np.float32,
        )
        X = Xy.iloc[:, :-1]
        y = Xy.iloc[:, -1].astype(np.int32)

        return X, y, None

    def extra_args(self) -> Dict[str, Any]:
        return {"num_class": 10}
