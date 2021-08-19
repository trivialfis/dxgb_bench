from dxgb_bench.utils import DataSet, fprint, read_csv, DType
import argparse
import os
import gzip


class Higgs(DataSet):
    def __init__(self, args: argparse.Namespace) -> None:
        self.uri = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"

        local_directory = os.path.join(args.local_directory, "HIGGS")
        self.retrieve(local_directory)

        self.csv_file = os.path.join(local_directory, "HIGGS.csv")
        extracted = os.path.exists(os.path.join(local_directory, "HIGGS.csv"))
        if not extracted:
            filename = os.path.join(local_directory, "HIGGS.csv.gz")
            assert os.path.exists(filename)
            with gzip.open(filename, "r") as z:
                fprint("Extracting", filename)
                data = z.read()
            with open(self.csv_file, "wb") as fd:
                fd.write(data)

        assert os.path.exists(self.csv_file)
        self.task = "binary:logistic"

    def load(self, args: argparse.Namespace) -> DType:
        colnames = ["label"] + ["feat-%02d" % i for i in range(1, 29)]
        df = read_csv(
            self.csv_file,
            header=None,
            names=colnames,
            backend=args.backend,
            dtype=None,
            sep=",",
        )
        y = df["label"]
        X = df[df.columns.difference(["label"])]
        return X, y, None
