from dask_bench.utils import DataSet
import os


class Epsilon(DataSet):
    def __init__(self, args):
        url_train = (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
            "/epsilon_normalized.bz2"
        )
        url_test = (
            "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary"
            "/epsilon_normalized.t.bz2"
        )
        local_directory = os.path.join(args.local_directory, "Epsilon")
        self.retrieve(local_directory)

    def load(self, args):
        pass
