from dask_bench.utils import DataSet
import os


class YearPrediction(DataSet):
    def __init__(self, args):
        self.uri = 'https://archive.ics.uci.edu/ml/machine-learning-' \
            'databases/00203/YearPredictionMSD.txt.zip'
        local_directory = os.path.join(args.local_directory, 'YearPrediction')
        self.retrieve(local_directory)

    def load(self, args):
        raise NotImplementedError()
