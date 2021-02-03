from dask import array as da
from dask import dataframe as dd
from dxgb_bench.utils import DataSet


def generate_random_array(rows, cols):
    X = da.random.random(size=(rows, cols))
    y = da.random.random(size=(rows,))
    return X, y


def generate_random_df(rows, cols):
    X, y = generate_random_array(rows, cols)
    X = dd.from_dask_array(X)
    y = dd.from_dask_array(y)
    # X.map_partitions
    return X, y


class Artificial(DataSet):
    def __init__(self, args):
        X, y = generate_random_df(args.rows, args.cols)

    def load(self):
        raise NotImplementedError()
