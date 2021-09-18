from typing import Tuple, Optional
import argparse
import os
import pickle
from concurrent.futures import ThreadPoolExecutor

from dxgb_bench.utils import DataSet, DType, fprint
from scipy import sparse
import numpy as np
import cupy as cp
import cupyx


def make_regression(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray]:
    print(args.n_samples, args.n_features, args.sparsity)
    n_threads = args.cpus

    def random_csr(t_id: int) -> sparse.csc_matrix:
        rng = np.random.default_rng(1994 * t_id)
        thread_size = args.n_samples // n_threads
        if t_id == n_threads - 1:
            n_samples = args.n_samples - t_id * thread_size
        else:
            n_samples = thread_size

        X = sparse.random(
            m=n_samples, n=args.n_features, density=1.0 - args.sparsity, random_state=rng
        ).tocsc()
        return X

    futures = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        for i in range(n_threads):
            futures.append(executor.submit(random_csr, i))

    results = []
    for f in futures:
        results.append(f.result())

    X = sparse.vstack(results, format="csc")
    y = np.zeros((args.n_samples, 1))

    rng = np.random.RandomState(1994)
    for i in range(X.shape[1]):
        size = X.indptr[i+1] - X.indptr[i]
        print(size, X[:, i].toarray().shape)
        if size != 0:
            y += X[:, i].toarray() * rng.randn(args.n_samples, 1) * 0.2

    X = X.tocsr()

    return X, y


class Generated(DataSet):
    def __init__(self, args: argparse.Namespace) -> None:
        if args.backend.find("dask") != -1:
            raise NotImplementedError()
        if args.task is None:
            raise ValueError("`task` is required to generate dataset.")
        if args.n_samples is None:
            raise ValueError("`n_samples` is required to generate dataset.")
        if args.n_features is None:
            raise ValueError("`n_features` is required to generate dataset.")
        if args.sparsity is None:
            raise ValueError("`sparsity` is required to generate dataset.")

        if args.task != "reg":
            raise NotImplementedError()
        self.dirpath = os.path.join(
            args.local_directory, f"{args.n_samples}-{args.n_features}-{args.sparsity}"
        )
        if not os.path.exists(self.dirpath):
            os.mkdir(self.dirpath)

        self.X_path = os.path.join(self.dirpath, "X.pkl")
        self.y_path = os.path.join(self.dirpath, "y.pkl")

        self.task: str = "reg:squarederror"

        if os.path.exists(self.X_path) and os.path.exists(self.y_path):
            return

        X, y = make_regression(args)
        # X = X.get()
        # y = cp.asnumpy(y)
        with open(self.X_path, "wb") as fd:
            pickle.dump(X, fd)
        with open(self.y_path, "wb") as fd:
            pickle.dump(y, fd)

    def load(self, args: argparse.Namespace) -> Tuple[DType, DType, Optional[DType]]:
        with open(self.X_path, "rb") as fd:
            X = pickle.load(fd)
        with open(self.y_path, "rb") as fd:
            y = pickle.load(fd)

        return X, y, None
