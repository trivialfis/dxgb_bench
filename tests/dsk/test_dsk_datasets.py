# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
from typing import Generator

import pytest
from dask_cuda import LocalCUDACluster
from distributed import Client, LocalCluster

from dxgb_bench.dsk import make_dense_regression


@pytest.fixture(scope="module")
def cpu_cluster() -> Generator[LocalCluster, None, None]:
    with LocalCluster() as cluster:
        yield cluster


@pytest.fixture(scope="module")
def cuda_cluster() -> Generator[LocalCUDACluster, None, None]:
    with LocalCUDACluster() as cluster:
        yield cluster


def test_make_dense_regression_cpu(cpu_cluster: LocalCluster) -> None:
    with Client(cpu_cluster) as client:
        X, y = make_dense_regression("cpu", 4096, 128, random_state=1994)
        assert (
            client.compute(X.shape[0]).result()
            == 4096
            == client.compute(y.shape[0]).result()
        )
        assert X.shape[1] == 128


def test_make_dense_regression_cuda(cuda_cluster: LocalCluster) -> None:
    with Client(cuda_cluster) as client:
        X, y = make_dense_regression("cuda", 4096, 128, random_state=1994)
        assert (
            client.compute(X.shape[0]).result()
            == 4096
            == client.compute(y.shape[0]).result()
        )
        assert X.shape[1] == 128
