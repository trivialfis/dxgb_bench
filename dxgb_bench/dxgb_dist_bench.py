from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np
import xgboost
from distributed import Client, LocalCluster, SSHCluster, get_worker, worker_client
from xgboost import collective as coll
from xgboost import dask as dxgb
from xgboost.callback import EvaluationMonitor
from xgboost.testing.dask import get_client_workers

from .dataiter import BenchIter, SynIterImpl
from .datasets.generated import make_dense_regression, save_Xy
from .utils import (
    DFT_OUT,
    add_data_params,
    add_device_param,
    add_hyper_param,
    make_params_from_args,
)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[dxgb-dist-bench]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


def _write_to_first(dirname: str, msg: str) -> None:
    path = os.path.join(dirname, "xgboost.log")
    with open(path, "a") as fd:
        print(msg.strip(), file=fd)


class ForwardLoggingMonitor(EvaluationMonitor):
    def __init__(self, dirname: str) -> None:
        self.dirname = dirname
        super().__init__(logger=lambda msg: _write_to_first(self.dirname, msg))


def make_batches(
    device: str,
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    worker_id: int,
    saveto: str,
    rs: int,
) -> None:
    for i in range(n_batches):
        X, y = make_dense_regression(
            device=device,
            n_samples=n_samples_per_batch,
            n_features=n_features,
            sparsity=0.0,
            random_state=rs,
        )
        rs += X.size
        save_Xy(X, y, i, [saveto])


def datagen(client: Client, args: argparse.Namespace) -> None:
    workers = get_client_workers(client)
    n_workers = len(workers)
    n_batches_per_worker = args.n_batches // n_workers
    assert n_batches_per_worker > 1
    futures = []
    for worker_id, worker in enumerate(workers):
        n_batches_prev = worker_id * n_batches_per_worker
        n_batches = min(n_batches_per_worker, args.n_batches - n_batches_prev)
        fut = client.submit(
            make_batches,
            n_samples_per_batch=args.n_samples_per_batch,
            n_features=args.n_features,
            n_batches=n_batches,
            worker_id=worker_id,
            saveto=args.saveto,
            workers=[worker],
        )
        n_batches_prev += n_batches
        futures.append(fut)
    client.gather(futures)


def train(
    args: argparse.Namespace, rabit_args: dict[str, Any], n_batches: int, rs: int
) -> xgboost.Booster:
    it_impl = SynIterImpl(
        n_samples_per_batch=args.n_samples_per_batch,
        n_features=args.n_features,
        n_batches=n_batches,
        sparsity=args.sparsity,
        assparse=args.assparse,
        device=args.device,
        rs=rs,
    )
    it = BenchIter(
        it_impl,
        is_ext=True,
        is_valid=False,
        device=args.device,
    )
    worker = get_worker()

    with worker_client() as client:
        with coll.CommunicatorContext(**rabit_args):
            params = make_params_from_args(args)
            n_threads = dxgb.get_n_threads(params, worker)
            params.update({"nthread": n_threads, "n_jobs": n_threads})
            with xgboost.config_context(nthread=n_threads, use_rmm=True):
                Xy = xgboost.ExtMemQuantileDMatrix(
                    it, max_quantile_batches=32, nthread=n_threads
                )
                booster = xgboost.train(
                    params,
                    Xy,
                    evals=[(Xy, "Train")],
                    num_boost_round=args.n_rounds,
                    callbacks=[ForwardLoggingMonitor()],
                )
    return booster


def bench(client: Client, args: argparse.Namespace) -> None:
    workers = get_client_workers(client)
    print(f"workers: {workers}")
    n_workers = len(workers)
    assert n_workers > 0
    rabit_args = client.sync(dxgb._get_rabit_args, client, n_workers)
    if not args.fly:
        raise NotImplementedError("--fly must be specified.")

    n_batches_per_worker = args.n_batches // n_workers
    assert n_batches_per_worker > 1
    futures = []
    for worker_id, worker in enumerate(workers):
        n_batches_prev = worker_id * n_batches_per_worker
        rs = n_batches_prev * args.n_samples_per_batch * args.n_features
        n_batches = min(n_batches_per_worker, args.n_batches - n_batches_prev)
        fut = client.submit(train, args, rabit_args, n_batches, rs)
        n_batches_prev += n_batches
        futures.append(fut)
    client.gather(futures)


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument("--cluster-type", choices=["local", "ssh"], required=True)
    parser.add_argument("--n_workers", type=int)
    parser.add_argument(
        "--hosts", type=str, help=";separated list of hosts.", required=False
    )
    parser.add_argument("--rpy", type=str, help="remote python path.", required=False)
    parser.add_argument("--username", type=str, help="SSH username.", required=False)

    parser = add_device_param(parser)
    parser = add_hyper_param(parser)
    parser = add_data_params(parser, required=True)
    args = parser.parse_args()

    if args.cluster_type == "local":
        with LocalCluster(n_workers=args.n_workers) as cluster:
            with Client(cluster) as client:
                bench(client, args)
    else:
        hosts = args.hosts.split(";")
        print(hosts)
        rpy = args.rpy
        username = args.username
        assert rpy is not None
        assert username is not None
        with SSHCluster(
            hosts=hosts, remote_python=rpy, connect_options={"username": username}
        ) as cluster:
            cluster.wait_for_workers(n_workers=len(hosts) - 1)
            with Client(cluster) as client:
                bench(client, args)


if __name__ == "__main__":
    cli_main()
