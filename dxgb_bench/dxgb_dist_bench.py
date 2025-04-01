# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import xgboost
from distributed import Client, LocalCluster, SSHCluster, get_worker, worker_client
from xgboost import collective as coll
from xgboost import dask as dxgb
from xgboost.callback import EvaluationMonitor
from xgboost.testing.dask import get_client_workers

from .dataiter import BenchIter, SynIterImpl
from .utils import (
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    make_params_from_args,
    setup_rmm,
)


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[dxgb-bench]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


class ForwardLoggingMonitor(EvaluationMonitor):
    def __init__(self, client: Client) -> None:
        client.forward_logging(_get_logger().name)

        super().__init__(logger=lambda msg: _get_logger().info(msg.strip()))


def train(
    args: argparse.Namespace,
    rabit_args: dict[str, Any],
    n_batches: int,
    rs: int,
    log_cb: EvaluationMonitor,
) -> xgboost.Booster:
    if args.device == "cuda":
        setup_rmm("arena")
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

    with worker_client(), coll.CommunicatorContext(**rabit_args):
        params = make_params_from_args(args)
        n_threads = dxgb.get_n_threads(params, worker)
        params.update({"nthread": n_threads})
        with xgboost.config_context(
            nthread=n_threads, use_rmm=True, verbosity=args.verbosity
        ):

            def log_fn(msg: str) -> None:
                if coll.get_rank() == 0:
                    _get_logger().info(msg)

            with Timer("Distributed", "ExtMemQdm", logger=log_fn):
                Xy = xgboost.ExtMemQuantileDMatrix(
                    it, max_quantile_batches=32, nthread=n_threads
                )

            with Timer("Distributed", "Train", logger=log_fn):
                booster = xgboost.train(
                    params,
                    Xy,
                    evals=[(Xy, "Train")],
                    num_boost_round=args.n_rounds,
                    verbose_eval=False,
                    callbacks=[log_cb],
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

    log_cb = ForwardLoggingMonitor(client)

    n_batches_per_worker = args.n_batches // n_workers
    assert n_batches_per_worker > 1

    with Timer("Distributed", "Total", logger=lambda msg: _get_logger().info(msg)):
        futures = []
        for worker_id, worker in enumerate(workers):
            n_batches_prev = worker_id * n_batches_per_worker
            rs = n_batches_prev * args.n_samples_per_batch * args.n_features
            n_batches = min(n_batches_per_worker, args.n_batches - n_batches_prev)
            fut = client.submit(train, args, rabit_args, n_batches, rs, log_cb)
            n_batches_prev += n_batches
            futures.append(fut)
        boosters = client.gather(futures)
        assert len(boosters) == n_workers
        assert all(b.num_boosted_rounds() == args.n_rounds for b in boosters)


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument(
        "--cluster_type", choices=["local", "ssh", "manual"], required=True
    )
    parser.add_argument("--n_workers", type=int)
    parser.add_argument(
        "--hosts", type=str, help=";separated list of hosts.", required=False
    )
    parser.add_argument("--rpy", type=str, help="remote python path.", required=False)
    parser.add_argument("--username", type=str, help="SSH username.", required=False)
    parser.add_argument(
        "--sched", type=str, help="path the to schedule config.", required=False
    )

    parser = add_device_param(parser)
    parser = add_hyper_param(parser)
    parser = add_data_params(parser, required=True)
    parser.add_argument("--verbosity", choices=[0, 1, 2, 3], default=1, type=int)
    args = parser.parse_args()

    if args.cluster_type == "local":
        if args.device == "cpu":
            with LocalCluster(n_workers=args.n_workers) as cluster:
                with Client(cluster) as client:
                    bench(client, args)
        else:
            from dask_cuda import LocalCUDACluster

            n_threads = os.cpu_count()
            assert n_threads is not None
            n_threads = max(n_threads // args.n_workers, 1)
            with LocalCUDACluster(
                n_workers=args.n_workers, threads_per_worker=n_threads
            ) as cluster:
                with Client(cluster) as client:
                    bench(client, args)
    elif args.cluster_type == "manual":
        sched = args.sched
        assert sched is not None
        with Client(scheduler_file=sched) as client:
            bench(client, args)
    else:
        hosts = args.hosts.split(";")
        print(hosts)
        rpy = args.rpy
        username = args.username
        assert rpy is not None
        assert username is not None
        with SSHCluster(
            hosts=hosts,
            remote_python=rpy,
            connect_options={"username": username, "known_hosts": None},
        ) as cluster:
            cluster.wait_for_workers(n_workers=len(hosts) - 1)
            with Client(cluster) as client:
                bench(client, args)
            logs = cluster.get_logs()
            for k, v in logs.items():
                print(f"{k}\n{v}")


if __name__ == "__main__":
    cli_main()
