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

from .dataiter import BenchIter, IterImpl, LoadIterStrip, StridedIter, SynIterImpl
from .utils import (
    DFT_OUT,
    TEST_SIZE,
    Opts,
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_rmm_param,
    fprint,
    has_chr,
    machine_info,
    make_params_from_args,
    merge_opts,
    save_results,
    setup_rmm,
    split_path,
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


def make_iter(
    opts: Opts, loadfrom: list[str], n_worker_batches: int, rs: int
) -> tuple[StridedIter, StridedIter | None]:
    if opts.on_the_fly:
        if opts.validation:
            n_train_samples = int(opts.n_samples_per_batch * (1.0 - TEST_SIZE))
            n_valid_samples = opts.n_samples_per_batch - n_train_samples
            it_train_impl: IterImpl = SynIterImpl(
                n_samples_per_batch=n_train_samples,
                n_features=opts.n_features,
                n_batches=n_worker_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
                rs=rs,
            )
            it_valid_impl: IterImpl | None = SynIterImpl(
                n_samples_per_batch=n_valid_samples,
                n_features=opts.n_features,
                n_batches=n_worker_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
                rs=rs + (n_train_samples * opts.n_features * n_worker_batches),
            )
        else:
            it_train_impl = SynIterImpl(
                n_samples_per_batch=opts.n_samples_per_batch,
                n_features=opts.n_features,
                n_batches=n_worker_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
                rs=rs,
            )
            it_valid_impl = None
    else:
        if opts.validation:
            it_train_impl = LoadIterStrip(loadfrom, False, TEST_SIZE, opts.device)
            it_valid_impl = LoadIterStrip(loadfrom, True, TEST_SIZE, opts.device)
        else:
            it_train_impl = LoadIterStrip(loadfrom, False, None, opts.device)
            it_valid_impl = None

    it_train = StridedIter(
        it_train_impl,
        start=coll.get_rank(),
        stride=coll.get_world_size(),
        is_ext=True,
        is_valid=False,
        device=opts.device,
    )
    if it_valid_impl is not None:
        it_valid = StridedIter(
            it_valid_impl,
            start=coll.get_rank(),
            stride=coll.get_world_size(),
            is_ext=True,
            is_valid=True,
            device=opts.device,
        )
    else:
        it_valid = None

    return it_train, it_valid


def train(
    opts: Opts,
    n_rounds: int,
    rabit_args: dict[str, Any],
    n_worker_batches: int,
    params: dict[str, Any],
    loadfrom: list[str],
    rs: int,
    log_cb: EvaluationMonitor,
    verbosity: int,
) -> tuple[xgboost.Booster, dict[str, Any]]:
    if opts.device == "cuda" and opts.mr is not None:
        setup_rmm(opts.mr)

    worker = get_worker()
    results: dict[str, Any] = {}

    def log_fn(msg: str) -> None:
        if coll.get_rank() == 0:
            _get_logger().info(msg)

    with worker_client(), coll.CommunicatorContext(**rabit_args):
        n_threads = dxgb.get_n_threads(params, worker)
        params.update({"nthread": n_threads})
        with xgboost.config_context(
            nthread=n_threads, use_rmm=True, verbosity=verbosity
        ):
            it_train, it_valid = make_iter(opts, loadfrom, n_worker_batches, rs)

            with Timer("Train", "DMatrix-Train", logger=log_fn):
                dargs = {
                    "data": it_train,
                    "max_bin": params["max_bin"],
                    "max_quantile_batches": 32,
                    "nthread": n_threads,
                }
                if has_chr():
                    dargs["cache_host_ratio"] = opts.cache_host_ratio
                Xy_train = xgboost.ExtMemQuantileDMatrix(**dargs)

            watches = [(Xy_train, "Train")]

            if it_valid is not None:
                with Timer("Train", "DMatrix-Valid"):
                    # cache_host_ratio is not used here, but set it anyway.
                    dargs = {
                        "data": it_valid,
                        "max_bin": params["max_bin"],
                        "ref": Xy_train,
                    }
                    if has_chr():
                        dargs["cache_host_ratio"] = opts.cache_host_ratio
                    Xy_valid = xgboost.ExtMemQuantileDMatrix(**dargs)
                    watches.append((Xy_valid, "Valid"))

            evals_result: dict[str, dict[str, float]] = {}
            with Timer("Train", "Train", logger=log_fn):
                booster = xgboost.train(
                    params,
                    Xy_train,
                    evals=watches,
                    num_boost_round=n_rounds,
                    verbose_eval=False,
                    callbacks=[log_cb],
                    evals_result=evals_result,
                )
    results["timer"] = Timer.global_timer()
    results["evals"] = evals_result
    return booster, results


def bench(
    client: Client,
    n_rounds: int,
    opts: Opts,
    params: dict[str, Any],
    loadfrom: list[str],
    verbosity: int,
) -> xgboost.Booster:
    workers = get_client_workers(client)
    fprint(f"Workers: {workers}")
    n_workers = len(workers)
    assert n_workers > 0
    rabit_args = client.sync(dxgb._get_rabit_args, client, n_workers)
    log_cb = ForwardLoggingMonitor(client)
    machine = machine_info(opts.device)

    n_batches_per_worker = opts.n_batches // n_workers
    assert n_batches_per_worker > 1

    with Timer("Train", "Total", logger=lambda msg: _get_logger().info(msg)):
        futures = []
        for worker_id, worker in enumerate(workers):
            n_batches_prev = worker_id * n_batches_per_worker
            rs = n_batches_prev * opts.n_samples_per_batch * opts.n_features
            n_batches = min(n_batches_per_worker, opts.n_batches - n_batches_prev)
            fut = client.submit(
                train,
                opts,
                n_rounds,
                rabit_args,
                n_batches,
                params,
                loadfrom,
                rs,
                log_cb,
                verbosity,
            )
            n_batches_prev += n_batches
            futures.append(fut)
        boosters: list[tuple[xgboost.Booster, dict[str, Any]]] = client.gather(futures)
        assert len(boosters) == n_workers
        assert all(b[0].num_boosted_rounds() == n_rounds for b in boosters)

    timers = [t["timer"] for _, t in boosters]
    client_timer = Timer.global_timer()

    max_timer: dict[str, dict[str, float]] = {}
    for timer in timers:
        for k, v in timer.items():
            assert isinstance(v, dict)
            if k not in max_timer:
                max_timer[k] = {}
            for k1, v1 in v.items():
                if k1 not in max_timer[k]:
                    max_timer[k][k1] = 0
                max_timer[k][k1] = max(max_timer[k][k1], v1)

    assert "Train" in max_timer
    max_timer["Train"]["Total"] = client_timer["Train"]["Total"]
    opts_dict = merge_opts(opts, params)
    opts_dict["n_rounds"] = n_rounds
    opts_dict["n_workers"] = n_workers
    results = {"opts": opts_dict, "timer": max_timer, "machine": machine}
    save_results(results)
    return boosters[0][0]


def local_cluster(device: str, n_workers: int, **kwargs: Any) -> LocalCluster:
    assert device in ["cpu", "cuda"]
    if device == "cpu":
        return LocalCluster(n_workers=n_workers, memory_limit=None, **kwargs)
    from dask_cuda import LocalCUDACluster

    n_threads = os.cpu_count()
    assert n_threads is not None
    n_threads = max(n_threads // n_workers, 1)
    return LocalCUDACluster(
        n_workers=n_workers,
        threads_per_worker=n_threads,
        memory_limit=None,
        **kwargs,
    )


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument(
        "--valid", action="store_true", help="Split for the validation dataset."
    )
    parser.add_argument("--loadfrom", type=str, default=DFT_OUT)
    parser.add_argument(
        "--cluster_type", choices=["local", "ssh", "sched"], required=True
    )
    parser.add_argument("--n_workers", type=int, required=True)
    parser.add_argument(
        "--hosts", type=str, help=";separated list of hosts.", required=False
    )
    parser.add_argument("--rpy", type=str, help="remote python path.", required=False)
    parser.add_argument("--username", type=str, help="SSH username.", required=False)
    parser.add_argument(
        "--sched", type=str, help="path the to schedule config.", required=False
    )

    parser = add_device_param(parser)
    parser = add_rmm_param(parser)
    parser = add_hyper_param(parser)
    parser = add_data_params(parser, required=True)
    args = parser.parse_args()

    opts = Opts(
        n_samples_per_batch=args.n_samples_per_batch,
        n_features=args.n_features,
        n_batches=args.n_batches,
        sparsity=args.sparsity,
        on_the_fly=args.fly,
        validation=args.valid,
        device=args.device,
        mr=args.mr,
        target_type=args.target_type,
        cache_host_ratio=args.cache_host_ratio,
    )
    loadfrom = split_path(args.loadfrom)
    params = make_params_from_args(args)

    if args.cluster_type == "local":
        with local_cluster(device=args.device, n_workers=args.n_workers) as cluster:
            with Client(cluster) as client:
                bench(client, args.n_rounds, opts, params, loadfrom, args.verbosity)
    elif args.cluster_type == "sched":
        sched = args.sched
        assert sched is not None
        with Client(scheduler_file=sched) as client:
            bench(client, args.n_rounds, opts, params, loadfrom, args.verbosity)
    else:
        hosts = args.hosts.split(";")
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
                bench(client, args.n_rounds, opts, params, loadfrom, args.verbosity)
            logs = cluster.get_logs()
            for k, v in logs.items():
                fprint(f"{k}\n{v}")


if __name__ == "__main__":
    cli_main()
