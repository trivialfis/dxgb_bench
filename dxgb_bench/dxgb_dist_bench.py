# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import os
import pickle
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeAlias

import xgboost
from xgboost.tracker import RabitTracker

from .utils import (
    DFT_OUT,
    Opts,
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_rmm_param,
    machine_info,
    make_params_from_args,
    merge_opts,
    save_results,
    split_path,
)


def make_iter_qdms(
    it_train: xgboost.DataIter, it_valid: xgboost.DataIter | None, max_bin: int
) -> tuple[xgboost.DMatrix, list[tuple[xgboost.DMatrix, str]]]:
    with Timer("Train", "DMatrix-Train"):
        dargs = {
            "data": it_train,
            "max_bin": max_bin,
            "max_quantile_batches": 32,
        }

        Xy_train: xgboost.DMatrix = xgboost.QuantileDMatrix(**dargs)

    watches = [(Xy_train, "Train")]

    if it_valid is not None:
        with Timer("Train", "DMatrix-Valid"):
            dargs = {
                "data": it_valid,
                "max_bin": max_bin,
                "ref": Xy_train,
            }
            Xy_valid = xgboost.QuantileDMatrix(**dargs)
            watches.append((Xy_valid, "Valid"))
    return Xy_train, watches


def get_numa_node_id(ordinal: int) -> int:
    import cuda.bindings.runtime as cudart

    status, value = cudart.cudaDeviceGetAttribute(
        cudart.cudaDeviceAttr.cudaDevAttrHostNumaId, ordinal
    )
    if status != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(cudart.cudaGetErrorString(status))
    return value


def bench(
    tmpdir: str,
    n_rounds: int,
    opts: Opts,
    params: dict[str, Any],
    n_workers: int,
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
) -> tuple[xgboost.Booster, dict[str, Any]]:
    assert n_workers > 0

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    rabit_args = tracker.worker_args()

    device = opts.device
    machine = machine_info(device)

    if opts.on_the_fly:
        n_batches_per_worker = opts.n_batches // n_workers
        assert n_batches_per_worker > 1

    R: TypeAlias = tuple[xgboost.Booster, dict[str, Any], Opts]

    def launch(wparams: dict[str, Any], worker_id: int) -> R:
        path = os.path.join(tmpdir, f"worker-params-{worker_id}.pkl")
        with open(path, "wb") as wfd:
            pickle.dump(wparams, wfd)

        env = os.environ.copy()

        if device == "cuda":
            nodeid = get_numa_node_id(worker_id)
            cmd = [
                "numactl",
                "--strict",
                f"--membind={nodeid}",
                f"--cpunodebind={nodeid}",
                "_dxgb-dist-impl",
                "--params",
                path,
            ]

            ordinals = [w % n_workers for w in range(worker_id, worker_id + n_workers)]
            devices = ",".join(map(str, ordinals))
            env["CUDA_VISIBLE_DEVICES"] = devices
        else:
            cmd = [
                "_dxgb-dist-impl",
                "--params",
                path,
            ]

        subprocess.check_call(cmd, env=env)
        path = os.path.join(tmpdir, f"worker-results-{worker_id}.pkl")
        with open(path, "rb") as rfd:
            booster, results, opts = pickle.load(rfd)
        return booster, results, opts

    with Timer("Train", "Total"):
        with ThreadPoolExecutor(max_workers=n_workers) as exe:
            futures = []
            for worker_id in range(n_workers):
                wparams = {
                    "opts": opts,
                    "n_rounds": n_rounds,
                    "rabit_args": rabit_args,
                    "params": params,
                    "loadfrom": loadfrom,
                    "verbosity": verbosity,
                    "worker_id": worker_id,
                    "is_extmem": is_extmem,
                }
                fut = exe.submit(launch, wparams, worker_id)
                futures.append(fut)

        train_results: list[R] = [f.result() for f in futures]
        assert len(train_results) == n_workers
        assert all(b[0].num_boosted_rounds() == n_rounds for b in train_results)

    boosters, w_results, w_opts = zip(*train_results)
    timers = [t["timer"] for t in w_results]
    evals = w_results[0]["evals"]

    n_total_batches = 0
    sparsity = 0.0
    n_features = 0
    n_samples_per_batch = 0
    for o in w_opts:
        n_total_batches += o.n_batches
        n_features = o.n_features
        n_samples_per_batch = o.n_samples_per_batch
        sparsity += o.sparsity
    sparsity /= n_workers

    client_timer = Timer.global_timer()

    # Merge the inferred opts
    opts.sparsity = sparsity
    if opts.on_the_fly:
        assert opts.n_batches == n_total_batches, (opts.n_batches, n_total_batches)
        assert n_features == opts.n_features, (n_features, opts.n_features)
        assert opts.n_samples_per_batch == n_samples_per_batch, (
            opts.n_samples_per_batch,
            n_samples_per_batch,
        )
    else:
        opts.n_batches = n_total_batches
        opts.n_features = n_features
        opts.n_samples_per_batch = n_samples_per_batch

    # Merge timers
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
    results = {
        "opts": opts_dict,
        "timer": max_timer,
        "evals": evals,
        "machine": machine,
    }
    save_results(results, "dist")
    return boosters[0], results


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument("--task", choices=["ext", "qdm"], required=True)
    parser.add_argument(
        "--valid", action="store_true", help="Split for the validation dataset."
    )
    parser.add_argument("--loadfrom", type=str, default=DFT_OUT)
    parser.add_argument("--cluster_type", choices=["local"], required=True)
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
    parser = add_data_params(parser, required=False)
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
    is_extmem = args.task == "ext"

    if args.cluster_type == "local":
        with tempfile.TemporaryDirectory() as tmpdir:
            bench(
                tmpdir,
                args.n_rounds,
                opts,
                params,
                args.n_workers,
                loadfrom,
                args.verbosity,
                is_extmem,
            )
    else:
        raise ValueError("Option removed.")


if __name__ == "__main__":
    cli_main()
