# Copyright (c) 2024-2026, Jiaming Yuan.  All rights reserved.
"""Distributed XGBoost benchmarking with pyhwloc for NUMA binding."""
from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import sys
import traceback
from functools import partial, update_wrapper, wraps
from typing import Any, Callable, ParamSpec, TypeAlias, TypeVar

import xgboost
from xgboost import collective as coll
from xgboost.tracker import RabitTracker

from .dataiter import IterImpl, LoadIterStrip, StridedIter, SynIterImpl
from .external_mem import make_extmem_qdms
from .utils import (
    DFT_OUT,
    TEST_SIZE,
    Opts,
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_rmm_param,
    fill_opts_shape,
    fprint,
    has_async_pool,
    machine_info,
    make_params_from_args,
    merge_opts,
    need_rmm,
    save_results,
    setup_rmm,
    split_path,
)

R = TypeVar("R")
P = ParamSpec("P")


def _try_run(fn: Callable[P, R]) -> Callable[P, R]:
    """Loky aborts the process without printing out any error message if there's an
    exception.

    """

    @wraps(fn)
    def inner(*args: P.args, **kwargs: P.kwargs) -> R:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            print(traceback.format_exc(), file=sys.stderr)
            raise RuntimeError("Running into exception in worker.") from e

    return inner


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[dxgb-bench]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


def _setup_pyhwloc_binding(worker_id: int, n_workers: int) -> None:
    """Set up CPU and memory binding using pyhwloc.

    This function should be called in a child process to configure NUMA bindings for the
    given worker.
    """
    import pyhwloc
    from pyhwloc.cuda_runtime import get_device
    from pyhwloc.topology import MemBindFlags, MemBindPolicy, TypeFilter

    with pyhwloc.from_this_system().set_io_types_filter(TypeFilter.KEEP_ALL) as topo:
        # Get CPU affinity for this GPU
        dev = get_device(topo, worker_id)
        cpuset = dev.get_affinity()

        print("Idx:", worker_id, "\nCPUSet:", cpuset)
        # Set CPU binding
        topo.set_cpubind(cpuset)
        # Set memory binding using cpuset (hwloc determines NUMA nodes from cpuset)
        topo.set_membind(cpuset, MemBindPolicy.BIND, MemBindFlags.STRICT)


def _loky_initializer(n_workers: int, mr: str | None, device: str) -> None:
    """Initializer for loky worker processes.

    This function runs in each child process before any tasks are executed.
    """
    # Get the worker index from loky process name (e.g., "LokyProcess-1")
    proc_name = mp.current_process().name
    if "-" in proc_name:
        _, sidx = proc_name.rsplit("-", 1)
        idx = int(sidx) - 1  # loky uses 1-based indexing
    else:
        idx = 0

    if device == "cuda":
        # Set CUDA_VISIBLE_DEVICES - rotate GPUs so each worker sees its GPU first
        ordinals = [w % n_workers for w in range(idx, idx + n_workers)]
        devices = ",".join(map(str, ordinals))
        os.environ["CUDA_VISIBLE_DEVICES"] = devices

        # Set up NUMA bindings using pyhwloc
        _setup_pyhwloc_binding(idx, n_workers)

        # Set up RMM if requested
        if mr is not None:
            setup_rmm(
                mr, worker_id=0
            )  # worker_id=0 because CUDA_VISIBLE_DEVICES is set


# ------------------------------------------------------------------------------
# Worker training implementation
# ------------------------------------------------------------------------------


def _make_iter(
    opts: Opts, loadfrom: list[str], is_extmem: bool
) -> tuple[StridedIter, StridedIter | None]:
    """Create data iterators for training and validation."""
    it_train_impl: IterImpl
    it_valid_impl: IterImpl | None

    if opts.on_the_fly:
        if opts.validation:
            n_train_samples = int(opts.n_samples_per_batch * (1.0 - TEST_SIZE))
            n_valid_samples = opts.n_samples_per_batch - n_train_samples
            it_train_impl = SynIterImpl(
                n_samples_per_batch=n_train_samples,
                n_features=opts.n_features,
                n_targets=opts.n_targets,
                n_batches=opts.n_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
            )
            it_valid_impl = SynIterImpl(
                n_samples_per_batch=n_valid_samples,
                n_features=opts.n_features,
                n_targets=opts.n_targets,
                n_batches=opts.n_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
            )
        else:
            it_train_impl = SynIterImpl(
                n_samples_per_batch=opts.n_samples_per_batch,
                n_features=opts.n_features,
                n_targets=opts.n_targets,
                n_batches=opts.n_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
            )
            it_valid_impl = None
    else:
        if opts.validation:
            it_train_impl = LoadIterStrip(loadfrom, False, TEST_SIZE, opts.device)
            it_valid_impl = LoadIterStrip(loadfrom, True, TEST_SIZE, opts.device)
        else:
            it_train_impl = LoadIterStrip(loadfrom, False, None, opts.device)
            it_valid_impl = None

        assert it_train_impl.n_batches % coll.get_world_size() == 0

    it_train = StridedIter(
        it_train_impl,
        start=coll.get_rank(),
        stride=coll.get_world_size(),
        is_ext=is_extmem,
        is_valid=False,
        device=opts.device,
    )
    if it_valid_impl is not None:
        it_valid = StridedIter(
            it_valid_impl,
            start=coll.get_rank(),
            stride=coll.get_world_size(),
            is_ext=is_extmem,
            is_valid=True,
            device=opts.device,
        )
    else:
        it_valid = None

    return it_train, it_valid


def _make_iter_qdms(
    it_train: xgboost.DataIter, it_valid: xgboost.DataIter | None, max_bin: int
) -> tuple[xgboost.DMatrix, list[tuple[xgboost.DMatrix, str]]]:
    """Create QuantileDMatrix for training and validation."""
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


@_try_run
def _train(
    _: int,  # worker index from pool.map, unused (setup done in initializer)
    opts: Opts,
    n_rounds: int,
    rabit_args: dict[str, Any],
    params: dict[str, Any],
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
) -> tuple[xgboost.Booster, dict[str, Any], Opts]:
    """Execute XGBoost training in a worker process.

    RMM and NUMA bindings are set up in ``_loky_initializer``.
    """
    devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    results: dict[str, Any] = {}

    def log_fn(msg: str) -> None:
        if coll.get_rank() == 0:
            _get_logger().info(msg)

    with coll.CommunicatorContext(**rabit_args):
        affos = os.sched_getaffinity(0)
        fprint(
            f"[dxgb-bench] {coll.get_rank()}: CPU Affinity: {affos}",
            f" devices: {devices}",
        )
        with xgboost.config_context(
            use_rmm=need_rmm(opts.mr),
            verbosity=verbosity,
            use_cuda_async_pool=opts.mr == "cuda" if has_async_pool() else None,
        ):
            it_train, it_valid = _make_iter(opts, loadfrom, is_extmem)
            if is_extmem:
                Xy_train, watches = make_extmem_qdms(
                    opts, params["max_bin"], it_train, it_valid
                )
            else:
                Xy_train, watches = _make_iter_qdms(
                    it_train, it_valid, params["max_bin"]
                )

            evals_result: dict[str, dict[str, float]] = {}
            with Timer("Train", "Train", logger=log_fn):
                booster = xgboost.train(
                    params,
                    Xy_train,
                    evals=watches,
                    num_boost_round=n_rounds,
                    verbose_eval=True,
                    evals_result=evals_result,
                )
    if len(watches) >= 2:
        opts = fill_opts_shape(opts, Xy_train, watches[1][0], it_train.n_batches)
    else:
        opts = fill_opts_shape(opts, Xy_train, None, it_train.n_batches)
    results["timer"] = Timer.global_timer()
    results["evals"] = evals_result
    return booster, results, opts


# ------------------------------------------------------------------------------
# Process orchestration
# ------------------------------------------------------------------------------


def _run_workers(
    n_rounds: int,
    opts: Opts,
    params: dict[str, Any],
    n_workers: int,
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
    rabit_args: dict[str, Any],
) -> list[tuple[xgboost.Booster, dict[str, Any], Opts]]:
    """Run distributed training using loky process pool with pyhwloc for NUMA binding."""
    from loky import get_reusable_executor

    with get_reusable_executor(
        max_workers=n_workers,
        initargs=(n_workers, opts.mr, opts.device),
        initializer=_loky_initializer,
    ) as pool:
        fn = update_wrapper(
            partial(
                _train,
                opts=opts,
                n_rounds=n_rounds,
                rabit_args=rabit_args,
                params=params,
                loadfrom=loadfrom,
                verbosity=verbosity,
                is_extmem=is_extmem,
            ),
            _train,
        )
        results = list(pool.map(fn, range(n_workers)))
    return results


def bench(
    n_rounds: int,
    opts: Opts,
    params: dict[str, Any],
    n_workers: int,
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
) -> tuple[xgboost.Booster, dict[str, Any]]:
    """Run distributed XGBoost benchmark."""
    assert n_workers > 0

    tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
    tracker.start()
    rabit_args = tracker.worker_args()

    device = opts.device
    machine = machine_info(device)

    if opts.on_the_fly:
        n_batches_per_worker = opts.n_batches // n_workers
        assert n_batches_per_worker > 1

    TrainResult: TypeAlias = tuple[xgboost.Booster, dict[str, Any], Opts]

    with Timer("Train", "Total"):
        train_results: list[TrainResult] = _run_workers(
            n_rounds=n_rounds,
            opts=opts,
            params=params,
            n_workers=n_workers,
            loadfrom=loadfrom,
            verbosity=verbosity,
            is_extmem=is_extmem,
            rabit_args=rabit_args,
        )

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


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


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
        n_targets=args.n_targets,
        n_batches=args.n_batches,
        sparsity=args.sparsity,
        on_the_fly=args.fly,
        validation=args.valid,
        device=args.device,
        mr=args.mr,
        target_type=args.target_type,
        cache_host_ratio=args.cache_host_ratio,
        min_cache_page_bytes=args.min_cache_page_bytes,
    )
    loadfrom = split_path(args.loadfrom)
    params = make_params_from_args(args)
    is_extmem = args.task == "ext"

    if args.cluster_type == "local":
        bench(
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
