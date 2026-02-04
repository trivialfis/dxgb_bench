# Copyright (c) 2024-2026, Jiaming Yuan.  All rights reserved.
"""Distributed XGBoost benchmarking with pyhwloc for NUMA binding."""

from __future__ import annotations

import argparse
import logging
import os
import pickle
import subprocess
import sys
import tempfile
import traceback
from functools import wraps
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
    """Wrapper to print exceptions in subprocess workers before re-raising."""

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


def setup_pyhwloc_binding(worker_id: int, n_workers: int) -> None:
    """Set up CPU and memory binding using pyhwloc.

    This function should be called in a child process to configure NUMA bindings for the
    given worker.
    """
    import pyhwloc

    # from pyhwloc.cuda_runtime import get_device
    from pyhwloc.cuda_runtime import get_device
    from pyhwloc.topology import MemBindFlags, MemBindPolicy, TypeFilter

    with pyhwloc.from_this_system().set_io_types_filter(TypeFilter.KEEP_ALL) as topo:
        # Get CPU affinity for this GPU
        dev = get_device(topo, worker_id)
        cpuset = dev.get_affinity()

        devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
        print(
            "Idx:",
            worker_id,
            "\nCPUSet:",
            cpuset,
            "\nDevices:",
            devices,
            "Worker ID:",
            worker_id,
        )
        # Set CPU binding
        topo.set_cpubind(cpuset)
        # Set memory binding using cpuset (hwloc determines NUMA nodes from cpuset)
        topo.set_membind(cpuset, MemBindPolicy.BIND, MemBindFlags.STRICT)


def _worker_init(worker_id: int, n_workers: int, mr: str | None, device: str) -> None:
    """Initialize a subprocess worker with NUMA bindings and RMM."""
    if device != "cuda":
        return

    # Set up NUMA bindings using pyhwloc
    setup_pyhwloc_binding(worker_id, n_workers)

    # Set up RMM if requested
    if mr is not None:
        setup_rmm(mr, worker_id=worker_id)


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
    worker_id: int,
    n_workers: int,
    opts: Opts,
    n_rounds: int,
    rabit_args: dict[str, Any],
    params: dict[str, Any],
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
) -> tuple[xgboost.Booster, dict[str, Any], Opts]:
    """Execute XGBoost training in a worker process.

    RMM and NUMA bindings are set up in ``_worker_init``.
    """
    # Initialize worker (NUMA bindings, RMM)
    _worker_init(worker_id, n_workers, opts.mr, opts.device)
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
# Subprocess Pool Abstraction
# ------------------------------------------------------------------------------


def _get_cuda_visible_devices(worker_id: int, n_workers: int) -> str:
    """Get CUDA_VISIBLE_DEVICES string for a worker.

    Rotate GPUs so each worker sees its GPU first.
    """
    ordinals = [w % n_workers for w in range(worker_id, worker_id + n_workers)]
    return ",".join(map(str, ordinals))


class SubprocessPool:
    """A process pool that spawns workers as subprocesses.

    This pool provides a map-like interface for running functions in separate
    subprocesses. Each worker gets its own CUDA_VISIBLE_DEVICES environment
    variable set appropriately for GPU isolation.

    Parameters
    ----------
    n_workers : int
        Number of worker processes to spawn.
    device : str
        Device type ("cuda" or "cpu"). If "cuda", CUDA_VISIBLE_DEVICES will be
        set for each worker.

    Examples
    --------
    >>> def compute(worker_id: int, n_workers: int, value: int) -> int:
    ...     # worker_id and n_workers are automatically injected
    ...     return value * 2
    ...
    >>> with SubprocessPool(n_workers=4, device="cuda") as pool:
    ...     args_list = [{"value": i} for i in range(4)]
    ...     results = pool.map(compute, args_list)
    >>> results
    [0, 2, 4, 6]
    """

    def __init__(self, n_workers: int, device: str = "cuda") -> None:
        self.n_workers = n_workers
        self.device = device
        self._tmpdir: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> "SubprocessPool":
        self._tmpdir = tempfile.TemporaryDirectory()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()
            self._tmpdir = None

    def _get_tmpdir(self) -> str:
        if self._tmpdir is None:
            raise RuntimeError("SubprocessPool must be used as a context manager")
        return self._tmpdir.name

    def map(
        self,
        fn: Callable[..., R],
        args_list: list[dict[str, Any]],
    ) -> list[R]:
        """Apply function to each set of arguments in separate subprocesses.

        Parameters
        ----------
        fn : Callable
            The function to execute. Must be picklable (e.g., a module-level
            function). The function will receive `worker_id` and `n_workers`
            as additional keyword arguments.
        args_list : list[dict]
            List of keyword argument dicts, one per worker. Length must equal
            n_workers.

        Returns
        -------
        list
            List of results from each worker, in order.

        Raises
        ------
        RuntimeError
            If any worker fails.
        ValueError
            If args_list length doesn't match n_workers.
        """
        if len(args_list) != self.n_workers:
            raise ValueError(
                f"args_list length ({len(args_list)}) must match "
                f"n_workers ({self.n_workers})"
            )

        tmpdir = self._get_tmpdir()

        # Serialize function and arguments for each worker
        task_paths: list[str] = []
        output_paths: list[str] = []

        for worker_id, args in enumerate(args_list):
            task_data = {
                "fn": fn,
                "args": args,
                "worker_id": worker_id,
                "n_workers": self.n_workers,
            }
            task_path = os.path.join(tmpdir, f"task_{worker_id}.pkl")
            output_path = os.path.join(tmpdir, f"result_{worker_id}.pkl")

            with open(task_path, "wb") as f:
                pickle.dump(task_data, f)

            task_paths.append(task_path)
            output_paths.append(output_path)

        # Spawn all worker subprocesses
        processes: list[subprocess.Popen[bytes]] = []

        for worker_id in range(self.n_workers):
            env = os.environ.copy()
            if self.device == "cuda":
                env["CUDA_VISIBLE_DEVICES"] = _get_cuda_visible_devices(
                    worker_id, self.n_workers
                )

            cmd = [
                sys.executable,
                "-m",
                "dxgb_bench.dxgb_dist_bench",
                "worker",
                "--task-path",
                task_paths[worker_id],
                "--output-path",
                output_paths[worker_id],
            ]
            proc = subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            processes.append(proc)

        # Wait for all workers and collect results
        results: list[R] = []
        errors: list[str] = []

        for worker_id, proc in enumerate(processes):
            stdout, stderr = proc.communicate()
            if stdout:
                print(stdout.decode(), end="")
            if stderr:
                print(stderr.decode(), end="", file=sys.stderr)

            if proc.returncode != 0:
                errors.append(
                    f"Worker {worker_id} failed with return code {proc.returncode}"
                )
                continue

            output_path = output_paths[worker_id]
            if os.path.exists(output_path):
                with open(output_path, "rb") as f:
                    result_data = pickle.load(f)
                if "error" in result_data:
                    errors.append(f"Worker {worker_id}: {result_data['error']}")
                else:
                    results.append(result_data["result"])
            else:
                errors.append(f"Worker {worker_id} did not produce output file")

        if errors:
            raise RuntimeError("Worker errors:\n" + "\n".join(errors))

        return results


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
    """Run distributed training using SubprocessPool."""
    # Build the common arguments for all workers
    common_args = {
        "opts": opts,
        "n_rounds": n_rounds,
        "rabit_args": rabit_args,
        "params": params,
        "loadfrom": loadfrom,
        "verbosity": verbosity,
        "is_extmem": is_extmem,
    }

    # Each worker gets the same arguments (worker_id/n_workers added by pool)
    args_list = [common_args.copy() for _ in range(n_workers)]

    with SubprocessPool(n_workers=n_workers, device=opts.device) as pool:
        results = pool.map(_train, args_list)

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
# Subprocess worker entry point
# ------------------------------------------------------------------------------


def _worker_main(task_path: str, output_path: str) -> None:
    """Entry point for subprocess workers.

    Loads a pickled task (function + arguments), executes it, and writes the
    result to an output file. This enables the SubprocessPool to run arbitrary
    functions in separate processes.

    Parameters
    ----------
    task_path : str
        Path to pickled task file containing:
        - fn: The function to execute
        - args: Keyword arguments dict for the function
        - worker_id: The worker's index
        - n_workers: Total number of workers
    output_path : str
        Path to write the pickled result.
    """
    # Load the task
    with open(task_path, "rb") as f:
        task = pickle.load(f)

    fn = task["fn"]
    args = task["args"]
    worker_id = task["worker_id"]
    n_workers = task["n_workers"]

    # Execute the function with worker_id and n_workers injected
    try:
        result = fn(worker_id=worker_id, n_workers=n_workers, **args)
        output = {"result": result}
    except Exception as e:
        # Capture exception info for the parent process
        output = {"error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}

    # Write result to output file
    with open(output_path, "wb") as f:
        pickle.dump(output, f)


# ------------------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------------------


def _add_worker_subparser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Add the 'worker' subparser for subprocess workers (internal use)."""
    worker_parser = subparsers.add_parser(
        "worker",
        help="Run as a subprocess worker (internal use by SubprocessPool).",
    )
    worker_parser.add_argument(
        "--task-path",
        type=str,
        required=True,
        help="Path to pickled task file.",
    )
    worker_parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to write pickled output.",
    )


def _add_run_subparser(subparsers: argparse._SubParsersAction[Any]) -> None:
    """Add the 'run' subparser for the main benchmark client."""
    run_parser = subparsers.add_parser(
        "run",
        help="Run distributed XGBoost benchmark.",
    )
    run_parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    run_parser.add_argument(
        "--task",
        choices=["ext", "qdm"],
        required=True,
        help="Task type: 'ext' for external memory, 'qdm' for QuantileDMatrix.",
    )
    run_parser.add_argument(
        "--valid",
        action="store_true",
        help="Split for the validation dataset.",
    )
    run_parser.add_argument(
        "--loadfrom",
        type=str,
        default=DFT_OUT,
        help="Path to load data from.",
    )
    run_parser.add_argument(
        "--cluster_type",
        choices=["local"],
        required=True,
        help="Cluster type for distributed training.",
    )
    run_parser.add_argument(
        "--n_workers",
        type=int,
        required=True,
        help="Number of workers.",
    )
    run_parser.add_argument(
        "--hosts",
        type=str,
        help=";separated list of hosts.",
    )
    run_parser.add_argument(
        "--rpy",
        type=str,
        help="Remote python path.",
    )
    run_parser.add_argument(
        "--username",
        type=str,
        help="SSH username.",
    )
    run_parser.add_argument(
        "--sched",
        type=str,
        help="Path to the schedule config.",
    )

    add_device_param(run_parser)
    add_rmm_param(run_parser)
    add_hyper_param(run_parser)
    add_data_params(run_parser, required=False)


def _handle_worker(args: argparse.Namespace) -> None:
    """Handle the 'worker' subcommand."""
    _worker_main(args.task_path, args.output_path)


def _handle_run(args: argparse.Namespace) -> None:
    """Handle the 'run' subcommand."""
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


def cli_main() -> None:
    parser = argparse.ArgumentParser(
        prog="dxgb_dist_bench",
        description="Distributed XGBoost benchmarking with NUMA binding.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    _add_worker_subparser(subparsers)
    _add_run_subparser(subparsers)

    args = parser.parse_args()

    if args.command == "worker":
        _handle_worker(args)
    elif args.command == "run":
        _handle_run(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    cli_main()
