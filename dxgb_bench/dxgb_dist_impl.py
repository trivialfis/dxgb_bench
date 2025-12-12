# Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import argparse
import logging
import os
import pickle
from typing import Any

import xgboost
from xgboost import collective as coll

from .dataiter import IterImpl, LoadIterStrip, StridedIter, SynIterImpl
from .external_mem import make_extmem_qdms
from .utils import TEST_SIZE, Opts, Timer, fill_opts_shape, fprint, setup_rmm


def _get_logger() -> logging.Logger:
    logger = logging.getLogger("[dxgb-bench]")
    logger.setLevel(logging.INFO)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        logger.addHandler(handler)
    return logger


def make_iter(
    opts: Opts, loadfrom: list[str], is_extmem: bool
) -> tuple[StridedIter, StridedIter | None]:
    if opts.on_the_fly:
        if opts.validation:
            n_train_samples = int(opts.n_samples_per_batch * (1.0 - TEST_SIZE))
            n_valid_samples = opts.n_samples_per_batch - n_train_samples
            it_train_impl: IterImpl = SynIterImpl(
                n_samples_per_batch=n_train_samples,
                n_features=opts.n_features,
                n_targets=opts.n_targets,
                n_batches=opts.n_batches,
                sparsity=opts.sparsity,
                assparse=False,
                target_type=opts.target_type,
                device=opts.device,
            )
            it_valid_impl: IterImpl | None = SynIterImpl(
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


def train(
    opts: Opts,
    n_rounds: int,
    rabit_args: dict[str, Any],
    params: dict[str, Any],
    loadfrom: list[str],
    verbosity: int,
    is_extmem: bool,
    worker_id: int,
) -> tuple[xgboost.Booster, dict[str, Any], Opts]:
    if opts.mr is not None:
        setup_rmm(opts.mr, worker_id)

        devices = os.getenv("CUDA_VISIBLE_DEVICES", None)
    else:
        devices = None

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
        with xgboost.config_context(use_rmm=True, verbosity=verbosity):
            it_train, it_valid = make_iter(opts, loadfrom, is_extmem)
            if is_extmem:
                Xy_train, watches = make_extmem_qdms(
                    opts, params["max_bin"], it_train, it_valid
                )
            else:
                Xy_train, watches = make_iter_qdms(
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


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, help="Path to pickled parameters")
    args = parser.parse_args()
    path = args.params
    tmpdir = os.path.dirname(path)

    with open(path, "rb") as fd:
        params = pickle.load(fd)
    worker_id = params["worker_id"]
    booster, results, opts = train(**params)
    path = os.path.join(tmpdir, f"worker-results-{worker_id}.pkl")
    with open(path, "wb") as fdw:
        pickle.dump((booster, results, opts), fdw)


if __name__ == "__main__":
    cli_main()
