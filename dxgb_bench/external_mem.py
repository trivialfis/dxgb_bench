"""Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import xgboost as xgb

from .dataiter import (
    TEST_SIZE,
    BenchIter,
    IterImpl,
    LoadIterStrip,
    SynIterImpl,
)
from .utils import Timer, setup_rmm


@dataclass
class Opts:
    n_samples_per_batch: int
    n_features: int
    n_batches: int
    sparsity: float
    on_the_fly: bool
    validation: bool
    device: str
    mr: str
    target_type: str


def make_iter(opts: Opts, loadfrom: list[str]) -> tuple[BenchIter, BenchIter | None]:
    if not opts.on_the_fly:
        # Load files
        if opts.validation:
            it_impl: IterImpl = LoadIterStrip(
                loadfrom, is_valid=False, test_size=TEST_SIZE, device=opts.device
            )
            train_it = BenchIter(
                it_impl,
                is_ext=True,
                is_valid=False,
                device=opts.device,
            )
            it_impl = LoadIterStrip(
                loadfrom, is_valid=True, test_size=TEST_SIZE, device=opts.device
            )
            valid_it = BenchIter(
                it_impl,
                is_ext=True,
                is_valid=True,
                device=opts.device,
            )
        else:
            it_impl = LoadIterStrip(
                loadfrom, is_valid=False, test_size=None, device=opts.device
            )
            train_it = BenchIter(
                it_impl,
                is_ext=True,
                is_valid=False,
                device=opts.device,
            )
            valid_it = None
        return train_it, valid_it

    # Generate data on the fly.
    if not opts.validation:
        it_impl = SynIterImpl(
            n_samples_per_batch=opts.n_samples_per_batch,
            n_features=opts.n_features,
            n_batches=opts.n_batches,
            sparsity=opts.sparsity,
            assparse=False,
            target_type=opts.target_type,
            device=opts.device,
        )
        it_train = BenchIter(
            it_impl,
            is_ext=True,
            is_valid=False,
            device=opts.device,
        )
        return it_train, None

    # Synthesize only the needed data for benchmarking purposes. We assume in the real
    # world users have prepared the data in ETL through specialized frameworks instead
    # of injecting complex logic in the iterator.
    n_train_samples = int(opts.n_samples_per_batch * (1.0 - TEST_SIZE))
    n_valid_samples = opts.n_samples_per_batch - n_train_samples
    it_train_impl = SynIterImpl(
        n_samples_per_batch=n_train_samples,
        n_features=opts.n_features,
        n_batches=opts.n_batches,
        sparsity=opts.sparsity,
        assparse=False,
        target_type=opts.target_type,
        device=opts.device,
    )
    it_valid_impl = SynIterImpl(
        n_samples_per_batch=n_valid_samples,
        n_features=opts.n_features,
        n_batches=opts.n_batches,
        sparsity=opts.sparsity,
        assparse=False,
        target_type=opts.target_type,
        device=opts.device,
    )
    it_train = BenchIter(it_train_impl, is_ext=True, is_valid=False, device=opts.device)
    it_valid = BenchIter(it_valid_impl, is_ext=True, is_valid=True, device=opts.device)
    return it_train, it_valid


def extmem_spdm_train(
    opts: Opts,
    params: dict[str, Any],
    n_rounds: int,
    loadfrom: list[str],
) -> xgb.Booster:
    if opts.device == "cuda":
        setup_rmm(opts.mr)

    it_train, it_valid = make_iter(opts, loadfrom=loadfrom)
    with Timer("ExtQdm", "DMatrix-Train"):
        Xy_train = xgb.DMatrix(it_train)

    watches = [(Xy_train, "Train")]

    if it_valid is not None:
        Xy_valid = xgb.DMatrix(it_valid)
        watches.append((Xy_valid, "Valid"))

    with Timer("ExtSparse", "train"):
        booster = xgb.train(
            params,
            Xy_train,
            num_boost_round=n_rounds,
            evals=watches,
            verbose_eval=True,
        )
    return booster


def extmem_qdm_train(
    opts: Opts,
    params: dict[str, Any],
    n_rounds: int,
    loadfrom: list[str],
) -> xgb.Booster:
    if opts.device == "cuda":
        setup_rmm(opts.mr)

    it_train, it_valid = make_iter(opts, loadfrom=loadfrom)
    with Timer("ExtQdm", "DMatrix-Train"):
        Xy_train = xgb.ExtMemQuantileDMatrix(
            it_train, max_bin=params["max_bin"], max_quantile_batches=32
        )

    watches = [(Xy_train, "Train")]

    if it_valid is not None:
        with Timer("ExtQdm", "DMatrix-Valid"):
            Xy_valid = xgb.ExtMemQuantileDMatrix(it_valid, ref=Xy_train)
            watches.append((Xy_valid, "Valid"))

    with Timer("ExtQdm", "train"):
        booster = xgb.train(
            params,
            Xy_train,
            num_boost_round=n_rounds,
            evals=watches,
            verbose_eval=True,
        )
    return booster
