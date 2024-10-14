from __future__ import annotations

import argparse
import gc
import os
from typing import Callable, List, Protocol, Tuple
from dataclasses import dataclass

import cupy as cp
import numpy as np
import rmm
import xgboost as xgb
from rmm.allocators.cupy import rmm_cupy_allocator
from xgboost.callback import TrainingCheckPoint

from .dataiter import BenchIter, IterImpl, LoadIterImpl, SynIterImpl, get_file_paths
from .datasets.generated import make_dense_regression
from .utils import Progress, Timer


def make_batches(
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    reuse: bool,
    tmpdir: str,
) -> List[Tuple[str, str]]:
    files: List[Tuple[str, str]] = []

    if reuse:
        for i in range(n_batches):
            X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
            y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
            if not os.path.exists(X_path) or not os.path.exists(y_path):
                files = []
                break
            files.append((X_path, y_path))

    if files:
        return files

    assert not files

    for i in range(n_batches):
        X, y = make_dense_regression(
            "cpu",
            n_samples_per_batch,
            n_features=n_features,
            random_state=i,
            sparsity=0.0,
        )
        X_path = os.path.join(tmpdir, "X-" + str(i) + ".npy")
        y_path = os.path.join(tmpdir, "y-" + str(i) + ".npy")
        np.save(X_path, X)
        np.save(y_path, y)
        files.append((X_path, y_path))
        print(f"Saved to {X_path} and {y_path}", flush=True)

    gc.collect()

    return files



def setup_rmm() -> None:
    print("Use `CudaAsyncMemoryResource`.", flush=True)
    use_rmm_pool = False
    if use_rmm_pool:
        rmm.reinitialize(pool_allocator=True, initial_pool_size=0)
        mr = rmm.mr.get_current_device_resource()
    else:
        mr = rmm.mr.CudaAsyncMemoryResource()
        mr = rmm.mr.PoolMemoryResource(mr)
        mr = rmm.mr.LoggingResourceAdaptor(mr, log_file_name="rmm_log")
        rmm.mr.set_current_device_resource(mr)
    cp.cuda.set_allocator(rmm_cupy_allocator)


@dataclass
class Opts:
    n_samples_per_batch: int
    n_features: int
    n_batches: int
    sparsity: float
    on_the_fly: bool
    validation: bool
    device: str


def make_iter(opts: Opts, tmpdir: str) -> tuple[BenchIter, BenchIter | None]:
    with Timer("MakeIter", "Make"):
        if not opts.on_the_fly:
            files = make_batches(
                n_samples_per_batch=opts.n_samples_per_batch,
                n_features=opts.n_features,
                n_batches=opts.n_batches,
                reuse=True,
                tmpdir=tmpdir,
            )
            it_impl: IterImpl = LoadIterImpl(files, device=opts.device)
        else:
            it_impl = SynIterImpl(
                n_samples_per_batch=opts.n_samples_per_batch,
                n_features=opts.n_features,
                n_batches=opts.n_batches,
                sparsity=opts.sparsity,
                assparse=False,
                device=opts.device,
            )

    it_train = BenchIter(it_impl, split=opts.validation, is_ext=True, is_eval=False)

    if opts.validation:
        it_valid = BenchIter(it_impl, split=opts.validation, is_ext=True, is_eval=True)
        return it_train, it_valid
    else:
        return it_train, None


def extmem_spdm_train(
    opts: Opts,
    n_bins: int,
    n_rounds: int,
    tmpdir: str,
) -> xgb.Booster:
    if opts.device == "cuda":
        setup_rmm()

    it_train, it_valid = make_iter(opts, tmpdir=tmpdir)
    with Timer("ExtQdm", "DMatrix-Train"):
        Xy_train = xgb.DMatrix(it_train)

    watches = [(Xy_train, "Train")]

    if it_valid is not None:
        Xy_valid = xgb.DMatrix(it_valid)
        watches.append((Xy_valid, "Valid"))


    with Timer("ExtSparse", "train"):
        booster = xgb.train(
            {
                "tree_method": "hist",
                "max_depth": 6,
                "device": "cuda",
                "max_bin": n_bins,
            },
            Xy_train,
            num_boost_round=n_rounds,
            evals=watches,
            verbose_eval=True,
        )
    return booster


def extmem_qdm_train(
    opts: Opts,
    n_bins: int,
    n_rounds: int,
    tmpdir: str,
) -> xgb.Booster:
    if opts.device == "cuda":
        setup_rmm()

    it_train, it_valid = make_iter(opts, tmpdir=tmpdir)
    with Timer("ExtQdm", "DMatrix-Train"):
        Xy_train = xgb.ExtMemQuantileDMatrix(it_train, max_bin=n_bins)

    watches = [(Xy_train, "Train")]

    if it_valid is not None:
        Xy_valid = xgb.ExtMemQuantileDMatrix(it_valid, ref=Xy_train)
        watches.append((Xy_valid, "Valid"))

    with Timer("ExtQdm", "train"):
        booster = xgb.train(
            {
                "tree_method": "hist",
                "max_depth": 6,
                "max_bin": n_bins,
                "device": opts.device,
            },
            Xy_train,
            num_boost_round=n_rounds,
            evals=watches,
            verbose_eval=False,
        )
    return booster


def extmem_qdm_inference(
    loadfrom: str,
    n_bins: int,
    n_samples_per_batch: int,
    n_features: int,
    n_batches: int,
    assparse: bool,
    sparsity: float,
    device: str,
    on_the_fly: bool,
    args: argparse.Namespace,
) -> None:
    if device == "cuda":
        setup_rmm()

    if not on_the_fly:
        X_files, y_files = get_file_paths(loadfrom)
        it_impl: IterImpl = LoadIterImpl(list(*zip(X_files, y_files)), device=device)
    else:
        it_impl = SynIterImpl(
            n_samples_per_batch=n_samples_per_batch,
            n_features=n_features,
            n_batches=n_batches,
            sparsity=sparsity,
            assparse=assparse,
            device=device,
        )
    it = BenchIter(it_impl, split=False, is_ext=True, is_eval=False)
    with Timer("inference", "Qdm"):
        Xy = xgb.ExtMemQuantileDMatrix(it, max_bin=n_bins)

    booster = xgb.Booster(model_file=args.model)
    booster.set_param({"device": device})
    with Timer("inference", args.predict_type):
        if args.predict_type == "value":
            booster.predict(Xy)
        elif args.predict_type == "contrib":
            booster.predict(Xy, pred_contribs=True)
        else:
            booster.predict(Xy, pred_interactions=True)
