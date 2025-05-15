#!/usr/bin/env python
"""Copyright (c) 2024-2025, Jiaming Yuan.  All rights reserved."""

import argparse
import os

import xgboost as xgb

from .external_mem import (
    qdm_train,
    spdm_train,
)
from .utils import (
    Opts,
    Timer,
    add_data_params,
    add_device_param,
    add_hyper_param,
    add_rmm_param,
    make_params_from_args,
    split_path,
)


def main(args: argparse.Namespace) -> None:
    n_batches = args.n_batches

    if args.fly:
        n = args.n_samples_per_batch * n_batches
    else:
        n = 0

    n_features = args.n_features
    opts = Opts(
        n_samples_per_batch=n // n_batches,
        n_features=n_features,
        n_batches=n_batches,
        sparsity=args.sparsity,
        on_the_fly=args.fly,
        validation=args.valid,
        device=args.device,
        mr=args.mr,
        target_type=args.target_type,
        cache_host_ratio=args.cache_host_ratio,
    )
    assert opts.mr is not None
    loadfrom = split_path(args.loadfrom)

    if args.task == "ext-sp":
        spdm_train(
            opts,
            params=make_params_from_args(args),
            n_rounds=args.n_rounds,
            loadfrom=loadfrom,
        )
    elif args.task == "ext-qdm":
        qdm_train(
            opts,
            params=make_params_from_args(args),
            n_rounds=args.n_rounds,
            loadfrom=loadfrom,
        )
    else:
        raise ValueError(f"Invalid task: {args.task}")

    print(Timer.global_timer())


def cli_main() -> None:
    dft_out = os.path.join(os.curdir, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=["ext-sp", "ext-qdm"],
        required=True,
        help="""
ext-sp: Use the sparse DMatrix (the DMatrix class in Python).
ext-qdm: Use the ExtMemQuantileDMatrix.
        """,
    )
    parser = add_rmm_param(parser)
    parser = add_device_param(parser)
    parser.add_argument("--loadfrom", type=str, default=dft_out)

    parser = add_data_params(parser, required=False, n_features=512)
    parser = add_hyper_param(parser)

    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument("--valid", action="store_true")

    args = parser.parse_args()
    if args.target_type == "bin" and args.fly is not True:
        raise NotImplementedError(
            "`--fly` must be true for binary classification target."
        )

    with xgb.config_context(verbosity=args.verbosity, use_rmm=True):
        main(args)
