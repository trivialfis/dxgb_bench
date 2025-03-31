#!/usr/bin/env python
import argparse
import os

import xgboost as xgb

from .external_mem import (
    Opts,
    extmem_qdm_inference,
    extmem_qdm_train,
    extmem_spdm_train,
)
from .utils import Timer, add_data_params, add_device_param, add_rmm_param, split_path


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
    )
    loadfrom = split_path(args.loadfrom)

    if args.task == "ext-sp":
        extmem_spdm_train(
            opts,
            n_bins=args.n_bins,
            n_rounds=args.n_rounds,
            loadfrom=loadfrom,
        )
    elif args.task == "ext-qdm":
        extmem_qdm_train(
            opts,
            n_bins=args.n_bins,
            n_rounds=args.n_rounds,
            loadfrom=loadfrom,
        )
    else:
        assert args.predict_type is not None
        assert args.model is not None
        extmem_qdm_inference(
            loadfrom=loadfrom,
            n_bins=args.n_bins,
            n_samples_per_batch=n // n_batches,
            n_features=n_features,
            n_batches=n_batches,
            assparse=False,
            sparsity=args.sparsity,
            device=args.device,
            on_the_fly=args.fly == 1,
            args=args,
        )

    print(Timer.global_timer())


def cli_main() -> None:
    dft_out = os.path.join(os.curdir, "data")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["ext-sp", "ext-qdm", "ext-inf"], required=True
    )
    parser = add_rmm_param(parser)
    parser = add_device_param(parser)
    parser.add_argument("--loadfrom", type=str, default=dft_out)

    parser = add_data_params(parser, required=False, n_features=512)

    parser.add_argument("--n_rounds", type=int, default=128)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument(
        "--fly",
        action="store_true",
        help="Generate data on the fly instead of loading it from the disk.",
    )
    parser.add_argument("--valid", action="store_true")

    parser.add_argument("--model", type=str, required=False)
    parser.add_argument(
        "--predict_type", choices=["value", "contrib", "interaction"], required=False
    )
    parser.add_argument("--verbosity", choices=[0, 1, 2, 3], default=3, type=int)

    args = parser.parse_args()

    with xgb.config_context(verbosity=args.verbosity, use_rmm=True):
        main(args)
