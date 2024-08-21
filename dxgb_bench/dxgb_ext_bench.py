#!/usr/bin/env python
import argparse
import os

import xgboost as xgb

from dxgb_bench.external_mem import (
    run_ext_qdm_cpu,
    run_external_memory,
    run_over_subscription,
)

from .utils import Timer


def main(args: argparse.Namespace) -> None:
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if args.size == "test":
        n = 2**16
    elif args.size == "small":
        n = 2**23
    else:
        n = 2**26 + 2**24

    n_batches = 32

    if args.task == "os":
        assert args.device == "cuda"
        run_over_subscription(
            data_dir,
            True,
            n_bins=256,
            n_samples_per_batch=n,
            n_batches=1,  # Single batch for OS bench.
            is_sam=True,
        )
    elif args.task == "osd":
        run_over_subscription(
            data_dir,
            True,
            n_bins=256,
            n_samples_per_batch=n,
            n_batches=1,  # Single batch for OS bench.
            is_sam=False,
        )
    elif args.task == "ext":
        assert args.device == "cuda"
        run_external_memory(
            data_dir,
            reuse=True,
            on_host=True,
            n_batches=n_batches,
            n_samples_per_batch=n // n_batches,
        )
    else:
        assert args.device == "cpu"
        run_ext_qdm_cpu(
            data_dir,
            reuse=True,
            n_bins=256,
            n_batches=n_batches,
            n_samples_per_batch=n // n_batches,
        )

    print(Timer.global_timer())


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task", choices=["os", "osd", "ext", "ext-qdm"], required=True
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument("--size", choices=["test", "small", "large"], default="small")
    args = parser.parse_args()

    with xgb.config_context(verbosity=3, use_rmm=True):
        main(args)