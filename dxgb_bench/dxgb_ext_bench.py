#!/usr/bin/env python
import argparse
import os

import xgboost as xgb

from dxgb_bench.external_mem import (
    run_ext_qdm_cpu,
    run_external_memory,
    run_over_subscription,
)


def main(args: argparse.Namespace) -> None:
    data_dir = "./data"
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    n = 2**28
    n_batches = 32

    if args.task == "os":
        assert args.device == "cuda"
        run_over_subscription(
            data_dir,
            True,
            n_bins=256,
            n_samples_per_batch=n // n_batches,
            n_batches=n_batches,
            is_sam=True,
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


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["os", "ext", "ext-qdm"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    args = parser.parse_args()

    with xgb.config_context(verbosity=3):
        main(args)
