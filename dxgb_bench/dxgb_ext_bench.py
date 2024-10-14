#!/usr/bin/env python
import argparse
import os

import xgboost as xgb

from .external_mem import run_ext_qdm, run_external_memory, run_inference
from .utils import Timer


def main(args: argparse.Namespace) -> None:
    data_dir = "./data"

    n_batches = args.n_batches

    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    if args.size == "test":
        n = 2**19 * n_batches
    elif args.size == "small":
        n = 2**23
    elif args.size == "custom":
        assert args.n_samples_per_batch > 0
        n = args.n_samples_per_batch * n_batches
    else:
        n = (2**23 + 2**22) * n_batches

    if args.task == "ext":
        assert args.device == "cuda"
        run_external_memory(
            data_dir,
            reuse=True,
            on_host=True,
            n_bins=args.n_bins,
            n_batches=n_batches,
            n_samples_per_batch=n // n_batches,
        )
    elif args.task == "ext-qdm":
        run_ext_qdm(
            data_dir,
            reuse=True,
            n_bins=args.n_bins,
            n_batches=n_batches,
            n_samples_per_batch=n // n_batches,
            device=args.device,
            n_rounds=args.n_rounds,
            sparsity=args.sparsity,
            on_the_fly=args.on_the_fly == 1,
            validation=args.valid,
        )
    else:
        assert args.predict_type is not None
        assert args.model is not None
        run_inference(
            data_dir,
            reuse=True,
            n_bins=args.n_bins,
            n_batches=n_batches,
            n_samples_per_batch=n // n_batches,
            device=args.device,
            on_the_fly=args.on_the_fly == 1,
            args=args,
        )

    print(Timer.global_timer())


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["ext", "ext-qdm", "inf"], required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], required=True)
    parser.add_argument(
        "--size", choices=["test", "small", "large", "custom"], default="small"
    )
    parser.add_argument("--n_samples_per_batch", type=int, required=False)
    parser.add_argument("--n_rounds", type=int, default=128)
    parser.add_argument("--n_batches", type=int, default=54)
    parser.add_argument("--n_bins", type=int, default=256)
    parser.add_argument("--on-the-fly", choices=[0, 1], default=1)
    parser.add_argument("--valid", action="store_true")
    parser.add_argument("--sparsity", type=float, default=0.0)

    parser.add_argument("--model", type=str, required=False)
    parser.add_argument(
        "--predict_type", choices=["values", "contribs", "interactions"], required=False
    )

    args = parser.parse_args()

    with xgb.config_context(verbosity=3, use_rmm=True):
        main(args)
