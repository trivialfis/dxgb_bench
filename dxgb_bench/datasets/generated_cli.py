"""Command-line interface for synthetic dataset generation."""

from __future__ import annotations

import argparse

from ..utils import DFT_OUT, add_data_params, add_device_param, split_path
from .generated import (
    DEFAULT_DATA_SEED,
    DEFAULT_TARGET_SEED,
    datagen,
    regenerate_targets,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the stored-generation and target-regeneration options."""
    add_data_params(parser, False)
    add_device_param(parser)
    parser.set_defaults(n_batches=None)
    parser.add_argument(
        "--n_binary",
        type=int,
        help="Generate this many binary columns, followed by continuous columns; regression only.",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        help=(
            f"Seed for mixed features (default {DEFAULT_DATA_SEED}) or replacement "
            f"targets (default {DEFAULT_TARGET_SEED})."
        ),
    )
    parser.add_argument(
        "--loadfrom",
        help="Share existing features and generate new regression targets. Comma separated shard directories.",
    )
    parser.add_argument(
        "--saveto",
        default=DFT_OUT,
        help="Comma separated list of output directories. Poor man's raid0.",
    )


def run(args: argparse.Namespace) -> None:
    """Validate CLI-only combinations and dispatch to the generation backend."""
    outdirs = split_path(args.saveto)
    if args.loadfrom is not None:
        shape = (
            args.n_samples_per_batch,
            args.n_features,
            args.n_batches,
            args.n_binary,
        )
        if (
            any(v is not None for v in shape)
            or args.assparse
            or args.sparsity != 0.0
            or args.target_type != "reg"
            or args.fmt != "auto"
        ):
            raise ValueError(
                "--loadfrom infers feature shapes and format; only regression targets can be generated."
            )
        regenerate_targets(
            split_path(args.loadfrom),
            outdirs,
            args.n_targets,
            device=args.device,
            random_state=DEFAULT_TARGET_SEED
            if args.data_seed is None
            else args.data_seed,
        )
        return

    if args.n_samples_per_batch is None or args.n_features is None:
        raise ValueError(
            "--n_samples_per_batch and --n_features are required without --loadfrom."
        )
    if args.data_seed is not None and args.n_binary is None:
        raise ValueError("--data_seed requires --n_binary or --loadfrom.")
    datagen(
        args.n_samples_per_batch,
        args.n_features,
        args.n_targets,
        1 if args.n_batches is None else args.n_batches,
        assparse=args.assparse,
        target_type=args.target_type,
        sparsity=args.sparsity,
        device=args.device,
        outdirs=outdirs,
        fmt=args.fmt,
        n_binary=args.n_binary,
        random_state=args.data_seed,
    )
