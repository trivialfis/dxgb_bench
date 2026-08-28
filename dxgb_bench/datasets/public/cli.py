"""Command-line interface for the public-dataset pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from .pipeline import DEFAULT_CACHE, PublicDatasetPipeline
from .registry import DATASETS

DESCRIPTION = "Fetch, process, cache, and validate public benchmark datasets."


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add public-dataset arguments to a standalone or parent parser."""
    names = sorted(DATASETS)
    parser.add_argument(
        "datasets",
        nargs="*",
        choices=names,
        help="Datasets to process (default: every registered dataset).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help=f"Prepared cache root (default: {DEFAULT_CACHE}).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Reprocess cached source files even when prepared arrays are valid.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Never access the network; fail if a required source is not cached.",
    )
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--download-only",
        action="store_true",
        help="Cache original source files without processing them.",
    )
    action.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate prepared caches without downloading or rebuilding them.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered datasets and exit.",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue processing other datasets after an error.",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    return add_arguments(parser)


def run(args: argparse.Namespace) -> None:
    """Execute a parsed public-dataset command."""
    names = list(DATASETS)
    if args.list:
        for name in names:
            spec = DATASETS[name]
            print(
                f"{name}\t{spec.task}\t{spec.rows}\t{spec.features}\t"
                f"{spec.outputs}\t{spec.title}"
            )
        return
    selected = args.datasets or names
    pipeline = PublicDatasetPipeline(cache_dir=args.cache_dir)
    failures = []
    for name in selected:
        try:
            if args.download_only:
                pipeline.fetch(name, offline=args.offline)
            elif args.validate_only:
                arrays = pipeline.load(name)
                print(f"Valid {name}: X={arrays.X.shape}, y={arrays.y.shape}")
            else:
                pipeline.ensure(name, rebuild=args.rebuild, offline=args.offline)
        except Exception as error:
            if not args.keep_going:
                raise
            failures.append(name)
            print(f"Failed {name}: {error}", file=sys.stderr, flush=True)
    if failures:
        raise RuntimeError(f"Failed datasets: {', '.join(failures)}")


def validate_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Validate combinations that argparse's mutually exclusive group cannot express."""
    if args.rebuild and (args.download_only or args.validate_only):
        parser.error("--rebuild cannot be combined with a single-stage action")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    """Run the public-dataset pipeline CLI."""
    parser = build_parser()
    args = validate_args(parser, parser.parse_args(argv))
    run(args)


if __name__ == "__main__":
    main()
