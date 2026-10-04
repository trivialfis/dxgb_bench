"""Benchmark binary/continuous feature groups with very different bin counts."""

import argparse
from itertools import product
from pathlib import Path

from dxgb_bench.suite import run_suite


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="New results directory."
    )
    parser.add_argument("--n_batches", type=int, default=4)
    parser.add_argument("--n_samples_per_batch", type=int, default=2**17)
    parser.add_argument("--n_rounds", type=int, default=16)
    parser.add_argument(
        "--gpu", type=int, default=0, help="nvidia-smi GPU index (default: 0)."
    )
    args = parser.parse_args()
    if min(args.n_batches, args.n_samples_per_batch, args.n_rounds) <= 0:
        parser.error("Batch, sample, and round counts must be positive.")

    common = [
        "dxgb-bench",
        "bench",
        "--task=qdm-iter",
        "--fly",
        "--device=cuda",
        "--mr=cuda",
        "--tree_method=hist",
        "--n_features=4080",
        "--n_binary=3072",
        "--data_seed=2026",
        "--n_bins=256",
        f"--n_rounds={args.n_rounds}",
        f"--n_samples_per_batch={args.n_samples_per_batch}",
        f"--n_batches={args.n_batches}",
    ]
    targets = [(1, "one_output_per_tree"), (4, "multi_output_tree")]
    commands = []
    for (n_targets, strategy), policy, max_depth in product(
        targets, ("depthwise", "lossguide"), (6, 8)
    ):
        commands.append(
            common
            + [
                f"--policy={policy}",
                f"--max_depth={max_depth}",
                f"--n_targets={n_targets}",
                f"--multi_strategy={strategy}",
            ]
        )
    run_suite(args.output_dir, args.gpu, commands)


if __name__ == "__main__":
    main()
