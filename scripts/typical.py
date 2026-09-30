import argparse
import os
import shlex
import shutil
import subprocess
from itertools import product
from pathlib import Path


def numa_nodes(gpu: str, query: str) -> str:
    """Query the GPU's nearest CPU (-C) or memory (-M) NUMA nodes."""
    output = subprocess.check_output(
        ["nvidia-smi", "topo", query, "-i", gpu], text=True
    )
    nodes = output.rpartition(":")[2].strip().replace(" ", "")
    if not all(node.isdecimal() for node in nodes.split(",")):
        raise RuntimeError(
            f"Cannot determine NUMA nodes for GPU {gpu}: {output.strip()}"
        )
    return nodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir", type=Path, required=True, help="New results directory."
    )
    parser.add_argument("--n_batches", type=int, required=True, help="Batches per run.")
    parser.add_argument("--n_samples_per_batch", type=int, default=2**20)
    parser.add_argument(
        "--gpu", type=int, default=0, help="nvidia-smi GPU index (default: 0)."
    )
    args = parser.parse_args()
    if args.n_batches <= 0 or args.n_samples_per_batch <= 0:
        parser.error("Batch count and samples per batch must be positive.")

    gpu_uuid = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(args.gpu),
            "--query-gpu=uuid",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    cpu_nodes = numa_nodes(gpu_uuid, "-C")
    mem_nodes = numa_nodes(gpu_uuid, "-M")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid}
    print(f"GPU {args.gpu}: CPU NUMA nodes {cpu_nodes}, memory NUMA nodes {mem_nodes}")

    output_dir: Path = args.output_dir
    output_dir.expanduser().mkdir(parents=True)
    common = [
        "numactl",
        f"--cpunodebind={cpu_nodes}",
        f"--membind={mem_nodes}",
        "dxgb-bench",
        "bench",
        "--task=qdm-iter",
        "--fly",
        "--device=cuda",
        "--mr=cuda",
        "--tree_method=hist",
        "--n_rounds=128",
        "--max_depth=6",
        "--n_bins=256",
        f"--n_samples_per_batch={args.n_samples_per_batch}",
        f"--n_batches={args.n_batches}",
    ]
    targets = [(1, "one_output_per_tree"), (4, "multi_output_tree")]
    for n_features, policy, (n_targets, strategy) in product(
        (256, 512), ("depthwise", "lossguide"), targets
    ):
        command = common + [
            f"--n_features={n_features}",
            f"--policy={policy}",
            f"--n_targets={n_targets}",
            f"--multi_strategy={strategy}",
        ]
        print(shlex.join(command), flush=True)
        # dxgb-bench writes the next incore-N.json in its working directory.
        subprocess.run(command, cwd=args.output_dir, env=env, check=True)

    archive = shutil.make_archive(
        str(args.output_dir),
        "zip",
        root_dir=args.output_dir.parent,
        base_dir=args.output_dir.name,
    )
    print(f"Saved results archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
