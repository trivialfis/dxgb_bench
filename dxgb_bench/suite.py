"""Helpers for running GPU benchmark suites in separate processes."""

import os
import shlex
import shutil
import subprocess
from collections.abc import Iterable
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


def run_suite(output_dir: Path, gpu: int, commands: Iterable[list[str]]) -> None:
    """Pin commands to a GPU and its NUMA nodes, then archive their results.

    Requires Linux with numactl and nvidia-smi, like the benchmark environment.
    The results directory must be new; every command writes its results there.
    """
    output_dir = output_dir.expanduser().resolve()
    gpu_uuid = subprocess.check_output(
        ["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"],
        text=True,
    ).strip()
    cpu_nodes = numa_nodes(gpu_uuid, "-C")
    mem_nodes = numa_nodes(gpu_uuid, "-M")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu_uuid}
    print(f"GPU {gpu}: CPU NUMA nodes {cpu_nodes}, memory NUMA nodes {mem_nodes}")
    output_dir.mkdir(parents=True)
    prefix = ["numactl", f"--cpunodebind={cpu_nodes}", f"--membind={mem_nodes}"]
    for command in commands:
        command = prefix + command
        print(shlex.join(command), flush=True)
        # dxgb-bench writes the next incore-N.json in its working directory.
        subprocess.run(command, cwd=output_dir, env=env, check=True)

    archive = shutil.make_archive(
        str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name
    )
    print(f"Saved results archive: {archive}", flush=True)
