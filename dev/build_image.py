from __future__ import annotations

import argparse
import os
import shutil
import subprocess


def main(args: argparse.Namespace) -> None:
    if args.target == "cpu":
        shutil.copyfile("dxgb_bench/dev/Dockerfile.cpu", "Dockerfile")
    else:
        shutil.copyfile("dxgb_bench/dev/Dockerfile.gpu", "Dockerfile")
    build_args = args.build_args
    cmd = ["docker", "build", ".", "-t", f"dxgb-bench-{args.target}:latest"]
    if build_args is not None:
        build_args_lst: list[str] = build_args.split(";")
        for kv in build_args_lst:
            cmd.extend(["--build-arg", kv])

    subprocess.check_call(cmd)
    os.remove("Dockerfile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="""

Examples:

    python ./dxgb_bench/dev/build_image.py --target=gpu --build-args="ARCH=x86;SM=90a"

    """)
    parser.add_argument(
        "--build-args", help=";separated list of docker build arguments."
    )
    parser.add_argument("--target", choices=["cpu", "gpu"], default="gpu")
    args = parser.parse_args()
    main(args)
