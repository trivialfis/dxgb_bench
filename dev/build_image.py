from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess


def main(args: argparse.Namespace) -> None:
    if args.target == "cpu":
        shutil.copyfile("dxgb_bench/dev/Dockerfile.cpu", "Dockerfile")
    else:
        shutil.copyfile("dxgb_bench/dev/Dockerfile.gpu", "Dockerfile")
    build_args = args.build_args
    if args.tag is None:
        tag = f"dxgb-bench-{args.target}-{args.arch}-{args.sm}:latest"
    else:
        tag = args.tag
    cmd = [
        "docker",
        "build",
        "--progress=plain",
        ".",
        "-t",
        tag,
        "--build-arg",
        f"SM={args.sm}",
        "--build-arg",
        f"ARCH={args.arch}",
    ]
    if args.install_xgboost:
        cmd.extend(["--build-arg", "INSTALL_XGBOOST=1"])
    else:
        cmd.extend(["--build-arg", "INSTALL_XGBOOST="])
    if args.xgboost_repo is not None:
        assert args.install_xgboost
        cmd.extend(["--build-arg", f"XGBOOST_REPO={args.xgboost_repo}"])
    else:
        cmd.extend(["--build-arg", "XGBOOST_REPO="])
    if args.xgboost_checkout is not None:
        assert args.install_xgboost
        cmd.extend(["--build-arg", f"XGBOOST_CHECKOUT={args.xgboost_checkout}"])
    else:
        cmd.extend(["--build-arg", "XGBOOST_CHECKOUT="])
    if build_args is not None:
        build_args_lst: list[str] = build_args.split(";")
        for kv in build_args_lst:
            cmd.extend(["--build-arg", kv])

    subprocess.check_call(cmd)
    os.remove("Dockerfile")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""

Examples:

    python ./dxgb_bench/dev/build_image.py --target=gpu --arch=aarch --sm=90a

    """
    )
    machine = platform.machine()
    dft_arch = "aarch" if machine.startswith("aarch") else "x86"
    parser.add_argument(
        "--arch", choices=["x86", "aarch"], required=False, default=dft_arch
    )
    parser.add_argument("--sm", type=str, required=True)
    parser.add_argument(
        "--build-args",
        help=";separated list of docker build arguments.",
        required=False,
    )
    parser.add_argument("--install-xgboost", action="store_true")
    parser.add_argument("--xgboost-checkout", default=None, type=str)
    parser.add_argument("--xgboost-repo", default=None, type=str)
    parser.add_argument("--target", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--tag", type=str, default=None)
    args = parser.parse_args()
    main(args)
