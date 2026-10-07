"""Build the CMake library before Hatchling assembles a wheel."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface
from packaging.tags import platform_tags


class CMakeBuildHook(BuildHookInterface):
    """Package a freshly built library, including for editable installations."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        root = Path(self.root)
        source = root / "dxgb_bench"
        build = root / "build" / "hatch"
        output = build / "lib"
        args = shlex.split(os.environ.get("CMAKE_ARGS", ""))
        for value in json.loads(os.environ.get("DXGB_CMAKE_ARGS", "[]")):
            args.extend(shlex.split(value))

        subprocess.run(
            [
                "cmake",
                "-S",
                str(source),
                "-B",
                str(build),
                "-G",
                os.environ.get("CMAKE_GENERATOR", "Ninja"),
                "-DCMAKE_BUILD_TYPE=Release",
                "-DCMAKE_CUDA_ARCHITECTURES=all",
                *args,
                "-DDXGB_USE_CUDA=ON",
                f"-DDXGB_OUTPUT_DIRECTORY={output}",
            ],
            check=True,
        )
        build_type = "Release"
        for line in (build / "CMakeCache.txt").read_text().splitlines():
            if line.startswith("CMAKE_BUILD_TYPE:"):
                build_type = line.split("=", 1)[1] or "Release"
        subprocess.run(
            [
                "cmake",
                "--build",
                str(build),
                "--config",
                build_type,
                "--parallel",
                os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", str(os.cpu_count() or 1)),
            ],
            check=True,
        )

        name = "dxgbbench.dll" if platform.system() == "Windows" else "libdxgbbench.so"
        library = output / name
        if not library.is_file():
            raise FileNotFoundError(f"CMake did not produce {library}")
        if version == "editable":
            shutil.copy2(library, source / name)
        else:
            build_data["force_include"][str(library)] = f"dxgb_bench/{name}"
        # This is a ctypes library, with no dependency on the CPython ABI.
        build_data["pure_python"] = False
        build_data["tag"] = f"py3-none-{next(platform_tags())}"
