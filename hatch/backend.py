"""Pass pip's CMake settings to the Hatchling build hook."""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Iterator

import hatchling.build
from hatchling.build import (
    build_sdist,
    get_requires_for_build_editable,
    get_requires_for_build_sdist,
    get_requires_for_build_wheel,
)


@contextmanager
def build_config(config_settings: dict[str, Any] | None) -> Iterator[None]:
    """Scope configuration to one build, preserving the caller's environment."""
    key = "DXGB_CMAKE_ARGS"
    previous = os.environ.get(key)
    try:
        if config_settings and "cmake.args" in config_settings:
            args = config_settings["cmake.args"]
            if isinstance(args, str):
                args = [args]
            if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
                raise ValueError("cmake.args must be a string or a list of strings")
            os.environ[key] = json.dumps(args)
        yield
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


def build_wheel(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    with build_config(config_settings):
        return hatchling.build.build_wheel(
            wheel_directory, config_settings, metadata_directory
        )


def build_editable(
    wheel_directory: str,
    config_settings: dict[str, Any] | None = None,
    metadata_directory: str | None = None,
) -> str:
    with build_config(config_settings):
        return hatchling.build.build_editable(
            wheel_directory, config_settings, metadata_directory
        )
