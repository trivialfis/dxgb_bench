"""Reusable fetch, process, cache, and validation pipeline for public datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .models import DatasetArrays, DatasetSpec, PreparedDataset
from .processors import PROCESSORS, Processor
from .registry import DATASETS
from .storage import download, save_array, sha256, write_json

CACHE_FORMAT_VERSION = 2


def default_cache_dir() -> Path:
    """Return the user-writable cache directory for prepared public datasets."""
    configured = os.environ.get("DXGB_BENCH_DATASET_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "dxgb_bench" / "datasets"


DEFAULT_CACHE = default_cache_dir()


def validate_prepared_values(spec: DatasetSpec, prepared: PreparedDataset) -> None:
    """Validate processor output before it enters or leaves the cache."""
    X = prepared.X
    y = prepared.y
    if X.shape != (spec.rows, spec.features):
        raise ValueError(
            f"{spec.name}: expected X shape {(spec.rows, spec.features)}, found {X.shape}"
        )
    expected_y = (
        (spec.rows,) if spec.task == "classification" else (spec.rows, spec.outputs)
    )
    if y.shape != expected_y:
        raise ValueError(f"{spec.name}: expected y shape {expected_y}, found {y.shape}")
    if X.dtype != np.float32:
        raise ValueError(f"{spec.name}: expected float32 features, found {X.dtype}")
    expected_y_dtype = np.int32 if spec.task == "classification" else np.float32
    if y.dtype != expected_y_dtype:
        raise ValueError(
            f"{spec.name}: expected {expected_y_dtype} labels, found {y.dtype}"
        )
    if np.isinf(X).any() or not np.isfinite(y).all():
        raise ValueError(f"{spec.name}: prepared arrays contain invalid values")
    if (
        len(prepared.feature_names) != spec.features
        or len(set(prepared.feature_names)) != spec.features
    ):
        raise ValueError(f"{spec.name}: feature names are missing or duplicated")
    if len(prepared.feature_types) != spec.features or not set(
        prepared.feature_types
    ) <= {"q", "c"}:
        raise ValueError(f"{spec.name}: invalid XGBoost feature types")

    for index, feature_type in enumerate(prepared.feature_types):
        if feature_type == "c":
            values = X[:, index]
            values = values[np.isfinite(values)]
            if np.any(values < 0) or not np.array_equal(values, np.floor(values)):
                raise ValueError(
                    f"{spec.name}: categorical feature "
                    f"{prepared.feature_names[index]} must contain non-negative "
                    "integer codes"
                )

    if spec.task == "classification":
        unique = np.unique(y)
        if not np.array_equal(unique, np.arange(spec.outputs, dtype=np.int32)):
            raise ValueError(f"{spec.name}: labels are not zero-based and contiguous")
    if prepared.strata is not None and (
        prepared.strata.shape != (spec.rows,) or prepared.strata.dtype != np.int32
    ):
        raise ValueError(f"{spec.name}: invalid split strata")
    if prepared.groups is not None:
        if prepared.groups.shape != (spec.rows,) or prepared.groups.dtype != np.int32:
            raise ValueError(f"{spec.name}: invalid split groups")
        if prepared.groups.min() < 0:
            raise ValueError(f"{spec.name}: split groups must be non-negative")
    if spec.split_kind == "stratified_group" and (
        prepared.strata is None or prepared.groups is None
    ):
        raise ValueError(f"{spec.name}: stratified group splitting needs both arrays")


class PublicDatasetPipeline:
    """Orchestrate public source downloads and versioned prepared caches."""

    def __init__(
        self,
        cache_dir: Path = DEFAULT_CACHE,
        registry: Mapping[str, DatasetSpec] = DATASETS,
        processors: Mapping[str, Processor] = PROCESSORS,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.registry = dict(registry)
        self.processors = dict(processors)

    def spec(self, name: str) -> DatasetSpec:
        """Return a registered specification with a useful unknown-name error."""
        try:
            return self.registry[name]
        except KeyError as error:
            available = ", ".join(sorted(self.registry))
            raise KeyError(
                f"Unknown dataset {name!r}; available: {available}"
            ) from error

    def dataset_dir(self, name: str) -> Path:
        return self.cache_dir / name

    def source_path(self, name: str) -> Path:
        spec = self.spec(name)
        return self.dataset_dir(name) / spec.source_filename

    def fetch(self, name: str, *, offline: bool = False) -> Path:
        """Ensure the original public source is present in the local cache."""
        spec = self.spec(name)
        source = self.source_path(name)
        if source.is_file():
            print(f"Using cached download {source}", flush=True)
            return source
        if offline:
            raise FileNotFoundError(
                f"No cached source for {name!r} at {source}; offline mode is enabled"
            )
        return download(spec.source_url, source)

    def process(self, name: str, source: Path | None = None) -> PreparedDataset:
        """Transform a downloaded source into the canonical in-memory representation."""
        spec = self.spec(name)
        source = Path(source) if source is not None else self.source_path(name)
        if not source.is_file():
            raise FileNotFoundError(f"Source for {name!r} is missing: {source}")
        try:
            processor = self.processors[name]
        except KeyError as error:
            raise KeyError(f"No source processor registered for {name!r}") from error
        prepared = processor(spec, source)
        prepared = PreparedDataset(
            X=np.ascontiguousarray(prepared.X, dtype=np.float32),
            y=np.ascontiguousarray(
                prepared.y,
                dtype=np.int32 if spec.task == "classification" else np.float32,
            ),
            feature_names=[str(feature) for feature in prepared.feature_names],
            feature_types=list(prepared.feature_types),
            strata=(
                np.ascontiguousarray(prepared.strata, dtype=np.int32)
                if prepared.strata is not None
                else None
            ),
            groups=(
                np.ascontiguousarray(prepared.groups, dtype=np.int32)
                if prepared.groups is not None
                else None
            ),
            details=dict(prepared.details),
        )
        validate_prepared_values(spec, prepared)
        return prepared

    def prepare(self, name: str, *, offline: bool = False) -> DatasetArrays:
        """Fetch, process, atomically cache, and reload one dataset."""
        spec = self.spec(name)
        source = self.fetch(name, offline=offline)
        print(f"Preparing {name}", flush=True)
        prepared = self.process(name, source)
        directory = self.dataset_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        incomplete = directory / ".incomplete"
        incomplete.write_text("prepared cache update in progress\n", encoding="utf-8")

        save_array(directory / "X.npy", prepared.X)
        save_array(directory / "y.npy", prepared.y)
        optional_arrays = {
            "strata.npy": prepared.strata,
            "groups.npy": prepared.groups,
        }
        for filename, array in optional_arrays.items():
            path = directory / filename
            if array is not None:
                save_array(path, array)
            elif path.exists():
                path.unlink()

        metadata = self._metadata(spec, source, prepared)
        write_json(directory / "metadata.json", metadata)
        incomplete.unlink()

        arrays = self.load(name)
        print(f"Prepared {name}: X={arrays.X.shape}, y={arrays.y.shape}", flush=True)
        return arrays

    def load(self, name: str) -> DatasetArrays:
        """Load and validate an existing prepared cache without network access."""
        spec = self.spec(name)
        directory = self.dataset_dir(name)
        required = [
            directory / "metadata.json",
            directory / "X.npy",
            directory / "y.npy",
        ]
        if (directory / ".incomplete").exists() or not all(
            path.is_file() for path in required
        ):
            raise FileNotFoundError(f"Prepared cache for {name} is incomplete")

        metadata = json.loads(required[0].read_text(encoding="utf-8"))
        if metadata.get("cache_format_version") != CACHE_FORMAT_VERSION:
            raise FileNotFoundError(f"Prepared cache for {name} uses an old format")
        for key, expected in {
            "dataset": spec.name,
            "task": spec.task,
            "rows": spec.rows,
            "features": spec.features,
            "outputs": spec.outputs,
        }.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"{name}: cache metadata {key!r} does not match the registry"
                )

        X = np.load(required[1], mmap_mode="r", allow_pickle=False)
        y = np.load(required[2], mmap_mode="r", allow_pickle=False)
        strata = self._load_optional_array(directory, metadata, "strata")
        groups = self._load_optional_array(directory, metadata, "groups")
        prepared = PreparedDataset(
            X=X,
            y=y,
            feature_names=list(metadata["feature_names"]),
            feature_types=list(metadata["feature_types"]),
            strata=strata,
            groups=groups,
        )
        validate_prepared_values(spec, prepared)
        return DatasetArrays(
            spec=spec,
            X=X,
            y=y,
            feature_names=prepared.feature_names,
            feature_types=prepared.feature_types,
            strata=strata,
            groups=groups,
            metadata=metadata,
        )

    def ensure(
        self, name: str, *, rebuild: bool = False, offline: bool = False
    ) -> DatasetArrays:
        """Return a valid cache, building it from the public source when needed."""
        if not rebuild:
            try:
                arrays = self.load(name)
            except FileNotFoundError:
                pass
            else:
                print(
                    f"Using prepared {name}: X={arrays.X.shape}, y={arrays.y.shape}",
                    flush=True,
                )
                return arrays
        return self.prepare(name, offline=offline)

    @staticmethod
    def _load_optional_array(
        directory: Path, metadata: Mapping[str, Any], stem: str
    ) -> np.ndarray | None:
        if not metadata.get(f"has_{stem}"):
            return None
        path = directory / f"{stem}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"Prepared cache is missing {path}")
        return np.load(path, mmap_mode="r", allow_pickle=False)

    def _metadata(
        self, spec: DatasetSpec, source: Path, prepared: PreparedDataset
    ) -> dict[str, Any]:
        categorical_features = [
            name
            for name, feature_type in zip(
                prepared.feature_names, prepared.feature_types, strict=True
            )
            if feature_type == "c"
        ]
        metadata: dict[str, Any] = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "dataset": spec.name,
            "title": spec.title,
            "task": spec.task,
            "target": spec.target,
            "repository_url": spec.repository_url,
            "source_url": spec.source_url,
            "source_file": spec.source_filename,
            "source_bytes": source.stat().st_size,
            "source_sha256": sha256(source),
            "citation": spec.citation,
            "license": spec.license,
            "rows": spec.rows,
            "features": spec.features,
            "outputs": spec.outputs,
            "split_kind": spec.split_kind,
            "feature_names": prepared.feature_names,
            "feature_types": prepared.feature_types,
            "categorical_features": categorical_features,
            "feature_dtype": str(prepared.X.dtype),
            "label_dtype": str(prepared.y.dtype),
            "has_strata": prepared.strata is not None,
            "has_groups": prepared.groups is not None,
        }
        conflicting = set(metadata).intersection(prepared.details)
        if conflicting:
            names = ", ".join(sorted(conflicting))
            raise ValueError(
                f"{spec.name}: processor metadata overrides reserved keys: {names}"
            )
        metadata.update(prepared.details)
        if spec.task == "classification":
            metadata["class_counts"] = np.bincount(
                prepared.y, minlength=spec.outputs
            ).tolist()
        else:
            metadata["target_mean"] = np.mean(
                prepared.y, axis=0, dtype=np.float64
            ).tolist()
            metadata["target_std"] = np.std(
                prepared.y, axis=0, dtype=np.float64
            ).tolist()
        return metadata
