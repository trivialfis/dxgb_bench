"""Reusable fetch, process, cache, and validation pipeline for public datasets."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import DatasetArrays, DatasetSpec, PreparedDataset
from .processors import PROCESSORS, Processor
from .registry import DATASETS
from .storage import download, file_lock, save_array, save_frame, sha256, write_json


def default_cache_dir() -> Path:
    """Return the user-writable cache directory for prepared public datasets."""
    configured = os.environ.get("DXGB_BENCH_DATASET_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_home = os.environ.get("XDG_CACHE_HOME")
    root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return root / "dxgb_bench" / "datasets"


DEFAULT_CACHE = default_cache_dir()


def _normalize_categories(frame: pd.DataFrame, dataset: str) -> pd.DataFrame:
    """Make pandas categories directly consumable by XGBoost."""
    for column in frame.select_dtypes(include="category"):
        categories = frame[column].cat.categories
        if pd.api.types.is_float_dtype(categories.dtype):
            values = frame[column].astype("Float64")
            if not values.dropna().mod(1).eq(0).all():
                raise ValueError(
                    f"{dataset}: categorical feature {column!r} has "
                    "non-integral floating-point levels"
                )
            frame[column] = values.astype("Int64").astype("category")
    return frame


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
    if not isinstance(y, np.ndarray) or y.shape != expected_y:
        raise ValueError(f"{spec.name}: expected y shape {expected_y}, found {y.shape}")
    expected_y_dtype = np.int32 if spec.task == "classification" else np.float32
    if y.dtype != expected_y_dtype:
        raise ValueError(
            f"{spec.name}: expected {expected_y_dtype} labels, found {y.dtype}"
        )
    if (
        len(prepared.feature_names) != spec.features
        or len(set(prepared.feature_names)) != spec.features
    ):
        raise ValueError(f"{spec.name}: feature names are missing or duplicated")
    if isinstance(X, pd.DataFrame):
        if list(X.columns) != prepared.feature_names:
            raise ValueError(
                f"{spec.name}: DataFrame columns do not match feature names"
            )
        invalid = [
            name
            for name, dtype in X.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
            and not isinstance(dtype, pd.CategoricalDtype)
        ]
        if invalid:
            raise ValueError(
                f"{spec.name}: non-numeric columns are not categorical: {invalid}"
            )
        for name in X.select_dtypes(include="category"):
            categories = X[name].cat.categories
            if categories.empty:
                raise ValueError(f"{spec.name}: categorical feature {name!r} is empty")
            if pd.api.types.is_float_dtype(categories.dtype):
                raise ValueError(
                    f"{spec.name}: categorical feature {name!r} has floating-point levels"
                )
        numeric = X.select_dtypes(include="number").to_numpy()
        if np.isinf(numeric).any():
            raise ValueError(f"{spec.name}: prepared features contain infinity")
    else:
        if X.dtype != np.float32:
            raise ValueError(f"{spec.name}: expected float32 features, found {X.dtype}")
        if np.isinf(X).any():
            raise ValueError(f"{spec.name}: prepared features contain infinity")
    if not np.isfinite(y).all():
        raise ValueError(f"{spec.name}: prepared labels contain invalid values")

    if spec.task == "classification":
        unique = np.unique(y)
        if not np.array_equal(unique, np.arange(spec.outputs, dtype=np.int32)):
            raise ValueError(f"{spec.name}: labels are not zero-based and contiguous")
    if prepared.split is not None:
        if prepared.split.shape != (spec.rows,) or prepared.split.dtype != np.int32:
            raise ValueError(f"{spec.name}: invalid split array")
        if not np.isin(prepared.split, [0, 1, 2]).all():
            raise ValueError(
                f"{spec.name}: split values must be train/valid/test codes"
            )
    if spec.split_kind in {"official_test", "predefined"} and prepared.split is None:
        raise ValueError(
            f"{spec.name}: {spec.split_kind} splitting needs a split array"
        )
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
    """Orchestrate public source downloads and prepared caches."""

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
        details = dict(prepared.details)
        if isinstance(prepared.X, pd.DataFrame):
            frame = prepared.X.copy()
            frame.columns = [str(column) for column in frame.columns]
            X: np.ndarray | pd.DataFrame = _normalize_categories(frame, name)
        else:
            X = np.ascontiguousarray(prepared.X, dtype=np.float32)

        if spec.task == "classification":
            labels = pd.Categorical(np.asarray(prepared.y).reshape(-1))
            y = labels.codes.astype(np.int32)
            details["class_labels"] = [str(label) for label in labels.categories]
        else:
            y = np.asarray(prepared.y, dtype=np.float32)
            if y.ndim == 1:
                y = y.reshape(-1, 1)

        prepared = PreparedDataset(
            X=X,
            y=np.ascontiguousarray(y),
            feature_names=[str(feature) for feature in prepared.feature_names],
            split=(
                np.ascontiguousarray(prepared.split, dtype=np.int32)
                if prepared.split is not None
                else None
            ),
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
            details=details,
        )
        validate_prepared_values(spec, prepared)
        return prepared

    def prepare(self, name: str, *, offline: bool = False) -> DatasetArrays:
        """Fetch, process, atomically cache, and reload one dataset."""
        lock = self.dataset_dir(name) / ".prepare.lock"
        with file_lock(lock):
            return self._prepare(name, offline=offline)

    def _prepare(self, name: str, *, offline: bool = False) -> DatasetArrays:
        spec = self.spec(name)
        source = self.fetch(name, offline=offline)
        print(f"Preparing {name}", flush=True)
        prepared = self.process(name, source)
        directory = self.dataset_dir(name)
        directory.mkdir(parents=True, exist_ok=True)
        incomplete = directory / ".incomplete"
        incomplete.write_text("prepared cache update in progress\n", encoding="utf-8")

        if isinstance(prepared.X, pd.DataFrame):
            save_frame(directory / "X.parquet", prepared.X)
            (directory / "X.npy").unlink(missing_ok=True)
        else:
            save_array(directory / "X.npy", prepared.X)
            (directory / "X.parquet").unlink(missing_ok=True)
        save_array(directory / "y.npy", prepared.y)
        optional_arrays = {
            "split.npy": prepared.split,
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

        arrays = self._load(name)
        print(f"Prepared {name}: X={arrays.X.shape}, y={arrays.y.shape}", flush=True)
        return arrays

    def load(self, name: str) -> DatasetArrays:
        """Load and validate an existing prepared cache without network access."""
        lock = self.dataset_dir(name) / ".prepare.lock"
        with file_lock(lock):
            return self._load(name)

    def _load(self, name: str) -> DatasetArrays:
        spec = self.spec(name)
        directory = self.dataset_dir(name)
        required = [directory / "metadata.json", directory / "y.npy"]
        if (directory / ".incomplete").exists() or not all(
            path.is_file() for path in required
        ):
            raise FileNotFoundError(f"Prepared cache for {name} is incomplete")

        metadata = json.loads(required[0].read_text(encoding="utf-8"))
        for key, expected in {
            "dataset": spec.name,
            "title": spec.title,
            "task": spec.task,
            "rows": spec.rows,
            "features": spec.features,
            "outputs": spec.outputs,
            "target": spec.target,
            "repository_url": spec.repository_url,
            "source_url": spec.source_url,
            "source_file": spec.source_filename,
            "split_kind": spec.split_kind,
            "citation": spec.citation,
            "license": spec.license,
            "registered_categorical_features": list(spec.categorical_features),
            "registered_numeric_features": list(spec.numeric_features),
            "dropped_features": list(spec.drop_features),
        }.items():
            if metadata.get(key) != expected:
                raise ValueError(
                    f"{name}: cache metadata {key!r} does not match the registry"
                )

        frame_path = directory / "X.parquet"
        array_path = directory / "X.npy"
        if frame_path.is_file():
            X: np.ndarray | pd.DataFrame = pd.read_parquet(frame_path)
            for column in metadata["categorical_features"]:
                X[column] = X[column].astype("category")
            X = _normalize_categories(X, name)
        elif array_path.is_file():
            X = np.load(array_path, mmap_mode="r", allow_pickle=False)
        else:
            raise FileNotFoundError(f"Prepared features for {name} are missing")
        y = np.load(required[1], mmap_mode="r", allow_pickle=False)
        split = self._load_optional_array(directory, metadata, "split")
        strata = self._load_optional_array(directory, metadata, "strata")
        groups = self._load_optional_array(directory, metadata, "groups")
        prepared = PreparedDataset(
            X=X,
            y=y,
            feature_names=list(metadata["feature_names"]),
            split=split,
            strata=strata,
            groups=groups,
        )
        validate_prepared_values(spec, prepared)
        return DatasetArrays(
            spec=spec,
            X=X,
            y=y,
            feature_names=prepared.feature_names,
            split=split,
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
            except (FileNotFoundError, ValueError) as error:
                print(f"Rebuilding {name}: {error}", flush=True)
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
        if isinstance(prepared.X, pd.DataFrame):
            categorical_features = list(
                prepared.X.select_dtypes(include="category").columns
            )
            feature_dtypes: str | dict[str, str] = {
                str(name): str(dtype) for name, dtype in prepared.X.dtypes.items()
            }
        else:
            categorical_features = []
            feature_dtypes = str(prepared.X.dtype)
        metadata: dict[str, Any] = {
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
            "categorical_features": categorical_features,
            "registered_categorical_features": list(spec.categorical_features),
            "registered_numeric_features": list(spec.numeric_features),
            "dropped_features": list(spec.drop_features),
            "feature_dtypes": feature_dtypes,
            "label_dtype": str(prepared.y.dtype),
            "has_split": prepared.split is not None,
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
        if prepared.split is not None:
            metadata["split_names"] = ["train", "validation", "test"]
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
