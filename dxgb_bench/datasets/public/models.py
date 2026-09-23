"""Data models shared by the public-dataset pipeline."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

Task = Literal["regression", "classification"]
SplitKind = Literal[
    "blocked",
    "official_test",
    "predefined",
    "purged_blocked",
    "random",
    "stratified",
    "stratified_group",
]


@dataclass(frozen=True)
class DatasetSpec:
    """Public source, prepared representation, and its preparation callable."""

    name: str
    title: str
    task: Task
    source_url: str
    source_filename: str
    repository_url: str
    rows: int
    features: int
    outputs: int
    split_kind: SplitKind
    citation: str
    license: str
    prepare: Processor
    target: str | None = None
    categorical_features: tuple[str, ...] = ()
    numeric_features: tuple[str, ...] = ()
    drop_features: tuple[str, ...] = ()
    feature_columns: tuple[str, ...] = ()
    target_columns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return serializable dataset metadata, excluding the preparation callable."""
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "prepare"
        }

    @property
    def classes(self) -> int:
        """Compatibility alias for classification-only consumers."""
        if self.task != "classification":
            raise AttributeError("classes is only defined for classification datasets")
        return self.outputs


@dataclass(frozen=True)
class PreparedDataset:
    """In-memory result produced from one downloaded public source."""

    X: np.ndarray | pd.DataFrame
    y: np.ndarray | pd.Series[Any]
    feature_names: list[str]
    split: np.ndarray | None = None
    strata: np.ndarray | None = None
    groups: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


Processor = Callable[[DatasetSpec, Path], PreparedDataset]


@dataclass(frozen=True)
class DatasetArrays:
    """Validated data loaded from the prepared cache."""

    spec: DatasetSpec
    X: np.ndarray | pd.DataFrame
    y: np.ndarray
    feature_names: list[str]
    split: np.ndarray | None
    strata: np.ndarray | None
    groups: np.ndarray | None
    metadata: dict[str, Any]
