"""Data models shared by the public-dataset pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

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
    """Immutable description of a public source and its prepared representation."""

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

    @property
    def classes(self) -> int:
        """Compatibility alias for classification-only consumers."""
        if self.task != "classification":
            raise AttributeError("classes is only defined for classification datasets")
        return self.outputs


@dataclass(frozen=True)
class PreparedDataset:
    """In-memory result produced from one downloaded public source."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    feature_types: list[str]
    strata: np.ndarray | None = None
    groups: np.ndarray | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetArrays:
    """Validated, memory-mapped arrays loaded from the prepared cache."""

    spec: DatasetSpec
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    feature_types: list[str]
    strata: np.ndarray | None
    groups: np.ndarray | None
    metadata: dict[str, Any]
