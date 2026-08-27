"""Reusable public dataset fetching, preparation, and caching pipeline."""

from .models import DatasetArrays, DatasetSpec, PreparedDataset, SplitKind, Task
from .pipeline import (
    CACHE_FORMAT_VERSION,
    DEFAULT_CACHE,
    PublicDatasetPipeline,
    default_cache_dir,
    validate_prepared_values,
)
from .processors import PROCESSORS, Processor, encode_labels, process_source
from .registry import DATASETS

__all__ = [
    "CACHE_FORMAT_VERSION",
    "DATASETS",
    "DEFAULT_CACHE",
    "PROCESSORS",
    "DatasetArrays",
    "DatasetSpec",
    "PreparedDataset",
    "Processor",
    "PublicDatasetPipeline",
    "SplitKind",
    "Task",
    "encode_labels",
    "default_cache_dir",
    "process_source",
    "validate_prepared_values",
]
