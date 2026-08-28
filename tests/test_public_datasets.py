"""Tests for public dataset fetching, caching, and categorical training."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from dxgb_bench.datasets.public import (
    DATASETS,
    PROCESSORS,
    DatasetSpec,
    PreparedDataset,
    PublicDatasetPipeline,
    default_cache_dir,
)
from dxgb_bench.datasets.public.cli import main as datasets_main


@dataclass
class ToyPipeline:
    pipeline: PublicDatasetPipeline
    upstream: Path
    processor_calls: list[str]


@pytest.fixture
def toy_pipeline(tmp_path: Path) -> ToyPipeline:
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(b"immutable public source\n")
    spec = DatasetSpec(
        name="toy",
        title="Toy classification",
        task="classification",
        source_url=upstream.resolve().as_uri(),
        source_filename="source.bin",
        repository_url="https://example.test/toy",
        rows=4,
        features=2,
        outputs=2,
        split_kind="stratified",
        citation="Synthetic test fixture.",
        license="CC0",
    )
    processor_calls: list[str] = []

    def processor(actual_spec: DatasetSpec, source: Path) -> PreparedDataset:
        assert actual_spec == spec
        assert source.read_bytes() == b"immutable public source\n"
        processor_calls.append(actual_spec.name)
        return PreparedDataset(
            X=np.asarray(
                [[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]],
                dtype=np.float32,
            ),
            y=np.asarray([0, 1, 0, 1], dtype=np.int32),
            feature_names=["first", "second"],
            details={"fixture": True},
        )

    pipeline = PublicDatasetPipeline(
        cache_dir=tmp_path / "cache",
        registry={"toy": spec},
        processors={"toy": processor},
    )
    return ToyPipeline(pipeline, upstream, processor_calls)


def test_registry_has_a_processor_for_every_dataset() -> None:
    assert set(DATASETS) == set(PROCESSORS)
    assert {
        "ames_housing",
        "adult",
        "amazon_employee",
        "airlines",
        "kick",
    } <= set(DATASETS)


def test_ensure_fetches_processes_caches_and_reuses(
    toy_pipeline: ToyPipeline,
) -> None:
    first = toy_pipeline.pipeline.ensure("toy")
    assert toy_pipeline.processor_calls == ["toy"]
    assert first.X.shape == (4, 2)
    assert first.y.tolist() == [0, 1, 0, 1]
    assert first.metadata["class_counts"] == [2, 2]
    assert first.metadata["fixture"] is True
    assert (
        toy_pipeline.pipeline.source_path("toy").read_bytes()
        == b"immutable public source\n"
    )

    second = toy_pipeline.pipeline.ensure("toy", offline=True)
    assert toy_pipeline.processor_calls == ["toy"]
    assert second.metadata["source_sha256"] == first.metadata["source_sha256"]


def test_rebuild_reprocesses_cached_source(toy_pipeline: ToyPipeline) -> None:
    toy_pipeline.pipeline.ensure("toy")
    toy_pipeline.upstream.unlink()
    rebuilt = toy_pipeline.pipeline.ensure("toy", rebuild=True, offline=True)
    assert toy_pipeline.processor_calls == ["toy", "toy"]
    assert rebuilt.X.shape == (4, 2)


def test_offline_fetch_requires_a_cached_source(toy_pipeline: ToyPipeline) -> None:
    with pytest.raises(FileNotFoundError, match="offline mode"):
        toy_pipeline.pipeline.fetch("toy", offline=True)


def test_default_cache_honors_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    configured = tmp_path / "shared-cache"
    monkeypatch.setenv("DXGB_BENCH_DATASET_CACHE", str(configured))
    assert default_cache_dir() == configured


def test_cli_lists_registered_datasets(capsys: pytest.CaptureFixture[str]) -> None:
    datasets_main(["--list"])
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == len(DATASETS)
    assert {line.split("\t", maxsplit=1)[0] for line in lines} == set(DATASETS)


def test_categorical_dataset_trains_xgboost(tmp_path: Path) -> None:
    datasets_main(["--cache-dir", str(tmp_path), "congressional_voting"])
    dataset = PublicDatasetPipeline(cache_dir=tmp_path).load("congressional_voting")
    dtrain = xgb.DMatrix(
        dataset.X,
        label=dataset.y,
        enable_categorical=True,
    )
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "tree_method": "hist",
            "max_depth": 2,
            "nthread": 1,
        },
        dtrain,
        num_boost_round=2,
    )

    assert isinstance(dataset.X, pd.DataFrame)
    assert all(isinstance(dtype, pd.CategoricalDtype) for dtype in dataset.X.dtypes)
    assert set(dataset.X.iloc[:, 0].cat.categories) == {"n", "y"}
    assert (tmp_path / "congressional_voting" / "X.parquet").is_file()
    assert "category_values" not in dataset.metadata
    assert np.isfinite(booster.predict(dtrain)).all()
