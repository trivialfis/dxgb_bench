"""Public dataset contracts, cache lifecycle, and training integration."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from dxgb_bench.datasets.public import (
    DATASETS,
    DatasetSpec,
    PreparedDataset,
    PublicDatasetPipeline,
    default_cache_dir,
    validate_prepared_values,
)
from dxgb_bench.datasets.public.cli import main as datasets_main


@pytest.fixture
def pipeline(tmp_path: Path) -> PublicDatasetPipeline:
    source = tmp_path / "source.bin"
    source.write_bytes(b"public source")

    def prepare(spec: DatasetSpec, path: Path) -> PreparedDataset:
        assert path.read_bytes() == b"public source"
        return PreparedDataset(
            X=pd.DataFrame(
                {
                    "first": pd.Series([0.0, 1.0, np.nan, 1.0], dtype="category"),
                    "second": np.arange(4, dtype=np.float32),
                }
            ),
            y=np.array([0, 1, 0, 1]),
            feature_names=["first", "second"],
            split=np.array([0, 0, 2, 2]),
        )

    spec = replace(
        DATASETS["congressional_voting"],
        name="toy",
        rows=4,
        features=2,
        source_url=source.as_uri(),
        source_filename=source.name,
        split_kind="official_test",
        prepare=prepare,
    )
    return PublicDatasetPipeline(tmp_path / "cache", registry={"toy": spec})


def test_registry_and_cli(capsys: pytest.CaptureFixture[str]) -> None:
    for name, spec in DATASETS.items():
        assert name == spec.name
        assert callable(spec.prepare)
        metadata = spec.to_dict()
        assert "prepare" not in metadata
        assert json.loads(json.dumps(metadata))["name"] == name
    datasets_main(["--list"])
    listed = [line.split("\t")[0] for line in capsys.readouterr().out.splitlines()]
    assert len(listed) == len(DATASETS)
    assert set(listed) == set(DATASETS)


def test_cache_lifecycle(pipeline: PublicDatasetPipeline, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="offline mode"):
        pipeline.fetch("toy", offline=True)
    with patch.object(pipeline, "process", wraps=pipeline.process) as process:
        first = pipeline.ensure("toy")
        assert first.X["first"].cat.categories.tolist() == [0, 1]
        assert pd.api.types.is_integer_dtype(first.X["first"].cat.categories.dtype)
        assert pd.isna(first.X["first"].iloc[2])
        np.testing.assert_array_equal(first.y, [0, 1, 0, 1])
        np.testing.assert_array_equal(first.split, [0, 0, 2, 2])
        assert first.metadata["class_counts"] == [2, 2]
        assert pipeline.source_path("toy").read_bytes() == b"public source"

        # Reuse an older cache, then explicitly rebuild from its retained source.
        metadata_path = pipeline.dataset_dir("toy") / "metadata.json"
        metadata = json.loads(metadata_path.read_text())
        del metadata["feature_columns"], metadata["target_columns"]
        metadata_path.write_text(json.dumps(metadata))
        (tmp_path / "source.bin").unlink()
        cached = pipeline.ensure("toy", offline=True)
        assert process.call_count == 1
        assert cached.metadata["source_sha256"] == first.metadata["source_sha256"]
        pd.testing.assert_frame_equal(cached.X, first.X)
        np.testing.assert_array_equal(cached.y, first.y)
        rebuilt = pipeline.ensure("toy", rebuild=True, offline=True)
        assert process.call_count == 2
        pd.testing.assert_frame_equal(rebuilt.X, first.X)
        np.testing.assert_array_equal(rebuilt.y, first.y)


@pytest.mark.parametrize(
    "field,value",
    [
        ("split_kind", "predefined"),
        ("feature_columns", ("second", "first")),
        ("target_columns", ("label",)),
    ],
)
def test_cache_rejects_changed_semantics(
    pipeline: PublicDatasetPipeline, field: str, value: str | tuple[str, ...]
) -> None:
    pipeline.ensure("toy")
    spec = replace(pipeline.spec("toy"), **{field: value})
    changed = PublicDatasetPipeline(pipeline.cache_dir, registry={"toy": spec})
    with pytest.raises(ValueError, match=field):
        changed.load("toy")


def test_empty_category(pipeline: PublicDatasetPipeline, tmp_path: Path) -> None:
    prepared = pipeline.process("toy", tmp_path / "source.bin")
    prepared.X["first"] = pd.Series([None] * 4, dtype="category")
    with pytest.raises(ValueError, match="categorical feature 'first' is empty"):
        validate_prepared_values(pipeline.spec("toy"), prepared)


@pytest.mark.parametrize(
    "name,features,targets",
    [
        ("sarcos", list(range(1, 22)), list(range(22, 29))),
        ("custom", [3, 1], [28, 22]),
    ],
)
def test_parquet_column_selection(
    tmp_path: Path, name: str, features: list[int], targets: list[int]
) -> None:
    values = np.arange(84, dtype=np.float64).reshape(3, 28)
    frame = pd.DataFrame(values, columns=[f"V{i}" for i in range(1, 29)])
    frame["unused"] = -1.0
    source = tmp_path / "source.parquet"
    frame[frame.columns[::-1]].to_parquet(source, index=False)
    spec = replace(DATASETS["sarcos"], rows=3, source_url=source.as_uri())
    if name == "custom":
        spec = replace(
            spec,
            name=name,
            features=len(features),
            outputs=len(targets),
            feature_columns=tuple(f"V{i}" for i in features),
            target_columns=tuple(f"V{i}" for i in targets),
        )
    dataset = PublicDatasetPipeline(tmp_path / "cache", registry={name: spec}).ensure(
        name
    )
    np.testing.assert_array_equal(dataset.X, values[:, np.array(features) - 1])
    np.testing.assert_array_equal(dataset.y, values[:, np.array(targets) - 1])
    assert dataset.X.dtype == dataset.y.dtype == np.float32
    assert dataset.feature_names == [f"V{i}" for i in features]
    assert dataset.metadata["target_names"] == [f"V{i}" for i in targets]


def test_default_cache_honors_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DXGB_BENCH_DATASET_CACHE", str(tmp_path))
    assert default_cache_dir() == tmp_path


def test_categorical_cli_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = replace(DATASETS["congressional_voting"], rows=4)
    monkeypatch.setitem(DATASETS, spec.name, spec)
    pipeline = PublicDatasetPipeline(tmp_path)
    source = pipeline.source_path(spec.name)
    source.parent.mkdir()
    frame = pd.DataFrame({f"vote{i}": ["n", "y", None, "n"] for i in range(16)})
    frame[spec.target] = ["democrat", "republican", "democrat", "republican"]
    frame.to_csv(source, index=False)
    datasets_main(["--cache-dir", str(tmp_path), "--offline", spec.name])
    dataset = pipeline.load(spec.name)
    assert all(isinstance(dtype, pd.CategoricalDtype) for dtype in dataset.X.dtypes)
    assert set(dataset.X.iloc[:, 0].cat.categories) == {"n", "y"}
    assert (pipeline.dataset_dir(spec.name) / "X.parquet").is_file()
    dtrain = xgb.DMatrix(dataset.X, label=dataset.y, enable_categorical=True)
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
    assert np.isfinite(booster.predict(dtrain)).all()
