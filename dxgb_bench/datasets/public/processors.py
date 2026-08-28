"""Dataset-specific transformations from public sources."""

from __future__ import annotations

import io
import tarfile
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import DatasetSpec, PreparedDataset
from .registry import DATASETS

Processor = Callable[[DatasetSpec, Path], PreparedDataset]


def _read_zip_csv(archive: zipfile.ZipFile, member: str, **kwargs: Any) -> pd.DataFrame:
    with archive.open(member) as source:
        return pd.read_csv(source, **kwargs)


def _coordinate_group_ids(coordinates: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Identify equal model-input coordinates without crossing scenarios."""
    coordinates = np.ascontiguousarray(coordinates, dtype=np.float32)
    row_dtype = np.dtype((np.void, coordinates.dtype.itemsize * coordinates.shape[1]))
    row_keys = coordinates.view(row_dtype).reshape(-1)
    groups = np.empty(coordinates.shape[0], dtype=np.int32)
    offset = 0
    for stratum in np.unique(strata):
        indices = np.flatnonzero(strata == stratum)
        _, inverse = np.unique(row_keys[indices], return_inverse=True)
        groups[indices] = inverse.astype(np.int32) + offset
        offset += int(inverse.max()) + 1
    return groups


def _prepare_sarcos(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    frame = pd.read_parquet(source)
    feature_names = [f"V{index}" for index in range(1, 22)]
    target_names = [f"V{index}" for index in range(22, 29)]
    return PreparedDataset(
        X=frame[feature_names].to_numpy(dtype=np.float32, copy=True),
        y=frame[target_names].to_numpy(dtype=np.float32, copy=True),
        feature_names=feature_names,
        details={
            "target_names": target_names,
            "raw_columns": list(frame.columns),
        },
    )


def _prepare_wave_energy(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    scenarios = ["Adelaide", "Perth", "Sydney", "Tasmania"]
    frames = []
    labels = []
    with zipfile.ZipFile(source) as archive:
        for code, scenario in enumerate(scenarios):
            member = f"WECs_DataSet/{scenario}_Data.csv"
            frame = _read_zip_csv(archive, member, header=None)
            expected_rows = 71_999 if scenario == "Adelaide" else 72_000
            if frame.shape != (expected_rows, 49):
                raise ValueError(f"Unexpected {member} shape: {frame.shape}")
            frames.append(frame)
            labels.append(np.full(frame.shape[0], code, dtype=np.int32))

    combined = pd.concat(frames, ignore_index=True)
    strata = np.concatenate(labels)
    coordinates = combined.iloc[:, :32].to_numpy(dtype=np.float32, copy=True)
    groups = _coordinate_group_ids(coordinates, strata)
    feature_names = (
        [f"X{index}" for index in range(1, 17)]
        + [f"Y{index}" for index in range(1, 17)]
        + ["scenario"]
    )
    target_names = [f"Power{index}" for index in range(1, 17)]
    features = pd.DataFrame(coordinates, columns=feature_names[:-1])
    features["scenario"] = pd.Series(
        np.concatenate(
            [np.repeat(name, frame.shape[0]) for name, frame in zip(scenarios, frames)]
        ),
        dtype="category",
    )
    return PreparedDataset(
        X=features,
        y=combined.iloc[:, 32:48].to_numpy(dtype=np.float32, copy=True),
        feature_names=feature_names,
        strata=strata,
        groups=groups,
        details={
            "target_names": target_names,
            "scenarios": scenarios,
            "excluded_columns": ["Total_Power"],
            "rows_per_scenario": {
                name: 71_999 if name == "Adelaide" else 72_000 for name in scenarios
            },
            "coordinate_groups": int(np.unique(groups).size),
            "source_metadata_discrepancy": (
                "The current UCI archive contains 71,999 Adelaide rows, one fewer "
                "than the 72,000 reported in the repository description."
            ),
        },
    )


def _prepare_large_wave_energy(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    scenarios = ["Perth", "Sydney"]
    frames = []
    labels = []
    with zipfile.ZipFile(source) as outer, outer.open("WEC.zip") as nested_source:
        nested_bytes = nested_source.read()
    with zipfile.ZipFile(io.BytesIO(nested_bytes)) as archive:
        for code, scenario in enumerate(scenarios):
            member = f"WEC/WEC_{scenario}_49.csv"
            frame = _read_zip_csv(archive, member)
            frames.append(frame)
            labels.append(np.full(frame.shape[0], code, dtype=np.int32))

    combined = pd.concat(frames, ignore_index=True)
    strata = np.concatenate(labels)
    coordinate_names = [name for name in combined if name.startswith(("X", "Y"))]
    power_names = [name for name in combined if name.startswith("Power")]
    if len(coordinate_names) != 98 or len(power_names) != 49:
        raise ValueError(
            "Unexpected large wave-energy schema: "
            f"{len(coordinate_names)} coordinates, {len(power_names)} powers"
        )
    coordinates = combined[coordinate_names].to_numpy(dtype=np.float32, copy=True)
    groups = _coordinate_group_ids(coordinates, strata)
    features = pd.DataFrame(coordinates, columns=coordinate_names)
    features["scenario"] = pd.Series(
        np.concatenate(
            [np.repeat(name, frame.shape[0]) for name, frame in zip(scenarios, frames)]
        ),
        dtype="category",
    )
    return PreparedDataset(
        X=features,
        y=combined[power_names].to_numpy(dtype=np.float32, copy=True),
        feature_names=coordinate_names + ["scenario"],
        strata=strata,
        groups=groups,
        details={
            "target_names": power_names,
            "scenarios": scenarios,
            "excluded_columns": ["qW", "Total_Power"],
            "rows_per_scenario": {
                scenario: int(frame.shape[0])
                for scenario, frame in zip(scenarios, frames, strict=True)
            },
            "ignored_duplicate_archive_member": "WEC_Perth_49.csv",
            "coordinate_groups": int(np.unique(groups).size),
        },
    )


def _prepare_tetouan(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    member = "Tetuan City power consumption.csv"
    with zipfile.ZipFile(source) as archive:
        frame = _read_zip_csv(archive, member)
    timestamp = pd.to_datetime(frame.pop("DateTime"), format="%m/%d/%Y %H:%M")
    if not timestamp.is_monotonic_increasing or not timestamp.is_unique:
        raise ValueError("Tetouan timestamps must be unique and increasing")

    target_names = [name for name in frame if name.startswith("Zone ")]
    target = frame.pop(target_names[0]).to_frame()
    for name in target_names[1:]:
        target[name] = frame.pop(name)
    elapsed_days = (timestamp - timestamp.min()).dt.total_seconds() / 86_400.0
    calendar = pd.DataFrame(
        {
            "elapsed_days": elapsed_days.astype(np.float32),
            "month": (timestamp.dt.month - 1).astype("category"),
            "day_of_week": timestamp.dt.dayofweek.astype("category"),
            "time_slot": (
                timestamp.dt.hour * 6 + timestamp.dt.minute.floordiv(10)
            ).astype("category"),
        }
    )
    features = pd.concat([frame.reset_index(drop=True), calendar], axis=1)
    return PreparedDataset(
        X=features,
        y=target.to_numpy(dtype=np.float32, copy=True),
        feature_names=list(features.columns),
        details={
            "target_names": target_names,
            "timestamp_start": timestamp.min().isoformat(),
            "timestamp_end": timestamp.max().isoformat(),
            "calendar_features": ["month", "day_of_week", "time_slot"],
            "source_metadata_discrepancy": (
                "The current UCI archive contains 52,416 data rows; the repository "
                "description's 52,417 count includes the CSV header."
            ),
        },
    )


def _prepare_sgemm(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    with zipfile.ZipFile(source) as archive:
        frame = _read_zip_csv(archive, "sgemm_product.csv")
    target_names = [f"Run{index} (ms)" for index in range(1, 5)]
    targets = frame.pop(target_names[0]).to_frame()
    for name in target_names[1:]:
        targets[name] = frame.pop(name)
    return PreparedDataset(
        X=frame.to_numpy(dtype=np.float32, copy=True),
        y=np.log1p(targets.to_numpy(dtype=np.float32, copy=True)),
        feature_names=list(frame.columns),
        details={
            "target_names": target_names,
            "target_transform": "log1p milliseconds",
        },
    )


def _prepare_rf1(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    frame = pd.read_parquet(source)
    target_names = [name for name in frame if "_48H__" in name]
    if len(target_names) != 8:
        raise ValueError(f"Expected eight RF1 targets, found {target_names}")
    targets = frame[target_names]
    features = frame.drop(columns=target_names)
    X = features.to_numpy(dtype=np.float32, copy=True)
    return PreparedDataset(
        X=X,
        y=targets.to_numpy(dtype=np.float32, copy=True),
        feature_names=list(features.columns),
        details={
            "target_names": target_names,
            "forecast_horizon_rows": 48,
            "feature_missing_values": int(np.isnan(X).sum()),
            "missing_value_handling": "XGBoost native missing-value routing",
        },
    )


def _prepare_uji_indoor_loc(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    feature_names = [f"WAP{index:03d}" for index in range(1, 521)]
    target_names = ["LONGITUDE", "LATITUDE"]
    with zipfile.ZipFile(source) as archive:
        training = _read_zip_csv(archive, "UJIndoorLoc/trainingData.csv")
        official_test = _read_zip_csv(archive, "UJIndoorLoc/validationData.csv")
    frame = pd.concat([training, official_test], ignore_index=True)
    split = np.concatenate(
        [
            np.zeros(training.shape[0], dtype=np.int32),
            np.full(official_test.shape[0], 2, dtype=np.int32),
        ]
    )
    excluded = [
        "FLOOR",
        "BUILDINGID",
        "SPACEID",
        "RELATIVEPOSITION",
        "USERID",
        "PHONEID",
        "TIMESTAMP",
    ]
    return PreparedDataset(
        X=frame[feature_names].to_numpy(dtype=np.float32, copy=True),
        y=frame[target_names].to_numpy(dtype=np.float32, copy=True),
        feature_names=feature_names,
        split=split,
        details={
            "target_names": target_names,
            "train_pool_rows": int(training.shape[0]),
            "official_test_rows": int(official_test.shape[0]),
            "excluded_columns": excluded,
            "official_split": (
                "The supplied training table is split 80/20 for fitting and "
                "validation; the supplied validation table is held out for test."
            ),
        },
    )


def _prepare_covertype(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    frame = pd.read_csv(source)
    target = frame.pop("Cover_Type")
    return PreparedDataset(
        X=frame.to_numpy(dtype=np.float32, copy=True),
        y=target,
        feature_names=list(frame.columns),
        details={
            "provider_encoding": (
                "Wilderness area and soil type indicators remain provider-supplied "
                "numeric one-hot columns."
            ),
        },
    )


def _prepare_poker(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    frame = pd.read_csv(source)
    target = frame.pop("CLASS")
    frame = frame.astype("category")
    return PreparedDataset(
        X=frame,
        y=target,
        feature_names=list(frame.columns),
    )


def _prepare_sensorless(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    with zipfile.ZipFile(source) as archive:
        frame = _read_zip_csv(
            archive,
            "Sensorless_drive_diagnosis.txt",
            sep=r"\s+",
            header=None,
        )
    target = frame.pop(frame.columns[-1])
    feature_names = [f"V{index}" for index in range(1, 49)]
    return PreparedDataset(
        X=frame.to_numpy(dtype=np.float32, copy=True),
        y=target,
        feature_names=feature_names,
    )


def _prepare_letter_recognition(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    feature_names = [
        "x_box",
        "y_box",
        "width",
        "height",
        "on_pixels",
        "x_mean",
        "y_mean",
        "x_variance",
        "y_variance",
        "xy_correlation",
        "x2y_mean",
        "xy2_mean",
        "x_edge_mean",
        "x_edge_correlation",
        "y_edge_mean",
        "y_edge_correlation",
    ]
    with zipfile.ZipFile(source) as archive:
        frame = _read_zip_csv(
            archive,
            "letter-recognition.data",
            names=["letter", *feature_names],
            header=None,
        )
    target = frame.pop("letter")
    return PreparedDataset(
        X=frame.to_numpy(dtype=np.float32, copy=True),
        y=target,
        feature_names=feature_names,
        split=np.concatenate(
            [
                np.zeros(16_000, dtype=np.int32),
                np.full(4_000, 2, dtype=np.int32),
            ]
        ),
        details={
            "train_pool_rows": 16_000,
            "official_test_rows": 4_000,
            "official_split": (
                "The first 16,000 provider-ordered rows form the training pool; "
                "the final 4,000 rows are held out for test."
            ),
        },
    )


def _prepare_gas_sensor_drift(spec: DatasetSpec, source: Path) -> PreparedDataset:
    del spec
    feature_names = [f"feature_{index:03d}" for index in range(1, 129)]
    feature_batches = []
    labels: list[int] = []
    batch_sizes = []
    with zipfile.ZipFile(source) as archive:
        for batch in range(1, 11):
            rows = []
            with archive.open(f"Dataset/batch{batch}.dat") as member:
                for raw_line in member:
                    fields = raw_line.decode("ascii").split()
                    labels.append(int(fields[0]))
                    row = np.zeros(128, dtype=np.float32)
                    for field in fields[1:]:
                        raw_index, raw_value = field.split(":", maxsplit=1)
                        row[int(raw_index) - 1] = float(raw_value)
                    rows.append(row)
            batch_array = np.vstack(rows)
            feature_batches.append(batch_array)
            batch_sizes.append(int(batch_array.shape[0]))

    split_rows = {
        "train": sum(batch_sizes[:6]),
        "validation": sum(batch_sizes[6:8]),
        "test": sum(batch_sizes[8:]),
    }
    return PreparedDataset(
        X=np.vstack(feature_batches),
        y=np.asarray(labels),
        feature_names=feature_names,
        split=np.concatenate(
            [
                np.full(rows, code, dtype=np.int32)
                for code, rows in enumerate(split_rows.values())
            ]
        ),
        details={
            "rows_per_batch": batch_sizes,
            "predefined_split_rows": split_rows,
            "predefined_split": (
                "Chronological batches 1-6 train, 7-8 validate, and 9-10 test."
            ),
        },
    )


def _prepare_openml_classification(spec: DatasetSpec, source: Path) -> PreparedDataset:
    target_names = {
        "devnagari_script": "character",
        "emnist_balanced": "class",
        "kuzushiji_49": "class",
        "dionis": "class",
        "aloi": "target",
    }
    target_name = target_names[spec.name]
    frame = pd.read_parquet(source)
    target = frame.pop(target_name)
    return PreparedDataset(
        X=frame.to_numpy(dtype=np.float32, copy=True),
        y=target,
        feature_names=list(frame.columns),
    )


def _prepare_categorical_frame(
    spec: DatasetSpec, frame: pd.DataFrame
) -> PreparedDataset:
    if spec.target is None:
        raise ValueError(f"{spec.name}: no target column is registered")

    target = frame.pop(spec.target)
    frame = frame.drop(columns=list(spec.drop_features))
    valid_rows = target.notna()
    frame = frame.loc[valid_rows].reset_index(drop=True)
    target = target.loc[valid_rows].reset_index(drop=True)

    for name in spec.numeric_features:
        frame[name] = pd.to_numeric(frame[name], errors="coerce")

    if spec.categorical_features == ("*",):
        categorical = set(frame.columns)
    else:
        categorical = set(spec.categorical_features)
        categorical.update(
            name
            for name, dtype in frame.dtypes.items()
            if not pd.api.types.is_numeric_dtype(dtype)
        )

    for name in categorical:
        frame[name] = frame[name].astype("category")

    return PreparedDataset(
        X=frame,
        y=target,
        feature_names=[str(name) for name in frame.columns],
    )


def _prepare_categorical_table(spec: DatasetSpec, source: Path) -> PreparedDataset:
    if source.suffix == ".parquet":
        frame = pd.read_parquet(source)
    else:
        frame = pd.read_csv(source)
    return _prepare_categorical_frame(spec, frame)


def _prepare_south_german_credit(spec: DatasetSpec, source: Path) -> PreparedDataset:
    with zipfile.ZipFile(source) as archive:
        frame = _read_zip_csv(
            archive,
            "SouthGermanCredit.asc",
            sep=r"\s+",
        )
    return _prepare_categorical_frame(spec, frame)


def _prepare_audiology(spec: DatasetSpec, source: Path) -> PreparedDataset:
    with zipfile.ZipFile(source) as archive:
        members = sorted(
            name for name in archive.namelist() if name.endswith((".data", ".test"))
        )
        frames = [
            _read_zip_csv(archive, name, header=None, na_values="?") for name in members
        ]
        names = archive.read("audiology.standardized.names").decode("utf-8")
    section = names.split("7. Attribute information:", maxsplit=1)[1].split(
        "class:", maxsplit=1
    )[0]
    columns = [
        line.split(":", maxsplit=1)[0].strip().removesuffix("()")
        for line in section.splitlines()
        if ":" in line
    ]
    frame = pd.concat(frames, ignore_index=True)
    frame.columns = [*columns, "identifier", spec.target]
    prepared = _prepare_categorical_frame(spec, frame)
    return PreparedDataset(
        X=prepared.X,
        y=prepared.y,
        feature_names=prepared.feature_names,
        split=np.concatenate(
            [
                np.full(
                    part.shape[0],
                    0 if member.endswith(".data") else 2,
                    dtype=np.int32,
                )
                for member, part in zip(members, frames, strict=True)
            ]
        ),
        details={"official_train_rows": 200, "official_test_rows": 26},
    )


_CENSUS_INCOME_COLUMNS = (  # noqa: SIM905
    "AAGE ACLSWKR ADTINK ADTOCC AHGA AHRSPAY AHSCOL AMARITL AMJIND AMJOCC "
    "ARACE AREORGN ASEX AUNMEM AUNTYPE AWKSTAT CAPGAIN GAPLOSS DIVVAL FILESTAT "
    "GRINREG GRINST HHDFMX HHDREL MARSUPWRT MIGMTR1 MIGMTR3 MIGMTR4 MIGSAME "
    "MIGSUN NOEMP PARENT PEFNTVTY PEMNTVTY PENATVTY PRCITSHP SEOTR VETQVA "
    "VETYN WKSWORK year income"
).split()


def _prepare_census_income(spec: DatasetSpec, source: Path) -> PreparedDataset:
    with zipfile.ZipFile(source) as archive:
        nested = next(name for name in archive.namelist() if name.endswith(".tar.gz"))
        with tarfile.open(fileobj=io.BytesIO(archive.read(nested)), mode="r:gz") as tar:
            members = sorted(
                (
                    member
                    for member in tar.getmembers()
                    if member.name.endswith((".data", ".test"))
                ),
                key=lambda member: member.name,
            )
            frames = [
                pd.read_csv(
                    tar.extractfile(member),
                    header=None,
                    names=_CENSUS_INCOME_COLUMNS,
                    na_values="?",
                    skipinitialspace=True,
                )
                for member in members
            ]
    frame = pd.concat(frames, ignore_index=True)
    prepared = _prepare_categorical_frame(spec, frame)
    return PreparedDataset(
        X=prepared.X,
        y=prepared.y,
        feature_names=prepared.feature_names,
        split=np.concatenate(
            [
                np.full(
                    part.shape[0],
                    0 if member.name.endswith(".data") else 2,
                    dtype=np.int32,
                )
                for member, part in zip(members, frames, strict=True)
            ]
        ),
        details={"official_train_rows": 199_523, "official_test_rows": 99_762},
    )


def _prepare_monks(spec: DatasetSpec, source: Path) -> PreparedDataset:
    problem = spec.name.removeprefix("monks_")
    with zipfile.ZipFile(source) as archive:
        members = sorted(
            (
                name
                for name in archive.namelist()
                if name.endswith((f"monks-{problem}.train", f"monks-{problem}.test"))
            ),
            key=lambda name: not name.endswith(".train"),
        )
        frames = [
            _read_zip_csv(
                archive,
                name,
                sep=r"\s+",
                header=None,
                names=["class", "a1", "a2", "a3", "a4", "a5", "a6", "ID"],
            )
            for name in members
        ]
    frame = pd.concat(frames, ignore_index=True)
    prepared = _prepare_categorical_frame(spec, frame)
    return PreparedDataset(
        X=prepared.X,
        y=prepared.y,
        feature_names=prepared.feature_names,
        split=np.concatenate(
            [
                np.full(
                    part.shape[0],
                    0 if member.endswith(".train") else 2,
                    dtype=np.int32,
                )
                for member, part in zip(members, frames, strict=True)
            ]
        ),
        details={
            "official_train_rows": int(frames[0].shape[0]),
            "official_test_rows": int(frames[1].shape[0]),
        },
    )


PROCESSORS: dict[str, Processor] = {
    "sarcos": _prepare_sarcos,
    "wave_energy": _prepare_wave_energy,
    "large_wave_energy": _prepare_large_wave_energy,
    "tetouan_power": _prepare_tetouan,
    "sgemm": _prepare_sgemm,
    "rf1": _prepare_rf1,
    "uji_indoor_loc": _prepare_uji_indoor_loc,
    "covertype": _prepare_covertype,
    "poker_hand": _prepare_poker,
    "sensorless_drive": _prepare_sensorless,
    "letter_recognition": _prepare_letter_recognition,
    "gas_sensor_drift": _prepare_gas_sensor_drift,
    "devnagari_script": _prepare_openml_classification,
    "emnist_balanced": _prepare_openml_classification,
    "kuzushiji_49": _prepare_openml_classification,
    "dionis": _prepare_openml_classification,
    "aloi": _prepare_openml_classification,
}

PROCESSORS.update(
    {
        name: _prepare_categorical_table
        for name, spec in DATASETS.items()
        if spec.target is not None
    }
)
PROCESSORS["audiology"] = _prepare_audiology
PROCESSORS["census_income_uci"] = _prepare_census_income
for problem in range(1, 4):
    PROCESSORS[f"monks_{problem}"] = _prepare_monks
PROCESSORS["south_german_credit"] = _prepare_south_german_credit


def process_source(spec: DatasetSpec, source: Path) -> PreparedDataset:
    """Run the registered processor for one downloaded source."""
    try:
        processor = PROCESSORS[spec.name]
    except KeyError as error:
        raise KeyError(f"No source processor registered for {spec.name!r}") from error
    return processor(spec, source)
