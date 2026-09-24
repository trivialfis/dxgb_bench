# Copyright (c) 2025, Jiaming Yuan. All rights reserved.
from __future__ import annotations

import json
from pathlib import Path

import pytest
import xgboost as xgb

from dxgb_bench.dxgb_bench import bench, cli_main, datagen
from dxgb_bench.testing import Device, devices
from dxgb_bench.utils import Opts, Timer


@pytest.fixture(autouse=True)
def benchmark_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    Timer.reset()


@pytest.mark.parametrize("device", devices())
@pytest.mark.parametrize(
    "task,fly,valid",
    [
        ("qdm", False, True),
        ("qdm-iter", False, True),
        ("ext-qdm-iter", False, True),
        ("ext-dm-iter", False, True),
        ("qdm-iter", True, False),
        ("ext-qdm-iter", True, False),
    ],
)
def test_bench(device: Device, task: str, fly: bool, valid: bool) -> None:
    opts = Opts(
        n_samples_per_batch=64,
        n_features=8,
        n_targets=1,
        n_batches=3,
        sparsity=0.2,
        on_the_fly=fly,
        validation=valid,
        device=device,
        mr=None,
        target_type="reg",
        cache_host_ratio=None,
        min_cache_page_bytes=0,
    )
    outdirs = ["data-0", "data-1"]
    if not fly:
        datagen(
            n_samples_per_batch=opts.n_samples_per_batch,
            n_features=opts.n_features,
            n_targets=opts.n_targets,
            n_batches=opts.n_batches,
            assparse=False,
            target_type=opts.target_type,
            sparsity=opts.sparsity,
            device=device,
            outdirs=outdirs,
            fmt="npy",
        )
        # Stored data determines the shape and batch count.
        opts.n_batches = 1
        opts.n_features = 512
        opts.n_samples_per_batch = 0
    booster, results = bench(
        task=task,
        loadfrom=outdirs,
        model_path=None,
        params={"device": device, "max_bin": 8, "max_depth": 2},
        opts=opts,
        n_rounds=2,
    )
    assert booster.num_features() == 8
    assert booster.num_boosted_rounds() == 2
    assert results["opts"]["n_batches"] == (1 if task == "qdm" else 3)
    assert results["opts"]["n_samples_per_batch"] == (192 if task == "qdm" else 64)
    assert set(results["evals"]) == ({"Train", "Valid"} if valid else {"Train"})


def test_bench_cli() -> None:
    cli_main(
        [
            "bench",
            "--task=ext-dm-iter",
            "--device=cpu",
            "--tree_method=approx",
            "--fly",
            "--n_samples_per_batch=32",
            "--n_batches=2",
            "--target_type=bin",
            "--n_rounds=2",
            "--max_depth=2",
            "--model_path=model.json",
        ]
    )
    saved = json.loads(Path("extmem-0.json").read_text())
    assert saved["opts"]["task"] == "ext-dm-iter"
    assert saved["opts"]["n_features"] == 512
    assert len(saved["evals"]["Train"]["logloss"]) == 2
    assert saved["timer"]["Train"]["Total"] >= saved["timer"]["Train"]["Train"] > 0
    assert xgb.Booster(model_file="model.json").num_boosted_rounds() == 2


@pytest.mark.parametrize(
    "args,message",
    [
        (["--task=qdm", "--fly"], "--fly requires an iterator task"),
        (["--task=ext-qdm-iter", "--tree_method=approx"], "requires --tree_method"),
        (["--task=qdm-iter", "--cache_host_ratio=0.5"], "--cache_host_ratio requires"),
    ],
)
def test_bench_invalid_args(
    args: list[str], message: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as exc:
        cli_main(["bench", *args])
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
