# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
import pytest

from dxgb_bench.testing import Device, devices
from dxgb_bench.utils import machine_info


@pytest.mark.parametrize("device", devices())
def test_machine_info(device: Device) -> None:
    info = machine_info(device)
    assert "system" in info
    assert "arch" in info
    assert "c2c" in info
    assert "gpus" in info
