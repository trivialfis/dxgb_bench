# Copyright (c) 2025, Jiaming Yuan.  All rights reserved.
import pytest

from dxgb_bench.testing import has_cuda
from dxgb_bench.utils import query_gpu


@pytest.mark.skipif(reason="No CUDA.", condition=not has_cuda())
def test_query_gpu() -> None:
    res = query_gpu()
    assert res["CUDA version"]
