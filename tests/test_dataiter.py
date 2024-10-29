# Copyright (c) 2024, Jiaming Yuan.  All rights reserved.
from __future__ import annotations

import numpy as np

from dxgb_bench.dataiter import find_shard_ids, get_valid_sizes


def test_shard_ids() -> None:
    a = [0, 3, 3]  # 2 shards
    indptr = np.cumsum(a)
    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 0)
    assert beg_idx == 0
    assert end_idx == 0
    assert beg_in_shard == 0
    assert end_in_shard == 1

    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 1)
    assert beg_idx == 0
    assert end_idx == 0
    assert beg_in_shard == 1
    assert end_in_shard == 2

    a = [0, 5, 5]  # 2 shards
    indptr = np.cumsum(a)
    n_train, n_valid = get_valid_sizes(indptr[-1])
    assert n_train == 8
    assert n_valid == 2
    beg_idx, beg_in_shard, end_idx, end_in_shard = find_shard_ids(indptr, 2)
    assert beg_idx == 0
    assert end_idx == 1
    assert beg_in_shard == 4
    assert end_in_shard == 1
