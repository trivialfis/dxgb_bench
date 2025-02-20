from contextlib import contextmanager
from enum import IntEnum, unique
from time import time
from typing import Generator, Optional

import tqdm


@contextmanager
def timer(name: str) -> Generator:
    start = time()
    try:
        yield
    finally:
        end = time()
        print(end - start)


@unique
class Method(IntEnum):
    ORD = 0  # Ordinal
    OHE = 1  # One-hot encoding
    NOHE = 2  # native One-hot encoding
    PART = 3  # partition-based


@unique
class Task(IntEnum):
    REG = 0,
    BIN = 1,
    MUL = 2,


def get_meth(algo: str) -> Method:
    meth = algo.split("-")[-1]
    if meth == "ord":
        return Method.ORD
    if meth == "ohe":
        return Method.OHE
    if meth == "nohe":
        return Method.NOHE
    if meth == "part":
        return Method.PART

    raise ValueError("Unknown method.")


class DownloadProgress:
    def __init__(self) -> None:
        self.pbar: Optional[tqdm.tqdm] = None

    def __call__(self, block_num: int, block_size: int, total_size: int) -> None:
        if self.pbar is None:
            self.pbar = tqdm.tqdm(total=total_size / 1024, unit="kB")

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(block_size / 1024)
        else:
            self.pbar.close()
            self.pbar = None
