from .mortgage import Mortgage
from .taxi import Taxi
from .higgs import Higgs
from .year import YearPrediction
from .covtype import Covtype

import argparse
from typing import Tuple
from dxgb_bench.utils import DataSet


def factory(name: str, args: argparse.Namespace) -> Tuple[DataSet, str]:
    if name.startswith("mortgage"):
        d = Mortgage(args)
        return d, d.task
    elif name == "taxi":
        d = Taxi(args)
        return d, d.task
    elif name == "higgs":
        d = Higgs(args)
        return d, d.task
    elif name == "year":
        d = YearPrediction(args)
        return d, d.task
    elif name == "covtype":
        d = Covtype(args)
        return d, d.task
    else:
        raise ValueError("Unknown dataset:", name)
