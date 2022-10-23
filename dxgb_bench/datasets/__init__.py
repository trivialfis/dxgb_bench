import argparse
from typing import Tuple

from dxgb_bench.utils import DataSet

from .airline import Airline
from .covtype import Covtype
from .epsilon import Epsilon
from .generated import Generated
from .higgs import Higgs
from .mortgage import Mortgage
from .taxi import Taxi
from .year import YearPrediction


def factory(name: str, args: argparse.Namespace) -> Tuple[DataSet, str]:
    if name.startswith("mortgage"):
        d: DataSet = Mortgage(args)
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
    elif name == "airline":
        d = Airline(args)
        return d, d.task
    elif name == "epsilon":
        d = Epsilon(args)
        return d, d.task
    elif name == "generated":
        d = Generated(args)
        return d, d.task
    else:
        raise ValueError("Unknown dataset:", name)
