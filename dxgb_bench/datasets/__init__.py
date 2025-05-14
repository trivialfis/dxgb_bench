import argparse
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .dataset import DataSet


def factory(name: str, args: argparse.Namespace) -> Tuple[DataSet, str]:
    if name.startswith("mortgage"):
        from .mortgage import Mortgage

        d: DataSet = Mortgage(args)
        return d, d.task
    elif name == "taxi":
        from .taxi import Taxi

        d = Taxi(args)
        return d, d.task
    elif name == "higgs":
        from .higgs import Higgs

        d = Higgs(args)
        return d, d.task
    elif name == "year":
        from .year import YearPrediction

        d = YearPrediction(args)
        return d, d.task
    elif name == "covtype":
        from .covtype import Covtype

        d = Covtype(args)
        return d, d.task
    elif name == "airline":
        from .airline import Airline

        d = Airline(args)
        return d, d.task
    elif name == "epsilon":
        from .epsilon import Epsilon

        d = Epsilon(args)
        return d, d.task
    else:
        raise ValueError("Unknown dataset:", name)
