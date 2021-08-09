from .mortgage import Mortgage
from .taxi import Taxi
from .higgs import Higgs
from .year import YearPrediction


def factory(name: str, args):
    if name.startswith("mortgage"):
        d = Mortgage(args)
        return d.load(args), d.task
    elif name == "taxi":
        d = Taxi(args)
        return d.load(args), d.task
    elif name == "higgs":
        d = Higgs(args)
        return d.load(args), d.task
    elif name == "year":
        d = YearPrediction(args)
        return d.load(args), d.task
    else:
        raise ValueError("Unknown dataset:", name)
