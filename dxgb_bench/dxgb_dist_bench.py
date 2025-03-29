import argparse

import numpy as np
import xgboost
from xgboost import collective as coll
from xgboost.tracker import RabitTracker
from .utils import DFT_OUT, add_device_param, add_hyper_param, make_params_from_args

from distributed import Client

def hist_train(args, worker_id: int, tmpdir: str, device: str, rabit_args: dict) -> None:
    """The hist tree method can use a special data structure `ExtMemQuantileDMatrix` for
    faster initialization and lower memory usage.

    """

    # Make sure XGBoost is using RMM for all allocations.
    with coll.CommunicatorContext(**rabit_args), xgboost.config_context(use_rmm=True):
        # Generate the data for demonstration. The sythetic data is sharded by workers.
        files = make_batches(
            n_samples_per_batch=4096,
            n_features=16,
            n_batches=17,
            tmpdir=tmpdir,
            rank=coll.get_rank(),
        )
        # Since we are running two workers on a single node, we should divide the number
        # of threads between workers.
        n_threads = os.cpu_count()
        assert n_threads is not None
        n_threads = max(n_threads // coll.get_world_size(), 1)
        it = Iterator(device, files)
        Xy = xgboost.ExtMemQuantileDMatrix(
            it, missing=np.nan, enable_categorical=False, nthread=n_threads
        )
        params = make_params_from_args(args)
        booster = xgboost.train(
            params,
            Xy,
            evals=[(Xy, "Train")],
            num_boost_round=args.n_rounds,
        )


def cli_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--node", choices=["master", "worker"])
    parser.add_argument("--n_workers", type=int)
    parser.add_argument("--worker-id", type=int)
    parser.add_argument("--tracker-host", type=str)
    parser.add_argument("--tracker-port", type=int)
    parser = add_device_param(parser)
    parser = add_hyper_param(parser)
    args = parser.parse_args()

    if args.node == "master":
        n_workers = args.n_workers
        assert n_workers is not None
        tracker = RabitTracker(host_ip="127.0.0.1", n_workers=n_workers)
        tracker.start()
        rabit_args = tracker.worker_args()
        print(rabit_args)
        tracker.wait_for(300)
    else:
        worker_id = args.worker_id
        tracker_host = args.tracker_host
        tracker_port = args.tracker_port
        assert worker_id is not None
        assert tracker_host is not None
        assert tracker_port is not None
        rabit_args = {
            "dmlc_tracker_uri": tracker_host,
            "dmlc_tracker_port": tracker_port,
        }
        hist_train(worker_id, DFT_OUT, args.device, rabit_args)


if __name__ == "__main__":
    cli_main()
