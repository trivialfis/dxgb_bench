import argparse
import psutil
import os

# import dask_ml.metrics as dm
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

from dxgb_bench.datasets import factory as data_factory
from dxgb_bench.utils import Timer, fprint, TemporaryDirectory
from dxgb_bench import algorihm

import dask
import pandas
import distributed
import cudf
import dask_cudf
import dask_cuda
import xgboost
import cupy

import json


def packages_version():
    packages = {
        'dask': dask.__version__,
        'pandas': pandas.__version__,
        'distributed': distributed.__version__,
        'cudf': cudf.__version__,
        'dask_cudf': dask_cudf.__version__,
        'dask_cuda': dask_cuda.__version__,
        'xgboost': xgboost.__version__,
        'cupy': cupy.__version__
    }
    return packages


def print_version():
    fprint('Package version:')
    packages = packages_version()
    for name, version in packages.items():
        fprint('- ' + name + ':', version)
    fprint()


def main(args):
    print_version()
    dask.config.set({'temporary_directory': args.temporary_directory})
    if not os.path.exists(args.temporary_directory):
        os.mkdir(args.temporary_directory)

    with TemporaryDirectory(args.temporary_directory):
        with LocalCUDACluster(threads_per_worker=args.cpus) as cluster:
            print('dashboard link:', cluster.dashboard_link)
            with Client(cluster) as client:
                (X, y, w), task = data_factory(args.data, args)
                algo = algorihm.factory(args.algo, task, client, args)
                algo.fit(X, y, w)
                predictions = algo.predict(X).map_blocks(cupy.asarray)
                # metric = dm.mean_squared_error(y.values, predictions)
                # timer = Timer.global_timer()
                # timer[args.algo]['mse'] = metric

    if not os.path.exists(args.output_directory):
        os.mkdir(args.output_directory)

    # Don't override the previous result.
    i = 0
    while True:
        f = args.algo + '-cpus_' + str(args.cpus) + '-rounds_' + \
            str(args.rounds) + '-data_' + args.data + '-' + str(i) + '.json'
        path = os.path.join(args.output_directory, f)
        if os.path.exists(path):
            i += 1
            continue
        with open(path, 'w') as fd:
            timer = Timer.global_timer()
            timer['packages'] = packages_version()
            json.dump(timer, fd, indent=2)
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arguments for mortgage benchmark.')
    parser.add_argument('--local_directory', type=str,
                        help='Local directory for storing the dataset.',
                        default='dxgb_bench_workspace')
    parser.add_argument('--temporary_directory', type=str,
                        help='Temporary directory used for dask.',
                        default='dask_workspace')
    parser.add_argument('--output-directory', type=str,
                        help='Directory storing benchmark results.',
                        default='benchmark_outputs')
    parser.add_argument('--cpus', type=int,
                        default=psutil.cpu_count(logical=False))
    parser.add_argument('--algo',
                        type=str,
                        help='Use algorithm',
                        default='xgboost-dask-gpu')
    parser.add_argument('--rounds', type=int, default=100,
                        help='Number of boosting rounds.')
    parser.add_argument('--data', type=str, help='Name of dataset.',
                        required=True)
    parser.add_argument('--years', type=int, help='Years of mortgage dataset',
                        required=False,
                        default=1)
    parser.add_argument('--backend', type=str, help='Data loading backend.',
                        required=True)
    args = parser.parse_args()
    main(args)
