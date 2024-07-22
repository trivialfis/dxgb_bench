Some simple scripts for running xgboost-dask pipeline.  Heavily inspired by [[https://github.com/NVIDIA/gbm-bench][gbm-bench]] .
By default these scripts use _dask_cudf_ as bank-end for ETL.

# Available datasets:
  - mortgage
```
      --data=mortgage:2 for using 2 years of the data. 4, 8, 16, 17 are also available.
```
  - taxi yellow tripdata 2014-01  (Copied from rapids' demo)
  - higgs
  - YearPrediction

# Available algorithms:
  - GPU Histogram
  - CPU Histogram
  - CPU Approx

# Usage
  Install dxgb_bench:
```
    pip install .
```

```
    dxgb-bench --algo='xgboost-gpu-hist' --data='mortgage' --backend='dask_cudf'
```

  Format multiple benchmark results recorded in JSON into a single CSV file:
```
    dxgb-to-csv --json-directory='benchmark_outputs' --output-directory='benchmark_outputs'
```

  The result is saved into a JSON file along with packages' version.  Each run will try to
  avoid overwriting previous saved file, so one can run it multiple times without renaming
  output file at each run.

# Dependencies
  See XGBoost.yml for details.  XGBoost itself is assumed to be installed from source.