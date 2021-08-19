from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="dxgb-bench",
        version="0.0.0",
        description="Benchmark scripts for XGBoost.",
        scripts=["dxgb-bench"],
        install_requires=[
            "numpy",
            "pandas",
            "cudf",
            "dask",
            "distributed",
            "dask-cuda",
            "dask-cudf",
            "pynvml",
            "psutil",
        ],
        packages=find_packages(),
    )
