from setuptools import setup, find_packages


if __name__ == "__main__":
    setup(
        name="dxgb-bench",
        version="0.0.0",
        description="Benchmark scripts for XGBoost.",
        scripts=["dxgb-bench", "dxgb-to-csv"],
        install_requires=[
            "numpy",
            "pandas",
            "dask",
            "distributed",
            "dask-cuda",
            "pynvml",
            "psutil",
        ],
        packages=find_packages(),
    )
