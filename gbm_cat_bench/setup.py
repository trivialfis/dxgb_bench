from setuptools import find_packages, setup

if __name__ == "__main__":
    setup(
        name="cat_bench",
        version="0.0.0",
        description="Benchmark categorical features.",
        scripts=["run_cat_bench"],
        packages=find_packages(),
        license="Apache-2.0",
        maintainer="Jiaming Yuan",
        python_requires=">=3.8",
        install_requires=[
            "numpy",
            "scipy",
            # "xgboost>=1.6.1",
            "scikit-learn",
            "lightgbm",
        ],
    )
