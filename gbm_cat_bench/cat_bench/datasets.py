from __future__ import annotations

import os
import pickle
from typing import List, Optional, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from joblib import Memory
from pandas.api.types import is_categorical_dtype
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from .utils import DownloadProgress, Method, Task

memory = Memory("cachedir")

FetchReturn = Tuple[pd.DataFrame, np.ndarray, Task, Optional[List[bool]]]


# @memory.cache
def ames_housing(method: Method) -> FetchReturn:
    """
    Number of samples: 1460
    Number of features: 20
    Number of categorical features: 10
    Number of numerical features: 10
    """
    X, y = fetch_openml(data_id=42165, as_frame=True, return_X_y=True)
    print(y)

    categorical_columns_subset: list[str] = [
        "BldgType",  # 5, no nan
        "GarageFinish",  # 3, nan
        "LotConfig",  # 5, no nan
        "Functional",  # 7, no nan
        "MasVnrType",  # 4, nan
        "HouseStyle",  # 8, no nan
        "FireplaceQu",  # 5, nan
        "ExterCond",  # 5, no nan
        "ExterQual",  # 4, no nan
        "PoolQC",  # 3, nan
    ]

    numerical_columns_subset: list[str] = [
        "3SsnPorch",
        "Fireplaces",
        "BsmtHalfBath",
        "HalfBath",
        "GarageCars",
        "TotRmsAbvGrd",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "GrLivArea",
        "ScreenPorch",
    ]

    X = X[categorical_columns_subset + numerical_columns_subset]
    X[categorical_columns_subset] = X[categorical_columns_subset].astype("category")

    n_categorical_features = X.select_dtypes(include="category").shape[1]
    n_numerical_features = X.select_dtypes(include="number").shape[1]
    categorical_mask = [True] * n_categorical_features + [False] * n_numerical_features

    if method == Method.ORD:
        ordinal_encoder = make_column_transformer(
            (
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = ordinal_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.REG, categorical_mask
        return X, y, Task.REG, categorical_mask
    if method == Method.OHE:
        one_hot_encoder = make_column_transformer(
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = one_hot_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.REG, None
        print("return OHE")
        return X, y, Task.REG, None
    if method == Method.NOHE:
        return X, y, Task.REG, categorical_mask
    if method == Method.PART:
        return X, y, Task.REG, categorical_mask

    raise ValueError("Unknown method.")


def adult(method: Method) -> FetchReturn:
    X, y = fetch_openml(data_id=179, as_frame=True, return_X_y=True)
    y = y.cat.codes

    cat_mask = list(X.dtypes == "category")

    if method == Method.ORD:
        ordinal_encoder = make_column_transformer(
            (
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = ordinal_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.BIN, cat_mask
        return X, y, Task.BIN, cat_mask
    elif method == Method.OHE:
        one_hot_encoder = make_column_transformer(
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = one_hot_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.BIN, None
        return X, y, Task.BIN, None
    if method == Method.NOHE or method == Method.PART:
        return X, y, Task.BIN, cat_mask

    raise ValueError("Unknown method.")


def amazon_employee(method: Method) -> FetchReturn:
    "https://www.kaggle.com/c/amazon-employee-access-challenge/data"
    X, y = fetch_openml(data_id=4135, as_frame=True, return_X_y=True)
    print(X.shape)
    cat_mask = list(X.dtypes == "category")
    print(cat_mask)

    if method == Method.ORD:
        pass
    if method == Method.NOHE or method == Method.PART:
        return X, y, Task.BIN, cat_mask

    raise ValueError("Unknown method.")


def kick(method: Method) -> FetchReturn:
    """https://www.kaggle.com/c/DontGetKicked/data

    n_samples: 72983
    n_features: 32
    n_categorical_features: 18

    """
    X, y = fetch_openml(data_id=41162, as_frame=True, return_X_y=True)
    y = y.cat.codes
    cat_mask = list(X.dtypes == "category")
    print("n_samples:", X.shape[0])
    print("n_features:", X.shape[1])
    print("n_categorical_features:", len(list(filter(None, cat_mask))))

    if method == Method.ORD:
        pass
    if method == Method.NOHE or method == Method.PART:
        return X, y, Task.BIN, cat_mask

    raise ValueError("Unknown method.")


def retrieve(uri: str, local_directory: str) -> str:
    if not os.path.exists(local_directory):
        os.makedirs(local_directory)
    filename = os.path.join(local_directory, os.path.basename(uri))
    if not os.path.exists(filename):
        print(
            "Retrieving from {uri} to {filename}".format(uri=uri, filename=filename),
            flush=True,
        )
        urlretrieve(uri, filename, DownloadProgress())
    return filename


@memory.cache
def airline_categorical(method: Method) -> FetchReturn:
    url = "http://kt.ijs.si/elena_ikonomovska/datasets/airline/airline_14col.data.bz2"
    dataset_folder = "/home/fis/Others/datasets/resources/airline"
    local_url = os.path.join(dataset_folder, os.path.basename(url))

    if not os.path.isfile(local_url):
        retrieve(url, local_url)

    cols = [
        "Year",
        "Month",
        "DayofMonth",
        "DayofWeek",
        "CRSDepTime",
        "CRSArrTime",
        "UniqueCarrier",
        "FlightNum",
        "ActualElapsedTime",
        "Origin",
        "Dest",
        "Distance",
        "Diverted",
        "ArrDelay",
    ]

    # load the data as int16
    dtype = np.int16

    dtype_columns = {
        "Year": dtype,
        "Month": dtype,
        "DayofMonth": dtype,
        "DayofWeek": dtype,
        "CRSDepTime": dtype,
        "CRSArrTime": dtype,
        "FlightNum": dtype,
        "ActualElapsedTime": dtype,
        "Distance": dtype,
        "Diverted": dtype,
        "ArrDelay": dtype,
    }

    print("Reading dataset.", flush=True)
    df = pd.read_csv(local_url, names=cols, dtype=dtype_columns)

    def as_categorical() -> FetchReturn:
        for col in df.select_dtypes(["object"]).columns:
            df[col] = df[col].astype("category")

        print("df.dtypes:", df.dtypes)

        df["ArrDelay"] = 1 * (df["ArrDelay"] > 0)
        X = df[df.columns.difference(["ArrDelay"])]
        y = df["ArrDelay"].to_numpy(dtype=np.float32)

        cat_mask = []
        for dtype in X.dtypes:
            cat_mask.append(is_categorical_dtype(dtype))
        return X, y, Task.BIN, cat_mask

    print("ETL.", flush=True)
    if method == Method.NOHE or method == Method.PART:
        return as_categorical()
    elif method == Method.OHE:
        X, y, task, _ = as_categorical()
        one_hot_encoder = make_column_transformer(
            (
                OneHotEncoder(sparse=False, handle_unknown="ignore"),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = one_hot_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.BIN, None
        return X, y, Task.BIN, None
    else:
        ordinal_encoder = make_column_transformer(
            (
                OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=np.nan
                ),
                make_column_selector(dtype_include="category"),
            ),
            remainder="passthrough",
        )
        X = ordinal_encoder.fit_transform(X)
        if isinstance(X, np.ndarray):
            return pd.DataFrame(X), y, Task.BIN, None
        return X, y, Task.BIN, None

    raise ValueError("unreachable")


def fetch_data(name: str, method: Method) -> FetchReturn:
    if name == "ames_housing":
        return ames_housing(method)
    if name == "adult":
        return adult(method)
    if name == "employee":
        return amazon_employee(method)
    if name == "airline":
        return airline_categorical(method)
    if name == "taxi":
        # https://www.crowdanalytix.com/contests/mckinsey-big-data-hackathon
        pass
    if name == "Amazon":
        pass
    if name == "appetency":
        pass
    if name == "Click":
        pass
    if name == "internet":
        pass
    if name == "kick":
        return kick(method)
    if name == "mortgage":
        # https://www.crowdanalytix.com/contests/propensity-to-fund-mortgages
        pass
    if name == "Upselling":
        # https://kdd.org/kdd-cup/view/kdd-cup-2009/Data
        # https://medium.com/@kushaldps1996/customer-relationship-prediction-kdd-cup-2009-6b57d08ffb0
        pass
    if name == "Poverty_A":
        pass
    if name == "Poverty_B":
        pass
    if name == "Poverty_C":
        pass
    raise ValueError("Invalid dataset.")
