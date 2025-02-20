from __future__ import annotations

from abc import abstractmethod
from typing import Any

import lightgbm as lgb
import numpy as np
import xgboost as xgb
from joblib import Memory
from sklearn.base import is_classifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import OrdinalEncoder

from .datasets import fetch_data
from .utils import Method, Task, get_meth, timer

np.set_printoptions(precision=2, threshold=np.iinfo(np.int32).max)


class Algo:
    def __init__(self, method: Method, n_iter: int, depth: int):
        self.method = method
        self.n_iter = n_iter
        self.depth = depth

    @abstractmethod
    def fit(self, dataset: str) -> dict[str, Any]: ...


def avg_result(result: dict) -> dict[str, float]:
    fit_time = result["fit_time"]
    test_score = result["test_score"]

    avg = {}
    avg["fit_time"] = np.mean(fit_time)
    avg["test_score_mean"] = np.mean(np.abs(test_score))
    avg["test_score_std"] = np.std(test_score)
    return avg


class SklHist(Algo):
    def __init__(self, n_iter: int, depth: int, method: Method):
        super().__init__(method, n_iter, depth)

    def fit(self, dataset: str) -> dict[str, Any]:
        method = self.method
        X, y, task, cat_mask = fetch_data(dataset, method)
        if method == Method.ORD:
            est = HistGradientBoostingRegressor(
                random_state=0,
                early_stopping=False,
                max_iter=self.n_iter,
                max_depth=self.depth,
            )
        elif method == Method.OHE:
            est = HistGradientBoostingRegressor(
                random_state=0,
                early_stopping=False,
                max_iter=self.n_iter,
                max_depth=self.depth,
            )
        elif method == Method.NOHE:
            raise NotImplementedError(
                "Native One hot encoding is not supported by scikit-learn hist."
            )
        elif method == Method.PART:
            enc = OrdinalEncoder(
                handle_unknown="use_encoded_value", unknown_value=np.nan
            )
            X = enc.fit_transform(X)
            est = HistGradientBoostingRegressor(
                random_state=0,
                early_stopping=False,
                categorical_features=cat_mask,
                max_iter=self.n_iter,
                max_depth=self.depth,
            )

        scoring = "neg_mean_absolute_percentage_error"
        n_cv_folds = 3

        with timer(name=f"skl-hist-{method}"):
            result = cross_validate(est, X, y, cv=n_cv_folds, scoring=scoring)
            print(result)

        return avg_result(result)


USE_ONEHOT = np.iinfo(np.int32).max
USE_PART = 1


class Lgb(Algo):
    def __init__(
        self, n_iter: int, depth: int, max_cat_threshold: int, method: Method
    ) -> None:
        super().__init__(method, n_iter, depth)
        self.params = {
            "n_estimators": self.n_iter,
            "max_depth": self.depth,
            "boost_from_average": False,
            "min_data_in_leaf": 1,
            "min_sum_hessian_in_leaf": 1e-6,
            "n_jobs": 1,
            "learning_rate": 1.0,
            "cat_smooth": 0.0,
            "min_data_per_group": 1,
            "max_cat_threshold": max_cat_threshold,
            "min_data_in_bin": 1,
            "cat_l2": 0.0,
            "verbosity": -1,
        }

    def fit(self, dataset: str) -> dict[str, Any]:
        method = self.method
        X, y, task, cat_mask = fetch_data(dataset, method)

        if method == Method.ORD:
            est = lgb.LGBMRegressor(**self.params)
        elif method == Method.OHE:
            est = lgb.LGBMRegressor(**self.params)
        elif method == Method.NOHE:
            est = lgb.LGBMRegressor(max_cat_to_onehot=USE_ONEHOT)
            est.set_params(**self.params)
        elif method == Method.PART:
            est = lgb.LGBMRegressor(max_cat_to_onehot=USE_PART)
            est.set_params(**self.params)
        est.fit(X, y, eval_set=[(X, y)])
        results = est.evals_result_["training"]["l2"]
        results = [np.sqrt(r) for r in results]
        predt = est.predict(X)
        # est.booster_.dump_model()
        print(est.booster_.dump_model())
        print("predt\n", predt)
        print("results:", results)
        # print("unique:", np.unique(predt))
        # print("predt:", predt)
        # mse = mean_squared_error(y, est.predict(X))
        # print("MSE:", mse)
        # scoring = "neg_mean_absolute_percentage_error"
        # n_cv_folds = 3

        # with timer(name=f"lgb-{method}"):
        #     result = cross_validate(est, X, y, cv=n_cv_folds, scoring=scoring)
        #     print(result)

        # return avg_result(result)


class Xgb(Algo):
    def __init__(
        self,
        tree_method: str,
        n_iter: int,
        depth: int,
        max_cat_threshold: int,
        method: Method,
    ) -> None:
        super().__init__(method, n_iter, depth)
        self.tree_method = tree_method
        self.params = {
            "reg_lambda": 0.0,
            "reg_alpha": 0.0,
            "max_bin": 16,
            "max_depth": self.depth,
            "grow_policy": "depthwise",
            "n_estimators": self.n_iter,
            "min_child_weight": 1e-6,
            "tree_method": self.tree_method,
            "max_cat_threshold": max_cat_threshold,
            "learning_rate": 1.0,
        }

    def fit(self, dataset: str) -> dict[str, Any]:
        method = self.method

        X, y, task, cat_mask = fetch_data(dataset, method)
        if task == Task.REG:
            estimator = xgb.XGBRegressor
            self.params["base_score"] = 0.0
        else:
            estimator = xgb.XGBClassifier
            self.params["base_score"] = 0.5

        if method == Method.ORD:
            est = estimator(**self.params)
        elif method == Method.OHE:
            est = estimator(**self.params)
        elif method == Method.NOHE:
            est = estimator(enable_categorical=True, max_cat_to_onehot=USE_ONEHOT)
            est.set_params(**self.params)
        elif method == Method.PART:
            est = estimator(enable_categorical=True, max_cat_to_onehot=USE_PART)
            est.set_params(**self.params)

        if task == Task.BIN:
            scoring = "roc_auc"
        elif task == Task.MUL:
            scoring = "roc_auc_ovr"
        else:
            scoring = "neg_mean_absolute_percentage_error"

        n_cv_folds = 3
        tm = self.tree_method
        with timer(name=f"xgb-{tm}-{method}"):
            result = cross_validate(est, X, y, cv=n_cv_folds, scoring=scoring)
            print(result)

        return avg_result(result)


def make_estimator(algo: str, n_iter: int, depth: int, max_cat_threshold: int) -> Algo:
    est_name = algo.split("-")[0]
    if est_name == "skl":
        return SklHist(n_iter, depth, get_meth(algo))
    if est_name == "lgb":
        return Lgb(n_iter, depth, max_cat_threshold, get_meth(algo))
    if est_name == "xgb":
        tree_method = algo.split("-")[1]
        if tree_method == "dhist":
            tree_method = "gpu_hist"
        return Xgb(tree_method, n_iter, depth, max_cat_threshold, get_meth(algo))

    raise ValueError("Unknown implementation.")


def run_all_algo(
    dataset: str, n_iter: int, depth: int, max_cat_threshold: int
) -> dict[str, dict]:
    results: dict[str, dict] = {}
    for algo in AVAILABLE_ALGOS:
        est = make_estimator(algo, n_iter, depth, max_cat_threshold)
        result = est.fit(dataset)
        results[algo] = result

    return results


AVAILABLE_ALGOS = [
    # scikit-learn
    "skl-ord",
    "skl-ohe",
    "skl-part",
    # XGBoost hist
    "xgb-hist-ord",
    "xgb-hist-ohe",
    "xgb-hist-nohe",
    "xgb-hist-part",
    # XGBoost approx
    "xgb-approx-ord",
    "xgb-approx-ohe",
    "xgb-approx-nohe",
    "xgb-approx-part",
    # XGBoost exact
    "xgb-exact-ord",
    "xgb-exact-ohe",
    # XGBoost GPU hist
    "xgb-dhist-ord",
    "xgb-dhist-ohe",
    "xgb-dhist-nohe",
    "xgb-dhist-part",
    # LightGBM
    "lgb-ord",
    "lgb-ohe",
    "lgb-nohe",
    "lgb-part",
]
