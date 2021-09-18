from dask import dataframe as dd
from collections import OrderedDict
import os
from distributed import wait
from dxgb_bench.utils import DataSet, fprint, read_csv
import tarfile

try:
    import cudf
    import dask_cudf
except ImportError:
    cudf = None
    dask_cudf = None


def datetime_name(backend):
    if backend == "dask_cudf" or backend == "cudf":
        return "date"
    else:
        return "date"


def convert_dtypes(df, dtypes, delta, backend):
    for i, column in enumerate(dtypes.items()):
        name = column[0]
        if column[1] == "str" and (backend == "cudf" or backend == "dask_cudf"):
            df[name] = (
                df[name].astype("category").cat.codes.astype("float32")
            )
        elif (
            df.dtypes[i].name == "datetime64[ms]"
            or df.dtypes[i].name == "datetime64[ns]"
        ):
            df[name] = df[name].dt.day
    return df


def concat(dfs, backend):
    if backend == "dask_cudf":
        return dask_cudf.concat(dfs)
    elif backend == "cudf":
        return cudf.concat(dfs)
    elif backend == "dask":
        return dd.multi.concat(dfs)
    else:
        raise ValueError("Unknown concat backend:", backend)


def load_dateframe(path, dtypes, cols, backend):
    dfs = []

    transformed = {}
    for key, t in dtypes.items():
        if t == "category":
            transformed[key] = "str"
        else:
            transformed[key] = t

    for root, subdir, files in os.walk(path):
        for f in files:
            fprint("Reading: ", f, "with", backend)
            df = read_csv(
                os.path.join(root, f),
                # blocksize="32MB",
                sep="|",
                dtype=transformed,
                header=None,
                names=cols,
                skiprows=1,
                backend=backend,
            )
            dfs.append(df)

    concated = concat(dfs, backend)
    concated = convert_dtypes(concated, dtypes, "1D", backend)
    return concated


def load_performance_data(path, backend):
    if backend == "dask_cudf" or backend == "cudf":
        category = "str"
    else:
        category = "category"
    dtypes = OrderedDict(
        [("loan_id", "uint64"),
         ("monthly_reporting_period", datetime_name(backend)),
         ("servicer", category),
         ("interest_rate", "float32"),
         ("current_actual_upb", "float32"),
         ("loan_age", "float32"),
         ("remaining_months_to_legal_maturity", "float32"),
         ("adj_remaining_months_to_maturity", "float32"),
         ("maturity_date", datetime_name(backend)),
         ("msa", "float32"),
         ("current_loan_delinquency_status", "int32"),
         ("mod_flag", category),
         ("zero_balance_code", category),
         ("zero_balance_effective_date", datetime_name(backend)),
         ("last_paid_installment_date", datetime_name(backend)),
         ("foreclosed_after", datetime_name(backend)),
         ("disposition_date", datetime_name(backend)),
         ("foreclosure_costs", "float32"),
         ("prop_preservation_and_repair_costs", "float32"),
         ("asset_recovery_costs", "float32"),
         ("misc_holding_expenses", "float32"),
         ("holding_taxes", "float32"),
         ("net_sale_proceeds", "float32"),
         ("credit_enhancement_proceeds", "float32"),
         ("repurchase_make_whole_proceeds", "float32"),
         ("other_foreclosure_proceeds", "float32"),
         ("non_interest_bearing_upb", "float32"),
         ("principal_forgiveness_upb", "float32"),
         ("repurchase_make_whole_proceeds_flag", category),
         ("foreclosure_principal_write_off_amount", "float32"),
         ("servicing_activity_indicator", category)])

    keys = [k for k, v in dtypes.items()]
    perf = load_dateframe(path=path, dtypes=dtypes, cols=keys, backend=backend)
    return perf


def load_acq_data(acq_dir, backend):
    if backend == "dask_cudf" or backend == "cudf":
        category = "str"
    else:
        category = "category"
    dtypes = OrderedDict([
        ("loan_id", "uint64"),
        ("orig_channel", category),
        ("seller_name", category),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", datetime_name(backend)),
        ("first_pay_date", datetime_name(backend)),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", category),
        ("loan_purpose", category),
        ("property_type", category),
        ("num_units", "int64"),
        ("occupancy_status", category),
        ("property_state", category),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", category),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", category),
        ("something", "int64")
    ])

    keys = [k for k, v in dtypes.items()]

    fprint("Number of columns:", len(keys))
    acq = load_dateframe(path=acq_dir, dtypes=dtypes, cols=keys, backend=backend)
    return acq


def load_parquet(backend, *args, **kwargs):
    if backend == "dask_cudf":
        return dask_cudf.read_parquet(*args, **kwargs)
    elif backend == "cudf":
        return cudf.read_parquet(*args, **kwargs)
    elif backend == "dask":
        return dd.read_parquet(*args, **kwargs)


def preprocessing(perf, acq, backend):
    df = perf.merge(acq, how="left", on=["loan_id"])
    y = df["current_loan_delinquency_status"]
    if backend == "cudf":
        X = df[df.columns.difference(["current_loan_delinquency_status"])]
    else:
        columns = list(df.columns).remove("current_loan_delinquency_status")
        X = df.loc[columns]
    y = y - y.min()
    y = y / y.max()
    return X, y


class Mortgage(DataSet):
    def __init__(self, args):
        mortgage = "notebook-mortgage-data/"
        aws = "http://rapidsai-data.s3-website.us-east-2.amazonaws.com/"
        prefix = aws + mortgage
        if len(args.data.split(":")) == 2:
            years = int(args.data.split(":")[1])
        elif len(args.data.split(":")) == 1:
            years = 1
        else:
            raise ValueError("Invalid format for mortgage dataset.")

        self.local_directory = os.path.join(
            args.local_directory, "mortgage-" + str(years)
        )

        if years == 1:
            self.uri = prefix + "mortgage_2000.tgz"
        elif years == 2:
            self.uri = prefix + "mortgage_2000-2001.tgz"
        elif years == 4:
            self.uri = prefix + "mortgage_2000-2003.tgz"
        elif years == 8:
            self.uri = prefix + "mortgage_2000-2007.tgz"
        elif years == 16:
            self.uri = prefix + "mortgage_2000-2015.tgz"
        elif years == 17:
            self.uri = prefix + "mortgage_2000-2016.tgz"

        self.retrieve(self.local_directory)

        filename = os.path.join(self.local_directory, os.path.basename(self.uri))
        assert os.path.exists(filename)

        self.acq_dir = os.path.join(self.local_directory, "acq")
        self.perf_dir = os.path.join(self.local_directory, "perf")
        self.names_path = os.path.join(self.local_directory, "names.csv")

        extracted = all(
            [os.path.exists(f) for f in [self.acq_dir, self.perf_dir, self.names_path]]
        )

        if not extracted:
            with tarfile.open(filename, "r:gz") as tarball:
                fprint("Extracting", filename)
                tarball.extractall(self.local_directory)

        self.task = "binary:logistic"

    def load(self, args):
        dirpath = os.path.join(self.local_directory, "mortgage-parquet-" + args.backend)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)

        parquet_X = os.path.join(dirpath, "mortgage-X")

        if not os.path.exists(parquet_X):
            acq = load_acq_data(self.acq_dir, args.backend)
            perf = load_performance_data(self.perf_dir, args.backend)
            train_X, train_y = preprocessing(perf, acq, args.backend)

            print("Saving to parquet:", parquet_X)
            train_X["labels"] = train_y
            train_X.to_parquet(parquet_X)
            if args.backend != "cudf":
                wait(train_X)
        else:
            print("Reading from cached parquet.")

        df = load_parquet(args.backend, parquet_X)
        y = df["labels"]
        X = df[df.columns.difference(["labels"])]
        return X, y, None
