'''Comes from rapids AI's demo script.  Many feature engineering stuffs here
are not really useful.

'''
from dxgb_bench.utils import DataSet, fprint
import dask_cudf
import math
import numpy as np
from dask.distributed import wait
import os
from math import cos, sin, asin, sqrt, pi


def cleanup(df):
    # # Data Cleanup
    #
    # As usual, the data needs to be massaged a bit before we can start adding
    # features that are useful to an ML model.
    #
    # For example, in the 2014 taxi CSV files, there are `pickup_datetime` and
    # `dropoff_datetime` columns. The 2015 CSVs have `tpep_pickup_datetime` and
    # `tpep_dropoff_datetime`, which are the same columns. One year has
    # `rate_code`, and another `RateCodeID`.
    #
    # Also, some CSV files have column names with extraneous spaces in them.
    #
    # Worst of all, starting in the July 2016 CSVs, pickup & dropoff latitude
    # and longitude data were replaced by location IDs, making the second half
    # of the year useless to us.
    #
    # We'll do a little string manipulation, column renaming, and concatenating
    # of DataFrames to sidestep the problems.

    # list of column names that need to be re-mapped
    remap_features = {}
    remap_features['tpep_pickup_datetime'] = 'pickup_datetime'
    remap_features['tpep_dropoff_datetime'] = 'dropoff_datetime'
    remap_features['ratecodeid'] = 'rate_code'

    # create a list of columns & dtypes the df must have
    must_haves_features = {
        'pickup_datetime': 'datetime64[ms]',
        'dropoff_datetime': 'datetime64[ms]',
        'passenger_count': 'int32',
        'trip_distance': 'float32',
        'pickup_longitude': 'float32',
        'pickup_latitude': 'float32',
        'rate_code': 'int32',
        'dropoff_longitude': 'float32',
        'dropoff_latitude': 'float32',
        'fare_amount': 'float32'
    }

    # helper function which takes a DataFrame partition
    def clean(df_part, remap, must_haves):
        # some col-names include pre-pended spaces remove & lowercase column
        # names
        tmp = {col: col.strip().lower() for col in list(df_part.columns)}
        df_part = df_part.rename(tmp)

        # rename using the supplied mapping
        df_part = df_part.rename(remap)

        # iterate through columns in this df partition
        for col in df_part.columns:
            # drop anything not in our expected list
            if col not in must_haves:
                df_part = df_part.drop(col)
                continue

            # fixes datetime error found by Ty Mckercher and fixed by Paul
            # Mahler
            if df_part[col].dtype == 'object' and col in [
                    'pickup_datetime', 'dropoff_datetime'
            ]:
                df_part[col] = df_part[col].astype('datetime64[ms]')
                continue

            # if column was read as a string, recast as float
            if df_part[col].dtype == 'object':
                df_part[col] = df_part[col].str.fillna('-1')
                df_part[col] = df_part[col].astype('float32')
            else:
                # downcast from 64bit to 32bit types
                # Tesla T4 are faster on 32bit ops
                if 'int' in str(df_part[col].dtype):
                    df_part[col] = df_part[col].astype('int32')
                if 'float' in str(df_part[col].dtype):
                    df_part[col] = df_part[col].astype('float32')
                df_part[col] = df_part[col].fillna(-1)

        return df_part

    df = df.map_partitions(clean, remap_features, must_haves_features)
    return df


# # Adding Interesting Features
#
# Dask & cuDF provide standard DataFrame operations, but also let you run "user
# defined functions" on the underlying data.
#
# cuDF's
# [apply_rows](https://rapidsai.github.io/projects/cudf/en/0.6.0/api.html#cudf.dataframe.DataFrame.apply_rows)
# operation is similar to Pandas's
# [DataFrame.apply](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.apply.html),
# except that for cuDF, custom Python code is [JIT compiled by
# numba](https://numba.pydata.org/numba-doc/dev/cuda/kernels.html) into GPU
# kernels.
#
# We'll use a Haversine Distance calculation to find total trip distance, and
# extract additional useful variables from the datetime fields.

# In[10]:


def haversine_distance_kernel(pickup_latitude, pickup_longitude,
                              dropoff_latitude, dropoff_longitude, h_distance):
    for i, (x_1, y_1, x_2, y_2) in enumerate(
            zip(pickup_latitude, pickup_longitude, dropoff_latitude,
                dropoff_longitude)):
        x_1 = pi / 180 * x_1
        y_1 = pi / 180 * y_1
        x_2 = pi / 180 * x_2
        y_2 = pi / 180 * y_2

        dlon = y_2 - y_1
        dlat = x_2 - x_1
        a = sin(dlat / 2)**2 + cos(x_1) * cos(x_2) * sin(dlon / 2)**2

        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers

        h_distance[i] = c * r


def day_of_the_week_kernel(day, month, year, day_of_week):
    for i, (d_1, m_1, y_1) in enumerate(zip(day, month, year)):
        if month[i] < 3:
            shift = month[i]
        else:
            shift = 0
        Y = year[i] - (month[i] < 3)
        y = Y - 2000
        c = 20
        d = day[i]
        m = month[i] + shift + 1
        day_of_week[i] = (d + math.floor(m * 2.6) + y + (y // 4) +
                          (c // 4) - 2 * c) % 7


def add_features(df):
    df['hour'] = df['pickup_datetime'].dt.hour
    df['year'] = df['pickup_datetime'].dt.year
    df['month'] = df['pickup_datetime'].dt.month
    df['day'] = df['pickup_datetime'].dt.day
    df['diff'] = df['dropoff_datetime'].astype(
        'int32') - df['pickup_datetime'].astype('int32')

    df['pickup_latitude_r'] = df['pickup_latitude'] // .01 * .01
    df['pickup_longitude_r'] = df['pickup_longitude'] // .01 * .01
    df['dropoff_latitude_r'] = df['dropoff_latitude'] // .01 * .01
    df['dropoff_longitude_r'] = df['dropoff_longitude'] // .01 * .01

    df = df.drop('pickup_datetime')
    df = df.drop('dropoff_datetime')

    df = df.apply_rows(haversine_distance_kernel,
                       incols=[
                           'pickup_latitude', 'pickup_longitude',
                           'dropoff_latitude', 'dropoff_longitude'
                       ],
                       outcols=dict(h_distance=np.float32),
                       kwargs=dict())

    df = df.apply_rows(day_of_the_week_kernel,
                       incols=['day', 'month', 'year'],
                       outcols=dict(day_of_week=np.float32),
                       kwargs=dict())

    df['is_weekend'] = (df['day_of_week'] < 2).astype(np.float32)
    return df


def load(base_path):
    data_path = os.path.join(base_path, 'yellow_tripdata_2014-01.csv')
    df_2014 = dask_cudf.read_csv(data_path)
    df_2014 = cleanup(df_2014)

    # apply a list of filter conditions to throw out records with missing or
    # outlier values
    query_frags = [
        'fare_amount > 0 and fare_amount < 500',
        'passenger_count > 0 and passenger_count < 6',
        'pickup_longitude > -75 and pickup_longitude < -73',
        'dropoff_longitude > -75 and dropoff_longitude < -73',
        'pickup_latitude > 40 and pickup_latitude < 42',
        'dropoff_latitude > 40 and dropoff_latitude < 42'
    ]
    taxi_df = df_2014
    taxi_df = taxi_df.query(' and '.join(query_frags))

    fprint(taxi_df.head().to_pandas())
    # actually add the features
    taxi_df = taxi_df.map_partitions(add_features)
    # inspect the result
    taxi_df.head().to_pandas()

    # # Pick a Training Set
    #
    # Let's imagine you're making a trip to New York on the 25th and want to
    # build a model to predict what fare prices will be like the last few days
    # of the month based on the first part of the month. We'll use a query
    # expression to identify the `day` of the month to use to divide the data
    # into train and test sets.
    #
    # The wall-time below represents how long it takes your GPU cluster to load
    # data from the Google Cloud Storage bucket and the ETL portion of the
    # workflow.

    # create a Y_train ddf with just the target variable
    X_train = taxi_df.query('day < 25').persist()
    Y_train = X_train[[
        'fare_amount'
    ]].persist()  # drop the target variable from the training ddf
    X_train = X_train[X_train.columns.difference(['fare_amount'])]

    # this wont return until all data is in GPU memory
    done = wait([X_train, Y_train])
    fprint('Done:', done)

    # # Train the XGBoost Regression Model
    #
    # The wall time output below indicates how long it took your GPU cluster to
    # train an XGBoost model over the training set.
    return X_train, Y_train


class Taxi(DataSet):
    def __init__(self, args):
        self.uri = 'https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2014-01.csv'
        self.taxi_directory = os.path.join(
            args.local_directory, 'taxi', '2014')
        self.retrieve(self.taxi_directory)
        self.task = 'reg:squarederror'

    def load(self, args):
        train_X, train_y = load(self.taxi_directory)
        return train_X, train_y, None
