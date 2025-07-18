#!/usr/bin/env python
# coding: utf-8

import logging

from zenml import step, pipeline
from typing import Annotated

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error


from pathlib import Path

import pandas as pd

import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

models_folder = Path('models')
models_folder.mkdir(exist_ok=True)

@step
def read_dataframe(year: int, month: int) -> Annotated[pd.DataFrame, "df"]:
    url = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    df = pd.read_parquet(url)
    logging.info(f"Loaded {len(df)} records from {url}")

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    logging.info(f"Size of dataframe in bytes {df.memory_usage(deep=True).sum()} ")

    return df

@step
def train_and_log_model(df: pd.DataFrame) -> None:

    categorical = ['PULocationID', 'DOLocationID']
    train_dicts = df[categorical].to_dict(orient='records')

    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)

    # train model
    target = 'duration'
    y_train = df[target].values

    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_train)

    with mlflow.start_run():
        # Log model
        mlflow.sklearn.log_model(
            sk_model=lr,
            artifact_path="models",
            registered_model_name="nyc-taxi-linear-regression"
        )
        # Log metrics and parameters
        mlflow.log_metric("rmse", root_mean_squared_error(y_train, y_pred))
        mlflow.log_param("intercept", lr.intercept_)
        logging.info(f"rmse {root_mean_squared_error(y_train, y_pred)} ")
        logging.info(f"Model's intercept_ {lr.intercept_} ")
    

@pipeline
def run():
    df_train = read_dataframe(year=2023, month=3)
    train_and_log_model(df_train)

if __name__ == "__main__":
    run()