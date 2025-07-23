import os
import subprocess
import pandas as pd
from datetime import datetime

def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

def prepare_data(df, categorical):
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df

def save_read_data():
    # Test input
    data = [
        (None, None, dt(1, 1), dt(1, 10)),
        (1, 1, dt(1, 2), dt(1, 10)),
        (1, None, dt(1, 2, 0), dt(1, 2, 59)),
        (3, 4, dt(1, 2, 0), dt(2, 2, 1)),
    ]
    columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
    df = pd.DataFrame(data, columns=columns)

    categorical = ['PULocationID', 'DOLocationID']
    actual_df = prepare_data(df, categorical)[categorical + ['duration']]

    options = {
        'client_kwargs': {
            'endpoint_url': os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:4566")
        }
    }

    input_file = "s3://nyc-duration/tests/integration.parquet"

    df.to_parquet(input_file, engine='pyarrow', compression=None, index=False, storage_options=options)
    print(df)
    print(f'writing dataframe to {input_file}')

    print(f'executing batch.py with test input data')
    env_vars = os.environ.copy()
    env_vars["INPUT_FILE_PATTERN"] = "s3://nyc-duration/tests/integration.parquet"
    env_vars["OUTPUT_FILE_PATTERN"] = "s3://nyc-duration/tests/integration_results.parquet"
    env_vars["S3_ENDPOINT_URL"] = "http://127.0.0.1:4566"

    subprocess.run([ "python", "batch.py", "2023", "1" ], env=env_vars, check=True)    
    
if __name__ == "__main__":
    save_read_data()