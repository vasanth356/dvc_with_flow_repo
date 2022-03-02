import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import logging
import warnings

logging.basicConfig(level = logging.WARN)
logger = logging.getLogger(__name__)


from mlflow.tracking import MlflowClient

# Create an experiment with a name that is unique and case sensitive.
# client = MlflowClient()
# experiment_id = client.create_experiment("mlflow scikit experiment6")

import dvc.api

path = r'data\winequality-red.csv'
repo = r'C:\Users\LENOVO\PycharmProjects\dvc_with_flow_repo'
version = 'version5'

data_url =  dvc.api.get_url(
    path = path,
    repo = repo,
    rev = version
    )

mlflow.set_experiment ('dvc_with_flow_2')


def get_data():

    try:
        df = pd.read_csv(data_url, sep=",")
        return df
    except Exception as e:
        raise e

def evaluate(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":

    warnings.filterwarnings("ignore")
    np.random.seed(40)

    df = get_data()

    train, test = train_test_split(df)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    train_y = train[["quality"]]
    test_y = test[["quality"]]

    with mlflow.start_run():
        client = MlflowClient()
        alpha = 1
        l1_ratio=1
        
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        pred = lr.predict(test_x)

        rmse, mae, r2 = evaluate(test_y, pred)

        print(f"Elastic net params: alpha: {alpha}, l1_ratio: {l1_ratio}")
        print(f"Elastic net metric: rmse:{rmse}, mae: {mae}, r2:{r2}")

        mlflow.log_param("data_url", data_url)
        mlflow.log_param("data_version", version)
        mlflow.log_param("number_rows", df.shape[0])
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)
        
        # mlflow model logging 
        mlflow.sklearn.log_model(lr, "model")
