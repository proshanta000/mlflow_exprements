# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import tempfile
import shutil

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

# Import dagshub and initialize it BEFORE mlflow calls
import dagshub
dagshub.init(repo_owner='proshanta000', repo_name='mlflow_exprements', mlflow=True)

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

mlflow.set_tracking_uri("https://dagshub.com/proshanta000/mlflow_exprements.mlflow")


logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
    lr.fit(train_x, train_y)

    predicted_qualities = lr.predict(test_x)

    (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

    mlflow.set_experiment("winequality-red")

    # This is your main MLflow run block
    with mlflow.start_run():
   
        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("   RMSE: %s" % rmse)
        print("   MAE: %s" % mae)
        print("   R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Training Info", "Basic LR model winequality-red")

         # Infer the model signature
        signature = infer_signature(train_x, lr.predict(train_x))

        # --- Start of new, reliable model logging method ---
        # Create a temporary directory to save the model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = temp_dir + "/wine_model"
            mlflow.sklearn.save_model(
                sk_model=lr,
                path=model_path,
                # Note: signature and input_example are optional but good practice
                signature=infer_signature(train_x, lr.predict(train_x)), 
                input_example=train_x
            )
            mlflow.log_artifacts(model_path, "wine_model")
        # --- End of new model logging method ---