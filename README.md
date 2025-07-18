## ML FLOW experiments



ml flow remote : https://dagshub.com/proshanta000/mlflow_exprements.mlflow


import dagshub
dagshub.init(repo_owner='proshanta000', repo_name='mlflow_exprements', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)



export MLFLOW_TRACKING_USERNAME=proshanta000
export MLFLOW_TRACKING_PASSWORD=pro12112122@
export MLFLOW_TRACKING_USERNAME=d856c7bfcbe6c5c979320b3160b26a5a3e1f4355