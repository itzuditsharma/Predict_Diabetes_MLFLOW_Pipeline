import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/uditsharma8959/machinelearningpipeline.mlflow"
os.environ['MLFLOW_TRACKING_USERNAME'] = "uditsharma8959"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "c8d12c4ec8e693086fff73d44acbb6cadd404b2c"

# Load parameters from params.yaml 
params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop(columns=['Outcome'])
    y = data['Outcome']

    mlflow.set_tracking_uri("https://dagshub.com/uditsharma8959/machinelearningpipeline.mlflow")

    # load the model from the disk 
    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to MLFLOW 
    mlflow.log_metric("accuracy", accuracy)
    print(f"Model Accuracy: {accuracy}")


if __name__ == '__main__':
    evaluate(params['data'], params['model'])
