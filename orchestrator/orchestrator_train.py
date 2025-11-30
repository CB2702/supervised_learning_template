# add orchestration details here
import sys
import os
sys.path.insert(1, os.getcwd())
import pandas as pd
from packages.rename.ml_ops.data_import.data_import import run_import_data
from packages.rename.ml_ops.data_processing.data_processing import run_data_processing
from packages.rename.ml_ops.training.train import run_model_training
import mlflow
import shutil

def main():
    project_name = "update"
    model_name = "update"
    experiment_name = f"{project_name}_experiment_log"
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment_tag("version", "1.0")
    mlflow.set_experiment(experiment_name)
    mlflow.start_run()
    path = 'rename'
    print(f'Importing data from {path}.')
    df = run_import_data(path = path, name = 'update')
    mlflow.log_parameter("model_name", model_name)
    mlflow.log_metric("import_data_nrows", len(df))
    mlflow.log_metric("import_data_ncols", len(df.columns))
    print('Successfully imported data! See head sample below:')
    print(df.head())
    print(f"Number of rows in df: {len(df)}")
    print(f"Number of cols in df: {len(df.columns)}")

    print('Processing data for feature engineering.')
    df = run_data_processing(df)
    print('Successfully run feature engineering for dataset! See head sample below:')
    print(df.head())
    print(f"Number of rows in processed df: {len(df)}")
    print(f"Number of columns in processed df: {len(df.columns)}")

    mlflow.sklearn.autolog()
    print('Running model training. This may take some time.')
    model = run_model_training(df, 0.3, 0.5)
    try:
        mlflow.sklearn.save_model(model, f"{project_name}_{model_name}_multiclass_classifier")
    except:
        print('Model file already exists. Deleting and replacing')
        try:
            shutil.rmtree("landmines_dense_nn_multiclass_classifier")
            mlflow.sklearn.save_model(model, f"{project_name}_{model_name}_multiclass_classifier")
        except Exception as exp:
            print('Could not remove file.')
            raise Exception("Could not remove file") from exp

    return None

if __name__ == '__main__':
    main()