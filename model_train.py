# IMPORT

import sys
from util import model_experiment, save_model,load_csv_to_pandas
import h2o
import mlflow
import mlflow.h2o

# STATIC VARIABLES
model_type = "classification" # classification / regression
target_column = "TARGET"
input_csv_path = "/home/mapr/dataset/bank-class-sample.csv"
delimiter_symbol = ";"
max_runtime_secs = 10
model_export_name = "/home/mapr/etc/mlflow/model_result/bank_class_sample"


def init_mlflow():
    # Set this variable to your MLflow server's DNS name
    mlflow_server = '178.128.58.69'

    # Tracking URI
    mlflow_tracking_uri = 'http://' + mlflow_server + ':5000'
    print("MLflow Tracking URI: %s" % (mlflow_tracking_uri))

    mlflow.set_tracking_uri(mlflow_tracking_uri)


if __name__ == "__main__":

    init_mlflow()
    h2o.init()

    # VARIABLE INITIATION
    #target_column = str(sys.argv[1]) if len(sys.argv) > 1 else target_column
    #input_csv_path = str(sys.argv[2]) if len(sys.argv) > 1 else input_csv_path
    #delimiter_symbol = str(sys.argv[3]) if len(sys.argv) > 1 else delimiter_symbol
    #max_runtime_secs = int(sys.argv[4]) if len(sys.argv) > 1 else max_runtime_secs
    #model_export_name = str(sys.argv[5]) if len(sys.argv) > 1 else model_export_name

    # LOAD DATA
    data = load_csv_to_pandas(input_csv_path, delimiter_symbol)

    # MODEL EXPERIMENT
    model = model_experiment(h2o, mlflow, data,target_column, max_runtime_secs)

    # MODEL EXPORT
    save_model(model, save_directory=model_export_name, mlflow=mlflow.h2o)
