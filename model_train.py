# IMPORT

import sys
from util import model_experiment, save_model,load_csv_to_pandas
import h2o
import mlflow
import mlflow.h2o

# STATIC VARIABLES
model_type = "classification" # classification / regression
target = "TARGET"
csv_path = "/home/mapr/dataset/bank-class-sample.csv"
delimiter = ";"
run_time = 10
export_path = "/opt/mlflow/model_result/bank_class_sample"


def init_mlflow():
    # Set this variable to your MLflow server's DNS name
    mlflow_server = '178.128.106.223'

    # Tracking URI
    mlflow_tracking_uri = 'http://' + mlflow_server + ':5000'
    print("MLflow Tracking URI: %s" % (mlflow_tracking_uri))

    mlflow.set_tracking_uri(mlflow_tracking_uri)


if __name__ == "__main__":

    #init_mlflow()
    h2o.init()

    # VARIABLE INITIATION
    target = str(sys.argv[1]) if len(sys.argv) > 1 else target
    csv_path = str(sys.argv[2]) if len(sys.argv) > 1 else csv_path
    delimiter = str(sys.argv[3]) if len(sys.argv) > 1 else delimiter
    run_time = int(sys.argv[4]) if len(sys.argv) > 1 else run_time
    export_path = str(sys.argv[5]) if len(sys.argv) > 1 else export_path

    # LOAD DATA
    data = load_csv_to_pandas(csv_path, delimiter)

    # MODEL EXPERIMENT
    model = model_experiment(h2o, mlflow, data,target, run_time)

    # MODEL EXPORT
    save_model(model, save_directory=export_path, mlflow=mlflow.h2o)
