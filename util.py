import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, auc, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def eval_metrics_regression(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def eval_metrics_binary_classification(actual, pred):
    accuracy_score_metrics = accuracy_score(actual, pred)
    return accuracy_score_metrics


def load_csv_to_pandas(path, delimiter = ';' ):
    data_pandas = pd.read_csv(path, sep=delimiter, error_bad_lines=False, index_col=0)
    return data_pandas


def model_experiment(h2o, mlflow, data, target, run_time = 100):
    from h2o.automl import H2OAutoML

    response = target

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    htrain = h2o.H2OFrame(train)
    htest = h2o.H2OFrame(test)

    # set the predictor names and the response column name
    train_x = htrain.columns
    train_x.remove(response)
    test_x = htest.columns
    test_x.remove(response)
    test_y = test[[response]]

    # Classification problem
    htrain[response] = htrain[response].asfactor()

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():
        aml = H2OAutoML(max_runtime_secs=run_time, balance_classes=True)
        aml.train(x=train_x, y=response, training_frame=htrain, validation_frame=htest)

        result_prediction = aml.predict(htest)
        predicted_qualities = h2o.as_list(result_prediction, use_pandas=True)

        accuracy_score_metrics = eval_metrics_binary_classification(test_y, predicted_qualities[['predict']])
        # Print out ElasticNet model metrics
        print("accuracy_score: %s" % accuracy_score_metrics)

        best_model = aml.leader

        # Log mlflow attributes for mlflow UI
        #mlflow.log_param("max_runtime_secs", max_runtime_secs)
        #mlflow.log_metric("accuracy_score", accuracy_score_metrics)

        return best_model


def remove_folder_linux(dirpath):
    import shutil
    import os

    if os.path.exists(dirpath) and os.path.isdir(dirpath):
        shutil.rmtree(dirpath)

    print("Folder exists. Removing previous model")


def save_model(model, mlflow, save_directory):

    # Log artifacts (output files)
    remove_folder_linux(save_directory)
    #mlflow.log_model(model, "model_bank_class")
    mlflow.save_model(model, save_directory)


