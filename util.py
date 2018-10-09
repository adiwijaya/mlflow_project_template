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


def load_csv_to_pandas(path, delimiter=';'):
    data_pandas = pd.read_csv(path, sep=delimiter, error_bad_lines=False, index_col=0)
    return data_pandas


def create_model(h2o, mlflow, data, target, run_time = 100):
    from h2o.automl import H2OAutoML

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data)

    htrain = h2o.H2OFrame(train)
    htest = h2o.H2OFrame(test)

    # set the predictor names and the response column name
    train_x = htrain.columns
    train_x.remove(target)
    test_x = htest.columns
    test_x.remove(target)
    test_y = test[[target]]

    # Classification problem change the target to factor
    htrain[target] = htrain[target].asfactor()

    # Start an MLflow run; the "with" keyword ensures we'll close the run even if this cell crashes
    with mlflow.start_run():

        # Initiating AutoML
        aml = H2OAutoML(max_runtime_secs=run_time, balance_classes=True)

        # Model Train
        aml.train(x=train_x, y=target, training_frame=htrain, validation_frame=htest)
        best_model = aml.leader

        # Model Evaluation
        evaluate_model_accuracy(model = best_model, test_data=htest, test_target =  test_y, h2o = h2o)

        # Log mlflow attributes for mlflow UI
        #mlflow.log_param("max_runtime_secs", max_runtime_secs)
        #mlflow.log_metric("accuracy_score", accuracy_score_metrics)

        return best_model


def evaluate_model_accuracy(model, test_data, test_target, h2o):
    # Model Predict Test
    result_prediction = model.predict(test_data)
    predicted_qualities = h2o.as_list(result_prediction, use_pandas=True)

    # Calculate Accuracy
    accuracy_score_metrics = eval_metrics_binary_classification(test_target, predicted_qualities[['predict']])

    # Print out ElasticNet model metrics
    print("accuracy_score: %s" % accuracy_score_metrics)


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


