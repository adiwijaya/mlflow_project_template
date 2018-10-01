# Import various libraries including matplotlib, sklearn, mlflow
import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score,auc, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import datasets

from util import model_train



if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
    # Load Diabetes datasets
    diabetes = datasets.load_diabetes()
    X = diabetes.data
    y = diabetes.target

    # Create pandas DataFrame for sklearn ElasticNet linear_model
    Y = np.array([y]).transpose()
    d = np.concatenate((X, Y), axis=1)
    cols = ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6', 'progression']
    data = pd.DataFrame(d, columns=cols)


    #############################################################
    model_train(data)