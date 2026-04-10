"""Utility helpers for the ML project.

This module provides helper functions for saving Python objects
and other common utilities used across the training and prediction pipelines.
"""

import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.logger import logging
from src.exception import CustomException

import joblib


def save_object(obj, file_path):
    """Serialize an object to disk using joblib.

    Parameters
    ----------
    obj : any
        The Python object to serialize.
    file_path : str
        Path to the target file where the object will be saved.

    Raises
    ------
    CustomException
        If saving fails due to an I/O or serialization error.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            joblib.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return joblib.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models):
    try:
        logging.info("Evaluating models")
        model_report = {}
        for model_name, model_info in models.items():
            gs = GridSearchCV(model_info["model"], model_info["params"], cv=5)
            gs.fit(X_train, y_train)

            best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            model_report[model_name] = test_model_score

        logging.info("Completed model evaluation")
        return model_report, best_model

    except Exception as e:
        raise CustomException(e, sys)
