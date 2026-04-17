import os

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression, Ridge

from src.exception import CustomException
from src.utils import evaluate_models, load_object, save_object


def test_save_object_and_load_object_roundtrip(tmp_path):
    obj = {"key": "value", "numbers": [1, 2, 3]}
    file_path = tmp_path / "artifacts" / "test_object.pkl"

    save_object(obj, str(file_path))

    assert os.path.exists(file_path), "Serialized object file should exist"

    loaded_obj = load_object(str(file_path))
    assert loaded_obj == obj, "Loaded object should match the original"


def test_load_object_raises_custom_exception_for_missing_file(tmp_path):
    missing_path = tmp_path / "does_not_exist.pkl"

    with pytest.raises(CustomException):
        load_object(str(missing_path))


def test_evaluate_models_returns_report_and_best_model():
    X, y = make_regression(n_samples=100, n_features=4, noise=5.0, random_state=42)
    X_train, X_test = X[:80], X[80:]
    y_train, y_test = y[:80], y[80:]

    models = {
        "linear": {"model": LinearRegression(), "params": {}},
        "ridge": {"model": Ridge(), "params": {"alpha": [0.1, 1.0]}},
    }

    model_report, best_model = evaluate_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        models=models,
    )

    assert isinstance(model_report, dict)
    assert set(model_report) == {"linear", "ridge"}
    assert isinstance(best_model, (LinearRegression, Ridge))
    assert hasattr(best_model, "predict")
    assert all(-1.0 <= score <= 1.0 for score in model_report.values())


def test_evaluate_models_raises_custom_exception_for_invalid_data():
    X_train = np.ones((10, 3))
    y_train = np.ones(5)
    X_test = np.ones((5, 3))
    y_test = np.ones(5)

    models = {
        "linear": {"model": LinearRegression(), "params": {}},
    }

    with pytest.raises(CustomException):
        evaluate_models(X_train, y_train, X_test, y_test, models)
