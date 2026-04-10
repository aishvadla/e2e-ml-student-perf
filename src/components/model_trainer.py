"""Model training component for the E2E ML project.

This module defines the model training pipeline that evaluates multiple
regression models using GridSearchCV, selects the best performer based on R²
score, and saves it to disk for downstream inference.
"""

import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models
from src.utils import save_object


@dataclass
class ModelTrainerConfig:
    """Configuration dataclass for model training paths and settings.

    Attributes:
        trained_model_file_path (str): Path where the trained model will be serialized.
    """

    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    """Trains and evaluates multiple regression models using GridSearchCV.

    This class orchestrates hyperparameter tuning across a variety of regression
    models and saves the best-performing model to disk based on test R² score.
    """

    def __init__(self):
        """Initialize the model trainer with configuration."""
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """Train and evaluate models on preprocessed data arrays.

        Separates features and target from combined arrays, defines model
        hyperparameter grids, evaluates each model via GridSearchCV, and
        saves the best-performing model to disk.

        Args:
            train_array (ndarray): Combined training features and target (last column).
            test_array (ndarray): Combined testing features and target (last column).

        Raises:
            CustomException: If training fails or no model meets minimum R² threshold of 0.6.

        Notes:
            The best model is selected as the one with the highest test R² score.
            Minimum threshold of 0.6 is enforced to ensure model quality.
        """
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            # Define model configurations with hyperparameter grids for tuning
            models = {
                "Decision Tree": {
                    "model": DecisionTreeRegressor(),
                    "params": {
                        "criterion": [
                            "squared_error",
                            "friedman_mse",
                            "absolute_error",
                            "poisson",
                        ],
                    },
                },
                "Random Forest": {
                    "model": RandomForestRegressor(),
                    "params": {"n_estimators": [8, 16, 32, 64, 128, 256]},
                },
                "AdaBoost": {
                    "model": AdaBoostRegressor(),
                    "params": {
                        "learning_rate": [0.1, 0.01, 0.5, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                },
                "Gradient Boosting": {
                    "model": GradientBoostingRegressor(),
                    "params": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                },
                "KNN": {
                    "model": KNeighborsRegressor(),
                    "params": {},
                },
                "Linear Regression": {
                    "model": LinearRegression(),
                    "params": {},
                },
                "Lasso": {
                    "model": Lasso(),
                    "params": {},
                },
                "Ridge": {
                    "model": Ridge(),
                    "params": {},
                },
                "SVR": {
                    "model": SVR(),
                    "params": {},
                },
                "CatBoost": {
                    "model": CatBoostRegressor(verbose=False),
                    "params": {
                        "depth": [6, 8, 10],
                        "learning_rate": [0.01, 0.05, 0.1],
                        "iterations": [30, 50, 100],
                    },
                },
                "XGBoost": {
                    "model": XGBRegressor(),
                    "params": {
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                },
            }

            # Evaluate all models and get reports
            model_report, best_model = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            # Find the model with highest test R² score
            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]

            # Enforce minimum performance threshold
            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            # Serialize and save the best model
            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
