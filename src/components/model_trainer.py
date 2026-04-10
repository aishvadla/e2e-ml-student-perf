import os
import sys
from dataclasses import dataclass

from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

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
                        # 'splitter':['best','random'],
                        # 'max_features':['sqrt','log2'],
                    },
                },
                "Random Forest": {
                    "model": RandomForestRegressor(),
                    "params": {
                        # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                        # 'max_features':['sqrt','log2',None],
                        "n_estimators": [8, 16, 32, 64, 128, 256]
                    },
                },
                "AdaBoost": {
                    "model": AdaBoostRegressor(),
                    "params": {
                        "learning_rate": [0.1, 0.01, 0.5, 0.001],
                        # 'loss':['linear','square','exponential'],
                        "n_estimators": [8, 16, 32, 64, 128, 256],
                    },
                },
                "Gradient Boosting": {
                    "model": GradientBoostingRegressor(),
                    "params": {
                        # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                        "learning_rate": [0.1, 0.01, 0.05, 0.001],
                        "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                        # 'criterion':['squared_error', 'friedman_mse'],
                        # 'max_features':['auto','sqrt','log2'],
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

            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
            )

            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_model_score = model_report[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with R2 score: {best_model_score}"
            )

            save_object(
                obj=best_model,
                file_path=self.model_trainer_config.trained_model_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    di_obj = DataIngestion()
    train_data_path, test_data_path = di_obj.initiate_data_ingestion()

    dt_obj = DataTransformation()
    train_array, test_array, preprocessor_path = dt_obj.initiate_data_transformation(
        train_data_path, test_data_path
    )

    model_obj = ModelTrainer()
    model_obj.initiate_model_trainer(train_array, test_array)
