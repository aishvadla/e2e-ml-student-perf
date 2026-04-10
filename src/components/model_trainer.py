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

from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "KNN": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "SVR": SVR(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "XGBoost": XGBRegressor(),
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
            logging.info(
                f"Best model found: {best_model_name} with R2 score: {model_report[best_model_name]}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    dt_obj = DataTransformation()
    train_array, test_array, preprocessor_path = dt_obj.initiate_data_transformation()

    model_obj = ModelTrainer()
    model_obj.initiate_model_trainer(train_array, test_array, preprocessor_path)
