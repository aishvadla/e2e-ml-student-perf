"""Data transformation component for the E2E ML project.

This module defines the transformation pipeline used to prepare raw training
and testing data for modeling. It handles feature selection, missing value
imputation, scaling, and encoding of categorical variables.
"""

import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    """Prepares raw tabular data for machine learning.

    This class orchestrates the ingestion of raw data and applies preprocessing
    steps to numeric, nominal, and ordinal features.
    """

    def __init__(self):
        """Initialize the data transformation component."""
        self.data_ingestion = DataIngestion()

    def initiate_data_transformation(self):
        """Run ingestion and transform the dataset for modeling.

        Returns:
            tuple: Transformed training features, training targets,
                transformed testing features, and testing targets.
        """
        try:
            logging.info("Entered Data Transformation component")

            train_data_path, test_data_path = (
                self.data_ingestion.initiate_data_ingestion()
            )

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            # Selecting the target variable for the task.
            y_train = train_data["math_score"]
            X_train = train_data.drop(columns=["math_score"])

            y_test = test_data["math_score"]
            X_test = test_data.drop(columns=["math_score"])

            # Identify numeric and categorical feature columns.
            numeric_features = X_train.select_dtypes(
                exclude=["string", "object"]
            ).columns.tolist()
            categorical_features = X_train.select_dtypes(
                include=["string", "object"]
            ).columns.tolist()

            # Define ordinal ordering for parental education level.
            ordinal_features = ["parental_level_of_education"]
            ordinal1_education_order = [
                "some high school",
                "high school",
                "some college",
                "associate's degree",
                "bachelor's degree",
                "master's degree",
            ]

            nominal_features = [
                col for col in categorical_features if col not in ordinal_features
            ]

            # Numeric pipeline: impute missing values and scale features.
            numeric_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler()),
                ]
            )

            # Nominal pipeline: impute and one-hot encode categorical features.
            nominal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore", drop="first", sparse_output=True
                        ),
                    ),
                ]
            )

            # Ordinal pipeline: impute and convert ordered categories to integers.
            ordinal_pipeline = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "encoder",
                        OrdinalEncoder(
                            categories=[ordinal1_education_order],
                            handle_unknown="use_encoded_value",
                            unknown_value=-1,
                        ),
                    ),
                ]
            )

            preprocessing_pipeline = ColumnTransformer(
                [
                    ("numeric", numeric_pipeline, numeric_features),
                    ("nominal", nominal_pipeline, nominal_features),
                    ("ordinal", ordinal_pipeline, ordinal_features),
                ]
            )

            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)

            logging.info("Data Transformation Completed")
            return X_train, y_train, X_test, y_test

        except Exception as e:
            raise CustomException(e, sys)
