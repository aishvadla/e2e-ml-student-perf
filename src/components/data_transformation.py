"""Data transformation component for the E2E ML project.

This module defines preprocessing pipelines used to prepare raw training and
testing data for modeling. It creates and saves a transformer that imputes
missing values, scales numeric features, one-hot encodes nominal categories,
and ordinally encodes ordered categorical features.
"""

import os
import sys

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from src.components.data_ingestion import DataIngestion
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """Prepare raw training and test data for modeling.

    This class builds preprocessing pipelines for numeric, nominal, and ordinal
    features, persists the fitted transformer, and transforms ingested datasets
    into arrays suitable for model training and evaluation.
    """

    def __init__(self):
        """Initialize the data transformation component.

        The component loads configuration and prepares the ingestion helper.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessing_pipeline(
        self, numeric_features, nominal_features, ordinal_features, ordinal_order
    ):
        """Build and save the preprocessing pipeline.

        The pipeline imputes missing values, scales numeric features, one-hot
        encodes nominal features, and ordinally encodes ordered categorical
        features.

        Parameters
        ----------
        numeric_features : list[str]
            Names of numeric feature columns.
        nominal_features : list[str]
            Names of nominal categorical feature columns.
        ordinal_features : list[str]
            Names of ordered categorical feature columns.
        ordinal_order : list[list[str]]
            Ordered categories for each ordinal feature.

        Returns
        -------
        sklearn.compose.ColumnTransformer
            The assembled preprocessing transformer.
        """
        logging.info("Creating preprocessing pipeline object")
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
                        categories=ordinal_order,
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

        save_object(
            obj=preprocessing_pipeline,
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
        )
        logging.info("Saved preprocessing pipeline object")

        return preprocessing_pipeline

    def initiate_data_transformation(self, train_data_path, test_data_path):
        """Run data ingestion and transform datasets for modeling.

        The method reads ingested train and test CSV files, separates the target
        variable, identifies numeric and categorical columns, fits the
        preprocessing pipeline on training features, and transforms both train
        and test feature sets.

        Returns:
            tuple:
                X_train_arr (numpy.ndarray): Transformed training feature matrix.
                y_train_arr (numpy.ndarray): Training target array.
                X_test_arr (numpy.ndarray): Transformed testing feature matrix.
                y_test_arr (numpy.ndarray): Testing target array.
                preprocessing_pipeline (ColumnTransformer): Fitted transformer.
        """
        try:
            logging.info("Entered Data Transformation component")

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            # Selecting the target variable for the task.
            target_column = "math_score"
            y_train_df = train_data[target_column]
            X_train_df = train_data.drop(columns=[target_column])

            y_test_df = test_data[target_column]
            X_test_df = test_data.drop(columns=[target_column])

            # Identify numeric and categorical feature columns.
            numeric_features = X_train_df.select_dtypes(
                exclude=["string", "object"]
            ).columns.tolist()
            categorical_features = X_train_df.select_dtypes(
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
            ordinal_order = [ordinal1_education_order]

            nominal_features = [
                col for col in categorical_features if col not in ordinal_features
            ]

            preprocessing_pipeline = self.get_preprocessing_pipeline(
                numeric_features, nominal_features, ordinal_features, ordinal_order
            )

            X_train_arr = preprocessing_pipeline.fit_transform(X_train_df)
            X_test_arr = preprocessing_pipeline.transform(X_test_df)

            train_arr = np.c_[X_train_arr, np.array(y_train_df)]
            test_arr = np.c_[X_test_arr, np.array(y_test_df)]

            logging.info("Data Transformation Completed")
            return (
                train_arr,
                test_arr,
                preprocessing_pipeline,
            )

        except Exception as e:
            raise CustomException(e, sys)
