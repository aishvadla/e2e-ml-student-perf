"""Prediction pipeline for the E2E ML project.

This module defines the inference workflow to load preprocessors and
trained models, transform raw input data, and generate predictions.
"""

import sys

import pandas as pd

from src.exception import CustomException
from src.utils import load_object
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    """End-to-end inference pipeline using trained models and preprocessors."""

    def __init__(self):
        """Initialize the prediction pipeline with config paths."""
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def predict(self, features):
        """Run inference on raw features using trained model.

        Args:
            features: Input features DataFrame or array.

        Returns:
            array: Predicted target values.

        Raises:
            CustomException: If loading models or making predictions fails.
        """
        try:
            preprocessor_path = self.data_transformation_config.preprocessor_obj_file_path
            model_path = self.model_trainer_config.trained_model_file_path
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_preprocessed = preprocessor.transform(features)
            prediction = model.predict(data_preprocessed)
            return prediction
        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """Encapsulates raw input features for prediction.

    Provides a structured representation of student data that can be
    converted to a DataFrame for model inference.
    """

    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        """Initialize student data container.

        Args:
            gender: Student gender.
            race_ethnicity: Student race/ethnicity category.
            parental_level_of_education: Parent's education level.
            lunch: Student lunch category.
            test_preparation_course: Test preparation status.
            reading_score: Student's reading score.
            writing_score: Student's writing score.
        """
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self) -> pd.DataFrame:
        """Convert raw input data to DataFrame format.

        Returns:
            pd.DataFrame: Single-row DataFrame with all student features.

        Raises:
            CustomException: If conversion fails.
        """
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)