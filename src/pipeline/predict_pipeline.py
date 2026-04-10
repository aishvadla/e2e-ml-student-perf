import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig


class PredictPipeline:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()

    def predict(self, features):
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
    def __init__(
        self,
        gender: str,
        race_ethnicity: str,
        parental_level_of_education,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int,
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_data_frame(self):
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