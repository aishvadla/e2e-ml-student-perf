import pandas as pd
import sys

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion

class DataTransformation():
    def __init__(self):
        self.data_ingestion = DataIngestion()

    def initiate_data_transformation(self):
        try:
            logging.info("Entered Data Transformation component")

            train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()

            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)

            # Choosing math_score to be the dependent variable for this test case
            y_train = train_data['math_score']
            X_train = train_data.drop(columns=['math_score'])

            y_test = test_data['math_score']
            X_test = test_data.drop(columns=['math_score'])

            # Capture the numeric and categorical (nominal and ordinal) features
            numeric_features = X_train.select_dtypes(exclude=['string', 'object']).columns.tolist()
            categorical_features = X_train.select_dtypes(include=['string', 'object']).columns.tolist()

            ordinal_features = ['parental_level_of_education']
            ordinal1_education_order = ['some high school', 'high school', 'some college', "associate's degree", "bachelor's degree", "master's degree"]

            nominal_features = [col for col in categorical_features if col not in ordinal_features]

            # Create feature transformation pipeline
            numeric_pipeline = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())])
            nominal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown="ignore", drop='first', sparse_output=True))])
            ordinal_pipeline = Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OrdinalEncoder(categories=[ordinal1_education_order], handle_unknown='use_encoded_value', unknown_value=-1))])

            preprocessing_pipeline = ColumnTransformer([('numeric', numeric_pipeline, numeric_features),
                                                        ('nominal', nominal_pipeline, nominal_features),
                                                        ('ordinal', ordinal_pipeline, ordinal_features)])
            
            X_train = preprocessing_pipeline.fit_transform(X_train)
            X_test = preprocessing_pipeline.transform(X_test)
            logging.info("Data Transformation Completed")
            return X_train, y_train, X_test, y_test
        
        except Exception as e:
            raise CustomException(e, sys)