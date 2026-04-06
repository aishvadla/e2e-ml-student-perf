"""Data ingestion component for the E2E ML project.

This module defines configuration and logic to load the raw student dataset,
persist a raw copy under the `artifacts` directory, and split the data into
train and test files for downstream model training.
"""

import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion file paths."""

    raw_data_path: str = os.path.join('artifacts', 'raw_data.csv')
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')


class DataIngestion:
    """Handles ingestion of raw data into train and test datasets."""

    def __init__(self, test_size=0.2, random_state=42):
        """Initialize the ingestion component.

        Args:
            test_size (float): Fraction of data to reserve for testing.
            random_state (int): Seed used for reproducible splitting.
        """
        self.ingestion_config = DataIngestionConfig()
        self.test_size = test_size
        self.random_state = random_state

    def initiate_data_ingestion(self):
        """Load raw data, save it, and split into train/test files.

        Returns:
            tuple[str, str]: Paths to the generated train and test CSV files.
        """
        logging.info('Entered the data ingestion component')
        try:
            dataset_path = os.path.join(os.path.dirname(__file__), '../../notebook/data/student.csv')
            df = pd.read_csv(dataset_path)
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info('Raw data saved to artifacts/raw_data.csv')

            logging.info('Train Test split initiated')
            train_set, test_set = train_test_split(
                df,
                test_size=self.test_size,
                random_state=self.random_state,
            )
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info('Data Ingestion Completed')
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise CustomException(e, sys)