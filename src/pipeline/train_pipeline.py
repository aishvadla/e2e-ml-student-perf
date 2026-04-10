"""Training pipeline orchestration for the E2E ML project.

This module provides end-to-end pipeline functions that sequence data ingestion,
transformation, and model training components to produce a complete training workflow.
"""

import numpy as np

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging


class TrainPipeline:
    """Orchestrates the end-to-end machine learning training workflow.

    This class encapsulates the complete training pipeline by combining
    data ingestion, preprocessing, and model training components into a
    single cohesive workflow.
    """

    def __init__(self):
        """Initialize the training pipeline with all required components.

        Creates instances of data ingestion, data transformation, and model
        trainer components that will be used throughout the pipeline.
        """
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def train(self):
        """Execute the complete end-to-end training pipeline.

        Orchestrates the full workflow in sequence:
        1. Data Ingestion — Load raw student data and split into train/test sets
        2. Data Transformation — Preprocess features (scaling, encoding, imputation)
        3. Model Training — Train and evaluate multiple regression models

        Returns:
            None

        Raises:
            CustomException: Propagated from any component if processing fails.

        Notes:
            All intermediate data and artifacts are saved to the artifacts/ directory.
            Execution details are logged to timestamped log files in logs/ directory.

        Example:
            >>> from src.pipeline.train_pipeline import TrainPipeline
            >>> pipeline = TrainPipeline()
            >>> pipeline.train()
        """
        logging.info("Starting end-to-end training pipeline")

        # Step 1: Data Ingestion
        logging.info("Initiating data ingestion...")

        train_data_path, test_data_path = self.data_ingestion.initiate_data_ingestion()
        logging.info(
            f"Data ingestion complete. Train: {train_data_path}, Test: {test_data_path}"
        )

        # Step 2: Data Transformation
        logging.info("Initiating data transformation...")
        train_array, test_array = self.data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info("Data transformation complete")

        # Step 3: Model Training
        logging.info("Initiating model training...")
        self.model_trainer.initiate_model_trainer(train_array, test_array)
        logging.info("Model training complete")

        logging.info("End-to-end training pipeline completed successfully")


if __name__ == "__main__":
    """Execute the training pipeline when run as a standalone script."""
    train_pipeline = TrainPipeline()
    train_pipeline.train()
