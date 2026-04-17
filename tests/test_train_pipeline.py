import numpy as np
from unittest.mock import patch

from src.pipeline.train_pipeline import TrainPipeline


@patch("src.pipeline.train_pipeline.ModelTrainer")
@patch("src.pipeline.train_pipeline.DataTransformation")
@patch("src.pipeline.train_pipeline.DataIngestion")
def test_train_pipeline_calls_components(
    mock_data_ingestion_cls, mock_data_transformation_cls, mock_model_trainer_cls
):
    """Verify TrainPipeline orchestrates ingestion, transformation, and training."""
    mock_ingestion = mock_data_ingestion_cls.return_value
    mock_transformation = mock_data_transformation_cls.return_value
    mock_trainer = mock_model_trainer_cls.return_value

    mock_ingestion.initiate_data_ingestion.return_value = (
        "train.csv",
        "test.csv",
    )
    train_array = np.array([[1.0, 2.0, 3.0]])
    test_array = np.array([[1.0, 2.0, 3.0]])
    mock_transformation.initiate_data_transformation.return_value = (
        train_array,
        test_array,
    )

    pipeline = TrainPipeline()
    pipeline.train()

    mock_ingestion.initiate_data_ingestion.assert_called_once()
    mock_transformation.initiate_data_transformation.assert_called_once_with(
        "train.csv",
        "test.csv",
    )
    mock_trainer.initiate_model_trainer.assert_called_once_with(
        train_array,
        test_array,
    )
