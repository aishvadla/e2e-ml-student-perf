import os
import numpy as np
import pytest

from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
from src.utils import load_object


def test_initiate_model_trainer_trains_and_saves_best_model(sample_arrays, tmp_path):
    """Test that model trainer evaluates models and saves the best one."""
    train_array, test_array, _ = sample_arrays

    trainer = ModelTrainer()
    trainer.model_trainer_config.trained_model_file_path = str(tmp_path / "model.pkl")

    # This should not raise an exception and should save the model
    trainer.initiate_model_trainer(train_array, test_array)

    # Check that the model file was created
    assert os.path.exists(trainer.model_trainer_config.trained_model_file_path), "Model pkl file is not generated"

    # Load the saved model and verify it's a fitted estimator
    saved_model = load_object(trainer.model_trainer_config.trained_model_file_path)
    assert hasattr(saved_model, "predict")


def test_model_trainer_config_has_correct_path():
    """Test that ModelTrainerConfig has the expected file path."""
    config = ModelTrainerConfig()
    expected_path = os.path.join("artifacts", "model.pkl")
    assert config.trained_model_file_path == expected_path
