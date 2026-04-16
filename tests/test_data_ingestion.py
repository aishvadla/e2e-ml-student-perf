import os

import pandas as pd

from src.components.data_ingestion import DataIngestion


def test_initiate_data_ingestion_writes_split_files(
    raw_csv, sample_df, tmp_path
):

    ingestion = DataIngestion(test_size=0.5, random_state=42)
    ingestion.ingestion_config.dataset_path = raw_csv
    ingestion.ingestion_config.raw_data_path = str(tmp_path / "raw_data.csv")
    ingestion.ingestion_config.train_data_path = str(tmp_path / "train.csv")
    ingestion.ingestion_config.test_data_path = str(tmp_path / "test.csv")

    train_path, test_path = ingestion.initiate_data_ingestion()

    assert train_path == ingestion.ingestion_config.train_data_path
    assert test_path == ingestion.ingestion_config.test_data_path
    assert os.path.exists(train_path), "train.csv was not created"
    assert os.path.exists(test_path), "test.csv was not created"
    assert os.path.exists(
        ingestion.ingestion_config.raw_data_path
    ), "raw_data.csv was not created"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    raw_df = pd.read_csv(ingestion.ingestion_config.raw_data_path)

    assert set(train_df.columns) == set(sample_df.columns)
    assert set(test_df.columns) == set(sample_df.columns)
    assert set(raw_df.columns) == set(sample_df.columns)
    assert len(train_df) > 0, "train split is empty"
    assert len(test_df) > 0, "test split is empty"
    assert len(raw_df) > 0

    assert train_df["math_score"].iloc[0] in sample_df["math_score"].values
    assert test_df["math_score"].iloc[0] in sample_df["math_score"].values
