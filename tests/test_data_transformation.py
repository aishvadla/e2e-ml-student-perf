import os

import numpy as np

from src.components.data_transformation import DataTransformation
from src.utils import load_object


def test_initiate_data_transformation_produces_transformed_arrays(
    sample_df, sample_arrays, tmp_path
):
    train_array, test_array, transformer = sample_arrays

    assert train_array.shape[0] == len(sample_df)
    assert test_array.shape[0] == len(sample_df)
    assert train_array.shape[1] > 1
    assert test_array.shape[1] == train_array.shape[1]

    # Verify that the target was preserved in the last column
    assert np.array_equal(train_array[:, -1], sample_df["math_score"].to_numpy())
    assert os.path.exists(
        transformer.data_transformation_config.preprocessor_obj_file_path
    ), "preprocessor pkl file is not generated"

    preprocessor = load_object(
        transformer.data_transformation_config.preprocessor_obj_file_path
    )
    assert hasattr(preprocessor, "transform")


def test_get_preprocessing_pipeline_has_expected_components(
    sample_df, sample_arrays, tmp_path
):

    _, _, transformer = sample_arrays

    numeric_features = ["reading_score", "writing_score"]
    nominal_features = ["gender", "race_ethnicity", "lunch", "test_preparation_course"]
    ordinal_features = ["parental_level_of_education"]
    ordinal_order = [
        [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree",
        ]
    ]

    pipeline = transformer.preprocessing_pipeline

    assert pipeline.named_transformers_["numeric"].named_steps["scaler"] is not None
    assert pipeline.named_transformers_["nominal"].named_steps["encoder"] is not None
    assert pipeline.named_transformers_["ordinal"].named_steps["encoder"] is not None
    assert pipeline.transformers[0][2] == numeric_features
    assert pipeline.transformers[1][2] == nominal_features
    assert pipeline.transformers[2][2] == ordinal_features
