import pytest
import pandas as pd

SAMPLE_CSV_CONTENT = """\
gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
female,group B,bachelor's degree,standard,none,72,72,74
male,group C,some college,standard,completed,69,90,88
female,group B,master's degree,standard,none,90,95,93
male,group A,associate's degree,free/reduced,none,47,57,44
male,group C,some college,standard,none,76,78,75
female,group D,associate's degree,free/reduced,completed,78,77,75
male,group D,high school,standard,none,65,77,74
female,group A,some high school,free/reduced,none,83,75,90
male,group B,some high school,free/reduced,none,53,70,70
female,group C,some high school,standard,none,75,85,82
male,group E,associate's degree,standard,none,73,80,82
female,group D,some college,standard,none,88,78,75
male,group B,some college,free/reduced,none,65,73,74
female,group C,some high school,standard,none,63,96,74
male,group D,some college,standard,completed,45,87,83
female,group E,some college,free/reduced,none,71,83,78
male,group B,some high school,standard,completed,46,54,58
female,group C,some college,standard,none,91,86,84
male,group D,high school,standard,none,55,65,62
female,group B,some high school,free/reduced,none,53,59,65
"""


@pytest.fixture()
def raw_csv(tmp_path) -> str:
    """Write a small CSV that mimics the Kaggle dataset and return its path."""
    csv_path = tmp_path / "stud.csv"
    csv_path.write_text(SAMPLE_CSV_CONTENT)
    return str(csv_path)


@pytest.fixture
def sample_df(raw_csv):
    """Returns a tiny version of your raw data for testing."""
    return pd.read_csv(raw_csv)


@pytest.fixture
def sample_arrays(sample_df, tmp_path):
    """Returns transformed train and test arrays from sample data."""
    from src.components.data_transformation import DataTransformation

    train_csv = tmp_path / "train.csv"
    test_csv = tmp_path / "test.csv"

    sample_df.to_csv(train_csv, index=False)
    sample_df.to_csv(test_csv, index=False)

    transformer = DataTransformation()
    transformer.data_transformation_config.preprocessor_obj_file_path = str(
        tmp_path / "preprocessor.pkl"
    )

    train_arr, test_arr = transformer.initiate_data_transformation(
        str(train_csv), str(test_csv)
    )

    return train_arr, test_arr, transformer
