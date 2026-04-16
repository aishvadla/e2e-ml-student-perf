import pytest
import pandas as pd


SAMPLE_CSV_CONTENT = """\
gender,race_ethnicity,parental_level_of_education,lunch,test_preparation_course,math_score,reading_score,writing_score
female,group B,bachelor's degree,standard,none,72,72,74
male,group C,some college,standard,completed,69,90,88
female,group B,master's degree,standard,none,90,95,93
male,group A,associate's degree,free/reduced,none,47,57,44
male,group C,some college,standard,none,76,78,75
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